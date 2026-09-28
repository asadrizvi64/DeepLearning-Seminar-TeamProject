"""Validation-backed fitting for the sequential (greedy) ablation.

The greedy search needs a fitting routine that produces a *held-out proof*: a set of
voxels the model never sees during training, scored after the fit. If a hyperparameter
change improves the held-out (validation) metric, the improvement is real and not
memorization of the training samples.

`fit_with_validation`:
  1. Splits the volume's voxels into a train pool and a fixed validation set.
  2. Trains by sampling ONLY from the train pool (zero leakage), intensity-biased.
  3. Logs train and validation PSNR over iterations.
  4. Returns (gs, history, result) where `result` has both held-out validation metrics
     and full-volume reconstruction metrics (PSNR/SSIM), plus fit time and cost.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .gaussians import GaussianSet
from .init import init_gaussians
from .densify import build_optimizer, project_parameterization
from .losses import mse_loss
from . import metrics as M


def make_val_split(n_voxels: int, val_fraction: float, seed: int):
    """Random disjoint train/val partition of flat voxel indices (reproducible)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_voxels, generator=g)
    n_val = int(n_voxels * val_fraction)
    return perm[n_val:], perm[:n_val]      # train_idx, val_idx


def _flat_to_xyz(flat: torch.Tensor, H: int, W: int):
    z = flat // (H * W)
    rem = flat % (H * W)
    y = rem // W
    x = rem % W
    return x.float(), y.float(), z.float()


def _psnr_at(gs, xyz, targets, max_val=1.0, bias=0.0):
    x, y, z = xyz
    pts = torch.stack([x, y, z], dim=-1)
    with torch.no_grad():
        pred = gs.query_density(pts) + bias
    return M.psnr(pred, targets, max_val=max_val)


def fit_with_validation(
    volume: np.ndarray,
    num_gaussians: int,
    iterations: int = 1500,
    batch_size: int = 2048,
    init_strategy: str = 'intensity_weighted',
    parameterization: str = 'full',
    init_scale: float = 2.0,
    intensity_bias: float = 0.7,
    val_fraction: float = 0.1,
    seed: int = 0,
    val_seed: int = 1234,
    device: str = None,
    eval_every: int = 250,
    log_every: int = 100,
    val_eval_subsample: int = 20000,
    full_recon: bool = True,
    cell_metrics: bool = False,
    cell_kwargs: dict = None,
    lr_position: float = 0.0016,
    track_displacement: bool = False,
    background: str = 'none',
    background_percentile: float = 10.0,
    init_kwargs: dict = None,
    loss_mode: str = 'mse',
    log_alpha: float = 10.0,
    weight_sigma: float = 8.0,
    weight_clip: tuple = (0.25, 4.0),
    max_scale: float = None,
    max_scale_n: int = None,
) -> tuple[GaussianSet, list, dict]:
    """`full_recon=False` skips the O(voxels x Gaussians) full-volume reconstruction,
    returning only the held-out validation metrics + cost. Use it for large-k capacity
    sweeps where the dense reconstruction dominates runtime.

    `cell_metrics=True` additionally scores nucleus recovery (precision / recall /
    cell_f1 / localization RMSE) by detecting cells in the reconstruction and matching
    them to cells detected in the target -- see `volsplat.cellmetrics`. Requires
    `full_recon=True`."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)

    volume = volume.astype(np.float32)
    D, H, W = volume.shape
    n_vox = D * H * W
    volume_t = torch.from_numpy(volume).to(device)
    flat_vals = volume_t.flatten()

    # ---- split voxels into train pool / validation set
    train_idx, val_idx = make_val_split(n_vox, val_fraction, val_seed)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    # intensity weights over the TRAIN pool only (for intensity-biased sampling)
    train_w = (flat_vals[train_idx] + 1e-3)
    train_w = train_w / train_w.sum()
    n_train = train_idx.numel()

    # fixed validation coordinates + targets
    val_x, val_y, val_z = _flat_to_xyz(val_idx, H, W)
    val_targets = flat_vals[val_idx]
    # a subsample used for the per-iteration validation curve (full set used at the end)
    if val_idx.numel() > val_eval_subsample:
        sub = torch.randperm(val_idx.numel(), device=device)[:val_eval_subsample]
        vc_x, vc_y, vc_z = val_x[sub], val_y[sub], val_z[sub]
        vc_t = val_targets[sub]
    else:
        vc_x, vc_y, vc_z, vc_t = val_x, val_y, val_z, val_targets

    # ---- explicit DC background.
    #
    # Without any background handling, ~87% of the Gaussian slots approximate a
    # near-constant pedestal that a single scalar reproduces at least as well, and MSE
    # is dominated by pedestal error rather than by structure.
    #
    # 'constant' -- LEARNABLE b added to the prediction. NON-IDENTIFIABLE and kept only
    #   to reproduce that finding: with 250 broad Gaussians, sum_i a_i G_i can itself
    #   approximate a constant, so raising b and lowering amplitudes leaves the
    #   prediction nearly unchanged. The gradient along that direction is ~0, so the
    #   split is decided by optimizer dynamics rather than by the loss. Measured: b was
    #   initialised at the 10th percentile (0.544) and driven DOWN to 0.197 while the
    #   support population barely moved (219 -> 217). The pedestal was never removed.
    #
    # 'residual' -- b FIXED, and the TARGET becomes R(x) = max(I(x) - b, 0). The
    #   Gaussians fit R directly; there is no b in the optimisation and therefore no
    #   degenerate direction. This genuinely removes the DC competition from the loss,
    #   which is what the pedestal hypothesis actually claims. For evaluation the
    #   reconstruction is b + sum_i a_i G_i, compared against the ORIGINAL volume, so
    #   every metric stays directly comparable to background='none'.
    #
    # b is a LOW percentile, not the mean, so R is almost everywhere non-negative --
    # amplitudes are positive by construction (softplus) and cannot represent a dip
    # below background. The percentile is pre-registered at 10 and must NOT be tuned:
    # the causal question is whether removing DC competition rescues faint nuclei, not
    # which percentile maximises F1.
    b0 = 0.0
    fit_flat = flat_vals                      # what the loss is computed against
    if background == 'constant':
        b0 = float(np.percentile(volume, background_percentile))
        bg = torch.tensor(b0, device=device, requires_grad=True)
        fit_volume = np.clip(volume - b0, 0.0, None)
    elif background == 'residual':
        b0 = float(np.percentile(volume, background_percentile))
        bg = None
        fit_volume = np.clip(volume - b0, 0.0, None)
        fit_flat = torch.from_numpy(fit_volume).to(device).flatten()
    elif background == 'none':
        bg = None
        fit_volume = volume
    else:
        raise ValueError(f"unknown background mode {background!r}")

    # ---- model + optimizer
    # `init_kwargs` is forwarded to the chosen strategy -- needed when the caller wants
    # to seed from its OWN candidate list (e.g. the same detector output a baseline is
    # scored against) rather than let the strategy run its own detector with its own,
    # possibly voxel-space, settings.
    gs = init_gaussians(fit_volume, num_gaussians, strategy=init_strategy,
                        init_scale=init_scale, seed=seed,
                        **(init_kwargs or {})).to(device)
    project_parameterization(gs, parameterization)
    optimizer = build_optimizer(gs, parameterization=parameterization,
                                lr_position=lr_position)
    init_positions = gs.positions.detach().clone() if track_displacement else None
    if bg is not None:
        optimizer.add_param_group({'params': [bg], 'lr': 0.01, 'name': 'background'})

    def add_bg(x):
        """Offset applied when comparing against the ORIGINAL volume.

        'constant': b is a live parameter. 'residual': b is the fixed scalar that was
        subtracted from the target. 'none': zero.
        """
        if bg is not None:
            return x + bg
        return x + b0 if background == 'residual' else x

    # ---- loss weighting.
    #
    # Under plain voxelwise MSE a faint nucleus contributes almost nothing: at contrast
    # 0.03 a total miss costs ~9e-4 per voxel over ~5% of the volume, while a 0.05 error
    # on the bright bulk costs ~2.5e-3 per voxel over ~95%. These modes test whether that
    # weighting is what sacrifices low-contrast nuclei, WITHOUT touching the data, the
    # initialization, the representation, or the background.
    #
    #   'mse'               control
    #   'log'               || log(1+aI) - log(1+aI_hat) ||^2 -- compresses the bright
    #                       end so dim structure is not drowned out
    #   'contrast_weighted' w(x)*(I-I_hat)^2 with w ~ 1/local_std, so locally salient
    #                       structure is equalized regardless of absolute brightness
    #
    # w is derived ONCE from the target and held fixed; it never enters the model. It is
    # normalized to unit median and CLIPPED -- an unbounded 1/sigma hands near-flat
    # regions absurd weight and would manufacture a different artefact.
    loss_w = None
    if loss_mode == 'contrast_weighted':
        from scipy.ndimage import gaussian_filter
        m = gaussian_filter(volume, sigma=weight_sigma)
        v = gaussian_filter(volume ** 2, sigma=weight_sigma) - m ** 2
        local_std = np.sqrt(np.clip(v, 0.0, None))
        w = 1.0 / (local_std + 1e-3)
        w = w / float(np.median(w))
        w = np.clip(w, weight_clip[0], weight_clip[1]).astype(np.float32)
        loss_w = torch.from_numpy(w).to(device).flatten()
    elif loss_mode not in ('mse', 'log'):
        raise ValueError(f'unknown loss_mode {loss_mode!r}')

    def compute_loss(pred, targets, sel):
        if loss_mode == 'log':
            return mse_loss(torch.log1p(log_alpha * pred.clamp_min(0)),
                            torch.log1p(log_alpha * targets.clamp_min(0)))
        if loss_mode == 'contrast_weighted':
            return (loss_w[sel] * (pred - targets) ** 2).mean()
        return mse_loss(pred, targets)

    n_int = int(batch_size * intensity_bias)
    n_uni = batch_size - n_int

    history = []
    t0 = time.time()
    for it in range(iterations):
        # --- sample a training batch from the TRAIN POOL ONLY (zero val leakage)
        # intensity_bias=0 (pure uniform) must skip multinomial, which rejects a
        # zero-sample request; for n_int > 0 the random stream is unchanged.
        sel_int = (train_idx[torch.multinomial(train_w, n_int, replacement=True)]
                   if n_int > 0 else train_idx[:0])
        sel_uni = train_idx[torch.randint(0, n_train, (n_uni,), device=device)]
        sel = torch.cat([sel_int, sel_uni])
        bx, by, bz = _flat_to_xyz(sel, H, W)
        jitter = torch.rand(sel.numel(), 3, device=device) - 0.5
        pts = torch.stack([bx, by, bz], dim=-1) + jitter
        # 'residual' trains against R = max(I - b, 0); the other modes train against I.
        targets = fit_flat[sel]

        # b is added during TRAINING only when it is a live parameter ('constant').
        # Under 'residual' the pedestal is already absent from the target, so adding it
        # back here would reinstate exactly the DC competition the mode removes.
        pred = gs.query_density(pts)
        if bg is not None:
            pred = pred + bg
        loss = compute_loss(pred, targets, sel)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        project_parameterization(gs, parameterization)
        if max_scale is not None:
            # Cap spatial extent. Trajectory analysis: Gaussians on LOST faint nuclei
            # broaden to 6.2x init scale (vs 4.5x kept), ending as wide low-peak blobs.
            # `max_scale_n` restricts the cap to the first N Gaussians -- with the
            # coverage initializer those are exactly the nucleus-seeded ones, so SUPPORT
            # Gaussians stay free to model broad tissue. A GLOBAL cap is confounded:
            # it pins the support population too and costs ~9.6 dB by itself.
            # UNITS: `max_scale` is in VOXELS and the same limit is applied to all three
            # axes, so on anisotropic data it is a different physical length per axis
            # (Tribolium cap 10 = 6.9 um in xy but 30 um in z -- effectively an xy-only
            # cap). Transferring it to another dataset means matching the xy extent in
            # microns, not reusing the voxel number.
            with torch.no_grad():
                lim = float(np.log(max_scale))
                if max_scale_n is None:
                    gs.log_scales.clamp_(max=lim)
                else:
                    gs.log_scales.data[:max_scale_n].clamp_(max=lim)

        if it % log_every == 0 or it == iterations - 1:
            train_p = M.psnr(pred, targets)
            history.append({'iter': it, 'train_psnr': float(train_p)})
        if eval_every > 0 and (it % eval_every == 0 or it == iterations - 1):
            # vc_t holds ORIGINAL intensities, so the offset must be added back for
            # both 'constant' (live b) and 'residual' (fixed b0).
            val_p = _psnr_at(gs, (vc_x, vc_y, vc_z), vc_t,
                             bias=float(bg.detach()) if bg is not None else b0)
            history.append({'iter': it, 'val_psnr': float(val_p)})

    fit_seconds = time.time() - t0

    # ---- final metrics
    # held-out validation (the proof): scatter metrics on the full val set
    val_pts = torch.stack([val_x, val_y, val_z], dim=-1)
    with torch.no_grad():
        val_pred = add_bg(gs.query_density(val_pts)).cpu().numpy()
    val_tgt_np = val_targets.cpu().numpy()
    result = {
        'val_psnr': M.psnr(val_pred, val_tgt_np),
        'val_mse':  M.mse(val_pred, val_tgt_np),
        'val_mae':  M.mae(val_pred, val_tgt_np),
        'val_corr': M.pearson_corr(val_pred, val_tgt_np),
    }

    if init_positions is not None:
        disp = torch.norm(gs.positions.detach() - init_positions, dim=1)
        result['disp_mean'] = float(disp.mean())
        result['disp_p90'] = float(torch.quantile(disp, 0.9))
        result['disp_max'] = float(disp.max())

    # ---- MANIPULATION CHECK. Verifies the intervention actually removed DC from the
    # problem the Gaussians were asked to solve, rather than merely perturbing the
    # optimisation. If `fit_target_energy` is not materially below `orig_energy`, any
    # downstream change in S_A / S_B / support count cannot be attributed to removing
    # background competition.
    result['loss_mode'] = loss_mode
    result['background_mode'] = background
    result['background_b'] = float(bg.detach()) if bg is not None else b0
    result['max_scale'] = max_scale
    result['max_scale_n'] = max_scale_n
    result['orig_mean'] = float(volume.mean())
    result['orig_energy'] = float((volume ** 2).mean())
    result['fit_target_mean'] = float(fit_volume.mean())
    result['fit_target_energy'] = float((fit_volume ** 2).mean())
    result['dc_energy_removed'] = 1.0 - result['fit_target_energy'] / max(
        result['orig_energy'], 1e-12)

    # cost metrics are always available (no reconstruction needed)
    n_params = M.param_count(num_gaussians, parameterization)
    result.update({
        'num_params': n_params,
        'bytes': n_params * 4,
        'compression': volume.size / max(n_params, 1),
        'fit_seconds': fit_seconds,
        'background_value': None if bg is None else float(bg.detach()),
    })

    if full_recon:
        # full-volume reconstruction (quality story; includes val voxels)
        # The offset must be restored for BOTH background modes before comparing against
        # the original volume: 'constant' carries a live b, 'residual' a fixed b0 that
        # was subtracted from the target. Omitting it for 'residual' scores the residual
        # against the full-intensity volume and reports a spuriously catastrophic PSNR.
        recon = gs.query_volume(volume.shape).cpu().numpy().astype(np.float32)
        recon = recon + (float(bg.detach()) if bg is not None else b0)
        full = M.compute_all(recon, volume, num_gaussians, parameterization,
                             fit_seconds=fit_seconds)
        result.update({
            'full_psnr': full['psnr'], 'full_ssim': full['ssim'],
            'full_mae': full['mae'], 'full_corr': full['corr'],
            'psnr_per_kparam': full['psnr_per_kparam'],
        })
        if cell_metrics:
            from .cellmetrics import score_cells
            result.update(score_cells(recon, volume, **(cell_kwargs or {})))
    return gs, history, result

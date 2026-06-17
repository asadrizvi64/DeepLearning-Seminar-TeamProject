"""

Supervision strategy
--------------------
At each iteration:
  1. Sample a batch of voxel coordinates (intensity-biased + uniform mixture).
  2. Evaluate the predicted density field at those exact points.
  3. MSE against the ground-truth voxel values, backprop, step.

"""
import copy
import time

import numpy as np
import torch
from tqdm import tqdm

from .gaussians import GaussianSet
from .losses import mse_loss, psnr, mae, ssim3d, ssim3d_slicewise
from .densify import densify, build_optimizer
from .data import intensity_weighted_sample
from .init import init_gaussians


# ------------------------------------------------------------------ initialization

def init_gaussians_from_volume(
    volume: np.ndarray,
    num_gaussians: int,
    init_scale=2.0,
    seed: int = 0,
) -> GaussianSet:
    """Backward-compatible intensity-weighted initialization.
    """
    return init_gaussians(
        volume, num_gaussians,
        strategy='intensity_weighted',
        init_scale=init_scale,
        seed=seed,
    )


# ------------------------------------------------------------------ batch sampling

def sample_training_points(
    volume_t: torch.Tensor,
    batch_size: int,
    intensity_bias: float = 0.7,
    eps: float = 1e-3,
):
    """Mixture sampling: `intensity_bias` fraction intensity-weighted, rest uniform.

    Balances structure-focused learning with coverage of empty space (so the model
    learns to be near zero there).
    """
    D, H, W = volume_t.shape
    device = volume_t.device
    n_int = int(batch_size * intensity_bias)
    n_uni = batch_size - n_int

    # Intensity-weighted
    flat = volume_t.flatten() + eps
    flat = flat / flat.sum()
    idx_int = torch.multinomial(flat, n_int, replacement=True)
    z_i = idx_int // (H * W)
    rem = idx_int % (H * W)
    y_i = rem // W
    x_i = rem % W

    # Uniform
    x_u = torch.randint(0, W, (n_uni,), device=device)
    y_u = torch.randint(0, H, (n_uni,), device=device)
    z_u = torch.randint(0, D, (n_uni,), device=device)

    x = torch.cat([x_i, x_u]).float()
    y = torch.cat([y_i, y_u]).float()
    z = torch.cat([z_i, z_u]).float()

    # Sub-voxel jitter so we supervise the continuous field, not the discrete grid
    jitter = torch.rand(batch_size, 3, device=device) - 0.5
    positions = torch.stack([x, y, z], dim=-1) + jitter

    # Target = voxel intensity at the (integer) sample location
    targets = volume_t[
        z.long().clamp(0, D - 1),
        y.long().clamp(0, H - 1),
        x.long().clamp(0, W - 1),
    ]
    return positions, targets


# ------------------------------------------------------------------ training loop

def train_static(
    volume: np.ndarray,
    num_gaussians: int = 2000,
    iterations: int = 3000,
    batch_size: int = 2048,
    densify_every: int = 0,        # 0 = no densification
    densify_start: int = 500,
    densify_end: int = 4000,
    grad_threshold: float = 2e-4,
    log_every: int = 100,
    eval_every: int = 1000,
    device: str = None,
    seed: int = 0,
    init_scale=2.0,
    init_strategy: str = 'intensity_weighted',
    # ---- convergence control (E3 addition; fully backward-compatible) --------
    patience: int = 0,
    min_delta: float = 0.05,
    stop_metric: str = 'psnr',
):
    """Fit a GaussianSet to a static 3D volume.

    Returns (gs, history). `history` is a list of dicts; entries vary in shape:
      - per-step log:     {'iter', 'loss', 'psnr', 'num_gaussians'}
      - densify event:    {'iter', 'densify': stats_dict}
      - full-volume eval: {'iter', 'full_psnr'}
      - final summary:    {'elapsed_seconds', 'final_count', 'final_iter',
                           'converged', 'best_full_psnr', 'best_full_ssim',
                           'best_iteration', 'convergence_metric',
                           'best_metric_value', 'stagnant_evaluations',
                           'total_evaluations'}

    Convergence parameters
    ----------------------
    patience : int
        Number of consecutive full-volume evaluations that must fail to improve
        `stop_metric` by `min_delta` before training stops early.
        0 (default) disables early stopping — reproduces the original behaviour.

    min_delta : float
        Minimum improvement in the stop_metric that counts as "real" progress.
        For PSNR (dB): 0.05 is conservative; for SSIM (0–1): 0.002 is typical.
        Default 0.05 is tuned for PSNR.

    stop_metric : str  {'psnr', 'ssim'}
        Which metric drives the patience / convergence decision.

        'psnr' (default): fast — no extra compute per eval checkpoint.

        'ssim': accurate — detects when structural fidelity plateaus even if
        per-voxel MSE still improves. Requires a full voxelization
        (gs.query_volume) at *every* eval checkpoint, which is O(D·H·W·G).
        For large budgets (G=20k) on a 64×128×128 ROI with eval_every=200 and
        8000 max iterations, expect ~40 SSIM evaluations × 10–30s each.
        Only use 'ssim' on small volumes or with large eval_every.

    Best-checkpoint restore
    -----------------------
    Whenever a new best value of `stop_metric` is observed during training,
    the model state_dict is saved. After the loop, the model is automatically
    restored to that best checkpoint before returning. This means:
      - The returned `gs` reflects the *best* quality seen, not the last step.
      - evaluate_metrics() called on the returned gs measures the best checkpoint.
      - If eval_every=0 (no checkpoints), no restore is performed.

    When patience=0 (default):
        converged = None   (convergence not monitored)
        final_iter = iterations - 1
        best-checkpoint restore still applies if eval_every > 0.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    volume_t = torch.from_numpy(volume.astype(np.float32)).to(device)
    gs = init_gaussians(
        volume, num_gaussians, strategy=init_strategy,
        init_scale=init_scale, seed=seed,
    ).to(device)
    optimizer = build_optimizer(gs)

    grad_accum = torch.zeros_like(gs.positions)
    grad_count = 0

    # -- convergence & best-checkpoint state -----------------------------------
    # best_* tracking is always active (when eval_every > 0) so we can restore
    # the best checkpoint before returning, independent of patience setting.
    best_psnr_seen          = -float('inf')  # highest PSNR ever observed (any eval)
    best_psnr_at_checkpoint = None           # PSNR AT the saved best_state_dict
    best_ssim_seen          = -float('inf')  # populated only when stop_metric='ssim'
    best_monitor            = -float('inf')  # tracks stop_metric value for patience
    best_iter               = 0
    best_state_dict         = None           # saved on every new best; restored at end
    total_evals             = 0
    stagnant_evals          = 0
    converged       = False
    final_iter      = iterations - 1

    history = []
    t0 = time.time()
    pbar = tqdm(range(iterations), desc='fit')

    for it in pbar:
        positions, targets = sample_training_points(volume_t, batch_size)
        pred = gs.query_density(positions)
        loss = mse_loss(pred, targets)

        optimizer.zero_grad()
        loss.backward()

        # Running mean of |grad| on positions for densification triggering
        if gs.positions.grad is not None:
            grad_accum = grad_accum * (grad_count / (grad_count + 1.0)) + \
                         gs.positions.grad.abs() / (grad_count + 1.0)
            grad_count += 1

        optimizer.step()

        if it % log_every == 0:
            with torch.no_grad():
                p = psnr(pred, targets)
            pbar.set_postfix(loss=f'{loss.item():.4e}', psnr=f'{p:.2f}', N=gs.num_gaussians)
            history.append({
                'iter': it, 'loss': float(loss.item()),
                'psnr': float(p), 'num_gaussians': gs.num_gaussians,
            })

        if (densify_every > 0
            and densify_start <= it <= densify_end
            and it > 0 and it % densify_every == 0):
            stats = densify(gs, grad_accum, grad_threshold=grad_threshold)
            optimizer = build_optimizer(gs)
            grad_accum = torch.zeros_like(gs.positions)
            grad_count = 0
            history.append({'iter': it, 'densify': stats})
            pbar.write(f'iter {it}: densify {stats}')

        if eval_every > 0 and (it + 1) % eval_every == 0:
            full_p = evaluate_full(gs, volume_t)
            history.append({'iter': it, 'full_psnr': float(full_p)})
            total_evals += 1

            # -- track best PSNR (always, regardless of stop_metric) ----------
            if full_p > best_psnr_seen:
                best_psnr_seen = full_p
                if stop_metric == 'psnr':
                    best_iter = it
                    best_state_dict = copy.deepcopy(gs.state_dict())
                    best_psnr_at_checkpoint = full_p

            # -- get monitor metric (PSNR fast path; SSIM expensive path) -----
            if stop_metric == 'ssim':
                monitor_val = evaluate_ssim(gs, volume_t)
                ssim_str = f'  SSIM={monitor_val:.4f}'
                if monitor_val > best_ssim_seen:
                    best_ssim_seen = monitor_val
                    best_iter = it
                    best_state_dict = copy.deepcopy(gs.state_dict())
                    # full_p was just computed for THIS iteration, so it is the
                    # correct PSNR value for the checkpoint being saved here —
                    # not best_psnr_seen, which may belong to a different,
                    # unsaved iteration (see evaluate_metrics bug analysis).
                    best_psnr_at_checkpoint = full_p
            else:
                monitor_val = full_p
                ssim_str = ''

            pbar.write(f'iter {it}: full-vol PSNR={full_p:.2f} dB{ssim_str}')

            # -- patience-based early stopping (no-op when patience == 0) -----
            if patience > 0:
                if monitor_val > best_monitor + min_delta:
                    best_monitor = monitor_val
                    stagnant_evals = 0
                else:
                    stagnant_evals += 1

                if stagnant_evals >= patience:
                    converged = True
                    final_iter = it
                    pbar.write(
                        f'  ↳ converged: best {stop_metric.upper()}='
                        f'{best_monitor:.4f}, '
                        f'{stagnant_evals} evals without >{min_delta} gain'
                    )
                    break

    # -- restore best checkpoint -----------------------------------------------
    # If any eval checkpoint ran, gs is now the best model seen during training
    # rather than the last-iteration state. This is the scientifically preferred
    # evaluation basis: it guards against late-training oscillation or overshoot.
    if best_state_dict is not None:
        gs.load_state_dict(best_state_dict)

    history.append({
        'elapsed_seconds':      time.time() - t0,
        'final_count':          gs.num_gaussians,
        'final_iter':           final_iter,
        'converged':            converged if patience > 0 else None,
        
        'best_full_psnr':       best_psnr_at_checkpoint,
        'best_full_ssim':       float(best_ssim_seen) if best_ssim_seen > -float('inf') else None,
        'best_iteration':       int(best_iter),
        
        'highest_psnr_observed': float(best_psnr_seen) if best_psnr_seen > -float('inf') else None,
        
        'convergence_metric':   stop_metric,
        'best_metric_value':    float(best_monitor) if best_monitor > -float('inf') else None,
        'stagnant_evaluations': int(stagnant_evals),
        'total_evaluations':    int(total_evals),
    })
    return gs, history


# ----------------------------------------------- projection-supervised training (P2)

def train_static_projection(
    volume: np.ndarray,
    num_gaussians: int = 2000,
    iterations: int = 3000,
    batch_size: int = None,        # unused; kept for API symmetry with train_static
    axes=('z', 'y', 'x'),
    densify_every: int = 0,
    densify_start: int = 500,
    densify_end: int = 4000,
    grad_threshold: float = 2e-4,
    log_every: int = 100,
    eval_every: int = 1000,
    device: str = None,
    seed: int = 0,
    init_scale=2.0,
    pixel_chunk: int = 4096,
    init_strategy: str = 'intensity_weighted',
):
   
    from .rasterize import project_alpha_blend, gt_projection

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    volume_t = torch.from_numpy(volume.astype(np.float32)).to(device)
    gs = init_gaussians(
        volume, num_gaussians, strategy=init_strategy,
        init_scale=init_scale, seed=seed,
    ).to(device)
    optimizer = build_optimizer(gs)

    # GT projections are static; compute once.
    gt_imgs = {a: gt_projection(volume_t, a) for a in axes}
    gt_max = {a: float(gt_imgs[a].max().item()) for a in axes}

    grad_accum = torch.zeros_like(gs.positions)
    grad_count = 0

    history = []
    t0 = time.time()
    pbar = tqdm(range(iterations), desc='fit-proj')

    for it in pbar:
        axis = axes[it % len(axes)]
        pred_img = project_alpha_blend(
            gs, axis, volume.shape, pixel_chunk=pixel_chunk
        )
        loss = mse_loss(pred_img, gt_imgs[axis])

        optimizer.zero_grad()
        loss.backward()
        if gs.positions.grad is not None:
            grad_accum = grad_accum * (grad_count / (grad_count + 1.0)) + \
                         gs.positions.grad.abs() / (grad_count + 1.0)
            grad_count += 1
        optimizer.step()

        if it % log_every == 0:
            with torch.no_grad():
                # Use the per-axis GT max so PSNR is meaningful for projection targets.
                p_proj = psnr(pred_img, gt_imgs[axis], max_val=max(gt_max[axis], 1e-8))
            pbar.set_postfix(loss=f'{loss.item():.4e}', proj_psnr=f'{p_proj:.2f}',
                             N=gs.num_gaussians, axis=axis)
            history.append({
                'iter': it, 'axis': axis, 'loss': float(loss.item()),
                'proj_psnr': float(p_proj), 'num_gaussians': gs.num_gaussians,
            })

        if (densify_every > 0 and densify_start <= it <= densify_end
                and it > 0 and it % densify_every == 0):
            stats = densify(gs, grad_accum, grad_threshold=grad_threshold)
            optimizer = build_optimizer(gs)
            grad_accum = torch.zeros_like(gs.positions)
            grad_count = 0
            history.append({'iter': it, 'densify': stats})
            pbar.write(f'iter {it}: densify {stats}')

        if eval_every > 0 and (it + 1) % eval_every == 0:
            full_p = evaluate_full(gs, volume_t)
            history.append({'iter': it, 'full_psnr': float(full_p)})
            pbar.write(f'iter {it}: full-volume (density) PSNR = {full_p:.2f}')

    history.append({
        'elapsed_seconds': time.time() - t0,
        'final_count': gs.num_gaussians,
        'final_iter':  iterations - 1,
        'converged':   None,
    })
    return gs, history


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def evaluate_full(gs: GaussianSet, volume_t: torch.Tensor, subsample: int = 50_000) -> float:
    """PSNR on a uniform random voxel subsample over the whole volume.

    Kept for backward-compatibility with E1/E2 and all existing scripts.
    """
    D, H, W = volume_t.shape
    device = volume_t.device
    n = min(subsample, D * H * W)
    x = torch.randint(0, W, (n,), device=device).float()
    y = torch.randint(0, H, (n,), device=device).float()
    z = torch.randint(0, D, (n,), device=device).float()
    positions = torch.stack([x, y, z], dim=-1)
    targets = volume_t[z.long(), y.long(), x.long()]
    pred = gs.query_density(positions)
    return psnr(pred, targets)


@torch.no_grad()
def evaluate_metrics(
    gs: GaussianSet,
    volume_t: torch.Tensor,
    subsample: int = 100_000,
    seed: int = 42,
) -> dict:
    """Compute PSNR and MAE on *identical* voxel coordinates.

    Using a fixed generator ensures:
      - PSNR and MAE are computed on the same voxels within one evaluation call.
      - The same voxels are used across all budget levels (seed=42 by default),
        so per-budget comparisons are on identical test sets.

    Parameters
    ----------
    gs        : trained GaussianSet
    volume_t  : (D, H, W) ground-truth volume tensor on the same device as gs
    subsample : number of voxels to sample (100k balances accuracy and speed)
    seed      : seed for the sampling generator; hold constant across all budgets

    """
    D, H, W = volume_t.shape
    device = volume_t.device
    n = min(subsample, D * H * W)

    # Isolated generator — does not touch the global RNG state
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.randint(0, W, (n,), device=device, generator=gen).float()
    y = torch.randint(0, H, (n,), device=device, generator=gen).float()
    z = torch.randint(0, D, (n,), device=device, generator=gen).float()

    positions = torch.stack([x, y, z], dim=-1)
    targets   = volume_t[z.long(), y.long(), x.long()]
    pred      = gs.query_density(positions)

    return {
        'psnr': psnr(pred, targets),
        'mae':  mae(pred, targets),
    }


@torch.no_grad()
def evaluate_ssim(
    gs: GaussianSet,
    volume_t: torch.Tensor,
    win_size: int = 7,
    max_voxels_3d: int = 5_000_000,
) -> float:
    """3D SSIM via full voxelization of the Gaussian field.
    """
    D, H, W = volume_t.shape
    n_voxels = D * H * W

    # Voxelize the predicted field — necessary for any spatial metric
    pred_vol   = gs.query_volume((D, H, W)).cpu().numpy().astype(np.float32)
    target_vol = volume_t.cpu().numpy().astype(np.float32)

    try:
        if n_voxels <= max_voxels_3d:
            return ssim3d(pred_vol, target_vol, data_range=1.0, win_size=win_size)
        else:
            print(
                f'  [ssim] volume has {n_voxels:,} voxels > {max_voxels_3d:,} threshold; '
                f'falling back to slice-wise 2D SSIM (averaged over {D} z-slices)'
            )
            return ssim3d_slicewise(pred_vol, target_vol, data_range=1.0, win_size=win_size)
    except Exception as e:
        print(f'  [ssim] computation failed: {e}. Returning -2.0')
        return -2.0
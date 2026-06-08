"""Static density-fitting training loop for Phase 1.

Supervision strategy
--------------------
At each iteration:
  1. Sample a batch of voxel coordinates (intensity-biased + uniform mixture).
  2. Evaluate the predicted density field at those exact points.
  3. MSE against the ground-truth voxel values, backprop, step.

This is the simplest defensible "render-and-compare" for v1. Phase 2 will add
(a) a slab-projection rasterizer (true 2D render-and-compare) and (b) a differentiable
voxelizer (bias-free reference), then run E1 to compare them.
"""
import time

import numpy as np
import torch
from tqdm import tqdm

from .gaussians import GaussianSet
from .losses import mse_loss, psnr, mae
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

    Kept so existing scripts keep working. New code should call `init_gaussians(...)`
    with an explicit `strategy` instead.
    In theory, you can initialize as written in the previous version.
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
):
    """Fit a GaussianSet to a static 3D volume.

    Returns (gs, history). `history` is a list of dicts; entries vary in shape (log,
    densify event, full-volume eval, final summary).

    `init_strategy` selects among the strategies in `volsplat.init.INIT_STRATEGIES`:
    'random', 'intensity_weighted' (P1 default), or 'local_maxima' (E2).
    Again though, the line intensity weighted could be removed, as in previous version.
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
            pbar.write(f'iter {it}: full-volume PSNR = {full_p:.2f}')

    history.append({
        'elapsed_seconds': time.time() - t0,
        'final_count': gs.num_gaussians,
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
    """Fit a GaussianSet via orthographic alpha-blending projection supervision.

    The "render-and-compare" path for Phase 2 / E1. Each iteration:
      1. Pick `axis = axes[it % len(axes)]`.
      2. Render an alpha-blended sum-projection of the Gaussian set along that axis.
      3. MSE against the GT sum-projection along the same axis.
      4. Backprop, step.

    Same init / densification / eval as `train_static`, so a head-to-head comparison
    is fair. The PSNR logged during training is on the *projection* targets (which
    can have a very different dynamic range from voxels) - the apples-to-apples
    comparison to voxel-query training is `evaluate_full` on the density field.
    """
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
    })
    return gs, history


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def evaluate_full(gs: GaussianSet, volume_t: torch.Tensor, subsample: int = 50_000) -> float:
    """PSNR on a uniform random voxel subsample over the whole volume."""
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
def evaluate_mae(gs: GaussianSet, volume_t: torch.Tensor, subsample: int = 50_000) -> float:
    """MAE on a uniform random voxel subsample over the whole volume.

    Added for E3 (or whatever is in that matrix thingy XD) as the second reconstruction quality metric.

    MAE is complementary to PSNR: it is in native intensity units [0, 1] and
    is not log-scaled, so it makes the absolute per-voxel error legible, and it
    is less sensitive to rare large outliers than PSNR's squared-error base.
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
    return mae(pred, targets)

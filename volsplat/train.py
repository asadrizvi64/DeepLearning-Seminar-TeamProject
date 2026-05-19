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
from .losses import mse_loss, psnr
from .densify import densify, build_optimizer
from .data import intensity_weighted_sample


# ------------------------------------------------------------------ initialization

def init_gaussians_from_volume(
    volume: np.ndarray,
    num_gaussians: int,
    init_scale=2.0,
    seed: int = 0,
) -> GaussianSet:
    """Intensity-weighted initialization.

    Positions sampled proportional to intensity; amplitude set to the local intensity;
    identity rotation.

    `init_scale` is either a scalar (isotropic) or a 3-tuple `(sx, sy, sz)` (anisotropic,
    per-axis in voxel units). For anisotropic-voxel data (e.g. light-sheet with 5x z
    anisotropy), pass something like `(2.0, 2.0, 0.4)` so the initial Gaussians are
    roughly isotropic in physical units.
    """
    positions = intensity_weighted_sample(volume, num_gaussians, seed=seed)

    # Nearest-neighbor intensity lookup at each position. positions are (x, y, z),
    # volume is (z, y, x).
    D, H, W = volume.shape
    ix = np.clip(positions[:, 0].round().astype(np.int64), 0, W - 1)
    iy = np.clip(positions[:, 1].round().astype(np.int64), 0, H - 1)
    iz = np.clip(positions[:, 2].round().astype(np.int64), 0, D - 1)
    intensities = volume[iz, iy, ix]
    amplitudes = np.clip(intensities, 0.05, None).astype(np.float32)

    if np.isscalar(init_scale):
        scales = np.full((num_gaussians, 3), float(init_scale), dtype=np.float32)
    else:
        s = np.asarray(init_scale, dtype=np.float32).reshape(3)
        scales = np.tile(s[None, :], (num_gaussians, 1))
    quats = np.zeros((num_gaussians, 4), dtype=np.float32)
    quats[:, 0] = 1.0

    return GaussianSet(
        positions=torch.from_numpy(positions),
        scales=torch.from_numpy(scales),
        quaternions=torch.from_numpy(quats),
        amplitudes=torch.from_numpy(amplitudes),
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
):
    """Fit a GaussianSet to a static 3D volume.

    Returns (gs, history). `history` is a list of dicts; entries vary in shape (log,
    densify event, full-volume eval, final summary).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    volume_t = torch.from_numpy(volume.astype(np.float32)).to(device)
    gs = init_gaussians_from_volume(volume, num_gaussians, init_scale=init_scale, seed=seed).to(device)
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

"""Integration-bias diagnostic for Phase 2 / E1.

Given a trained GaussianSet and the ground-truth volume, measure:
  * Voxel-wise MAE, RMSE, and signed mean error (pred - target).
  * The same metrics stratified by Gaussian-overlap count - how many Gaussians
    contribute non-trivially at each sampled point. This is the diagnostic for
    the R^2-Gaussian alpha-blending bias.

Hypothesis: a model fit with alpha-blending projection supervision will show
positive signed mean (it overshoots) and the overshoot will be larger in
regions of higher Gaussian overlap. A model fit with voxel-query supervision
should show ~zero signed mean and no overlap dependency.

The closed-form prediction (verified in the single-Gaussian rasterizer test) is
that the alpha-blend model inflates amplitudes by ~sqrt(2*pi)*sigma. So the
signed bias in isolated regions (overlap = 1) should already be positive at
roughly that scale; overlap > 1 should make it worse if the under-counting at
overlap interacts with the inflation.
"""
import numpy as np
import torch

from .gaussians import GaussianSet


@torch.no_grad()
def overlap_count(
    gs: GaussianSet,
    points: torch.Tensor,
    threshold: float = 0.1,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """For each point, count Gaussians whose normalized contribution > threshold.

    Normalized contribution = exp(-0.5 * mahalanobis_squared) - i.e. relative to
    the Gaussian's own peak, independent of amplitude. A point inside the
    "> threshold" region of N Gaussians has overlap count N.
    """
    mu = gs.positions
    R = gs.rotations
    scales = gs.scales
    P = points.shape[0]
    counts = torch.zeros(P, device=points.device, dtype=torch.int32)
    for s in range(0, P, chunk_size):
        e = min(s + chunk_size, P)
        diff = points[s:e].unsqueeze(1) - mu.unsqueeze(0)         # (Pc, G, 3)
        local = torch.einsum('gji,pgj->pgi', R, diff)
        normalized = local / scales.unsqueeze(0)
        quad = (normalized ** 2).sum(-1)                          # (Pc, G)
        gauss = torch.exp(-0.5 * quad)
        counts[s:e] = (gauss > threshold).sum(-1).to(torch.int32)
    return counts


@torch.no_grad()
def bias_diagnostic(
    gs: GaussianSet,
    volume_t: torch.Tensor,
    n_samples: int = 50_000,
    overlap_threshold: float = 0.1,
    overlap_bins=(0, 1, 2, 4, 8, 16),
    seed: int = 0,
) -> dict:
    """Compute overlap-stratified signed error for a trained model on a volume.

    Sampling: uniform random voxels (not intensity-biased), so the report covers
    both empty regions and structure. Subsampling keeps cost bounded for large
    volumes.

    Returns
    -------
    {
        'n_samples':     int,
        'mae':           float,
        'rmse':          float,
        'signed_mean':   float,
        'per_overlap':   [
            {'bin': '0',  'count': int, 'mae': float, 'signed_mean': float, 'rmse': float},
            ...
        ],
    }
    """
    torch.manual_seed(seed)
    D, H, W = volume_t.shape
    device = volume_t.device
    n = min(n_samples, D * H * W)

    x = torch.randint(0, W, (n,), device=device).float()
    y = torch.randint(0, H, (n,), device=device).float()
    z = torch.randint(0, D, (n,), device=device).float()
    points = torch.stack([x, y, z], dim=-1)
    pred = gs.query_density(points)
    target = volume_t[z.long(), y.long(), x.long()]
    signed = (pred - target).cpu().numpy()
    overlaps = overlap_count(gs, points, overlap_threshold).cpu().numpy()

    def _stats(mask, label):
        s = signed[mask]
        return {
            'bin':         label,
            'count':       int(mask.sum()),
            'mae':         float(np.mean(np.abs(s))),
            'signed_mean': float(np.mean(s)),
            'rmse':        float(np.sqrt(np.mean(s ** 2))),
        }

    per_overlap = []
    edges = list(overlap_bins) + [10 ** 9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (overlaps >= lo) & (overlaps < hi)
        if not mask.any():
            continue
        if hi - lo > 1:
            label = f'{lo}-{hi - 1}' if hi < 10 ** 8 else f'{lo}+'
        else:
            label = str(lo)
        per_overlap.append(_stats(mask, label))

    return {
        'n_samples':   int(n),
        'mae':         float(np.mean(np.abs(signed))),
        'rmse':        float(np.sqrt(np.mean(signed ** 2))),
        'signed_mean': float(np.mean(signed)),
        'per_overlap': per_overlap,
    }

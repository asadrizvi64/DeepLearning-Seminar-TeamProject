"""Time-resolved (3D+t) representations and a synthetic 4D phantom (P5).

Implements the first of the three temporal variants from the plan:

  * shared-identity per-frame  -- one GaussianSet per frame, but frame t is
    warm-started from frame t-1 so a Gaussian keeps its identity across time.
    Gives temporal consistency by construction and supports tracking (E11/E7).

The 4D phantom has blobs that move smoothly (and optionally divide), so we have
ground-truth motion + a controlled division event for the mitosis stress test
(E8) before touching real data.

Native-4D and deformation-based variants will slot in next to this; they share
the metric helpers (temporal_smoothness, frame_to_frame_consistency).
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from .gaussians import GaussianSet
from .train import train_static, evaluate_full
from .densify import build_optimizer
from .losses import mse_loss


# ----------------------------------------------------------- 4D phantom

def generate_phantom_4d(
    shape=(24, 48, 48),
    num_blobs: int = 12,
    num_frames: int = 8,
    motion_scale: float = 1.5,
    blob_scale_range=(2.0, 3.0),
    division_frame: int | None = None,
    seed: int = 0,
):
    """Generate a (T, D, H, W) synthetic time series of moving Gaussian blobs.

    Each blob drifts on a smooth random sinusoid. If `division_frame` is set, the
    first blob splits into two from that frame on (a controlled mitosis event).

    Returns (volumes, tracks) where volumes is a list of (D, H, W) float32 arrays
    and tracks is (num_frames, n_blobs, 3) ground-truth centers in (x, y, z).
    """
    rng = np.random.default_rng(seed)
    D, H, W = shape
    margin = max(blob_scale_range) * 2
    base = np.stack([
        rng.uniform(margin, W - margin, num_blobs),
        rng.uniform(margin, H - margin, num_blobs),
        rng.uniform(margin, D - margin, num_blobs),
    ], axis=-1).astype(np.float32)                              # (N, 3)
    scales = rng.uniform(*blob_scale_range, size=(num_blobs, 3)).astype(np.float32)
    amps = rng.uniform(0.8, 1.2, num_blobs).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, (num_blobs, 3)).astype(np.float32)
    freq = rng.uniform(0.5, 1.5, (num_blobs, 3)).astype(np.float32)

    volumes, tracks = [], []
    for t in range(num_frames):
        u = t / max(num_frames - 1, 1)
        disp = motion_scale * np.sin(2 * np.pi * freq * u + phase)  # (N, 3)
        pos = base + disp.astype(np.float32)

        p, s, a = pos, scales, amps
        if division_frame is not None and t >= division_frame:
            # Blob 0 splits into two that separate along +x / -x.
            sep = (t - division_frame + 1) * 1.5
            child = pos[0].copy(); child[0] += sep
            parent = pos[0].copy(); parent[0] -= sep
            p = np.concatenate([pos, child[None]], 0); p[0] = parent
            s = np.concatenate([scales, scales[:1]], 0)
            a = np.concatenate([amps, amps[:1]], 0)

        quats = np.zeros((p.shape[0], 4), np.float32); quats[:, 0] = 1.0
        gs = GaussianSet(torch.from_numpy(p), torch.from_numpy(s),
                         torch.from_numpy(quats), torch.from_numpy(a))
        volumes.append(gs.query_volume(shape).cpu().numpy().astype(np.float32))
        tracks.append(p.copy())
    return volumes, tracks


# ----------------------------------------------- shared-identity fitting

def _clone_gs(gs: GaussianSet) -> GaussianSet:
    """Deep-copy a GaussianSet's parameters into a fresh module."""
    new = GaussianSet(gs.positions.detach().clone(),
                      torch.exp(gs.log_scales.detach()).clone())
    import torch.nn as nn
    new.log_scales = nn.Parameter(gs.log_scales.detach().clone())
    new.quaternions = nn.Parameter(gs.quaternions.detach().clone())
    new.amp_logits = nn.Parameter(gs.amp_logits.detach().clone())
    return new


def fit_timeseries_shared_identity(
    volumes,
    num_gaussians: int = 800,
    iters_first: int = 1500,
    iters_warm: int = 400,
    batch_size: int = 2048,
    device: str = None,
    seed: int = 0,
):
    """Fit a shared-identity per-frame representation.

    Frame 0 is fit from scratch; each later frame is warm-started from the
    previous frame's fitted Gaussians and refined for `iters_warm` steps. Because
    a Gaussian is carried forward, index i is the same primitive across all frames
    (the property tracking needs).

    Returns (gaussian_sets, history) - one GaussianSet per frame.
    """
    from .train import sample_training_points

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Frame 0 from scratch.
    gs0, _ = train_static(volumes[0], num_gaussians=num_gaussians,
                          iterations=iters_first, batch_size=batch_size,
                          device=device, seed=seed, eval_every=0)
    sets = [gs0]
    history = [{'frame': 0, 'psnr': float(evaluate_full(
        gs0, torch.from_numpy(volumes[0]).to(device)))}]

    for t in range(1, len(volumes)):
        gs = _clone_gs(sets[-1]).to(device)
        opt = build_optimizer(gs)
        vt = torch.from_numpy(volumes[t].astype(np.float32)).to(device)
        for _ in range(iters_warm):
            pos, tgt = sample_training_points(vt, batch_size)
            loss = mse_loss(gs.query_density(pos), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        sets.append(gs)
        history.append({'frame': t, 'psnr': float(evaluate_full(gs, vt))})
    return sets, history


# ----------------------------------------------------------- temporal metrics

@torch.no_grad()
def temporal_smoothness(sets) -> float:
    """Mean position acceleration (2nd time-difference) across shared-identity
    Gaussians. Lower = smoother motion. Requires equal counts across frames."""
    counts = {s.num_gaussians for s in sets}
    if len(counts) != 1 or len(sets) < 3:
        return float('nan')
    pos = torch.stack([s.positions.detach() for s in sets], 0)   # (T, N, 3)
    accel = pos[2:] - 2 * pos[1:-1] + pos[:-2]                    # (T-2, N, 3)
    return float(accel.norm(dim=-1).mean())


@torch.no_grad()
def frame_to_frame_consistency(sets, volumes, device=None) -> list:
    """PSNR of frame t's volume rendered with frame (t-1)'s model. Higher = more
    consistent (less drift). Returns a list of per-pair PSNRs."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = []
    for t in range(1, len(sets)):
        vt = torch.from_numpy(volumes[t].astype(np.float32)).to(device)
        out.append(float(evaluate_full(sets[t - 1].to(device), vt)))
    return out

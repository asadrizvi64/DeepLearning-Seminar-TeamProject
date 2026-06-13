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
import torch.nn as nn

from .gaussians import GaussianSet, quaternion_to_rotation
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

# =================================================================== native-4D

class GaussianSet4D(nn.Module):
    """One set of Gaussians with a time dimension: each primitive has a temporal
    center t_i and extent sigma_t_i, so the whole sequence is O(N) (not O(T*N)).

    Density at spatial point x and time t:
        rho(x,t) = sum_i amp_i * exp(-0.5 * (||S_i^-1 R_i^T (x-mu_i)||^2
                                             + (t - t_i)^2 / s_t_i^2))

    Renders at any continuous t (supports temporal interpolation, E10).
    """
    def __init__(self, positions, scales, t_centers, t_scales,
                 quaternions=None, amplitudes=None):
        super().__init__()
        N = positions.shape[0]
        self.positions = nn.Parameter(positions.float())
        self.log_scales = nn.Parameter(torch.log(scales.float().clamp_min(1e-4)))
        if quaternions is None:
            quaternions = torch.zeros(N, 4); quaternions[:, 0] = 1.0
        self.quaternions = nn.Parameter(quaternions.float())
        if amplitudes is None:
            amplitudes = torch.ones(N)
        self.amp_logits = nn.Parameter(torch.log(torch.expm1(amplitudes.float().clamp_min(1e-4))))
        self.t_centers = nn.Parameter(t_centers.float())
        self.log_t_scales = nn.Parameter(torch.log(t_scales.float().clamp_min(1e-3)))

    @property
    def num_gaussians(self): return self.positions.shape[0]

    def query_density(self, points, t: float, chunk_size: int = 4096):
        """Density at (P,3) spatial points for scalar time t. Returns (P,)."""
        mu = self.positions
        scales = torch.exp(self.log_scales)
        R = quaternion_to_rotation(self.quaternions)
        amp = torch.nn.functional.softplus(self.amp_logits)
        t_c = self.t_centers
        t_s = torch.exp(self.log_t_scales)
        P = points.shape[0]
        out = torch.zeros(P, device=points.device, dtype=points.dtype)
        for s in range(0, P, chunk_size):
            e = min(s + chunk_size, P)
            diff = points[s:e].unsqueeze(1) - mu.unsqueeze(0)
            local = torch.einsum('gji,pgj->pgi', R, diff)
            quad = ((local / scales.unsqueeze(0)) ** 2).sum(-1)
            tq = ((t - t_c) / t_s) ** 2
            g = torch.exp(-0.5 * (quad + tq.unsqueeze(0)))
            out[s:e] = (amp.unsqueeze(0) * g).sum(-1)
        return out


def fit_native_4d(volumes, num_gaussians: int = 1200, iterations: int = 2500,
                  batch_size: int = 2048, device: str = None, seed: int = 0):
    """Fit a single GaussianSet4D to all frames. Each iter picks a random frame,
    samples spatial points there, and supervises the 4D density at that time."""
    from .train import sample_training_points
    from .init import init_gaussians
    from .losses import mse_loss
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed); np.random.seed(seed)
    T = len(volumes)

    # Init spatial params from the middle frame; spread t-centers across the sequence.
    base = init_gaussians(volumes[T // 2], num_gaussians, strategy='intensity_weighted',
                          seed=seed)
    rng = np.random.default_rng(seed)
    t_centers = torch.from_numpy(rng.uniform(0, T - 1, num_gaussians).astype(np.float32))
    t_scales = torch.full((num_gaussians,), max(T / 4.0, 1.0))
    gs = GaussianSet4D(base.positions.detach(), torch.exp(base.log_scales.detach()),
                       t_centers, t_scales,
                       base.quaternions.detach(),
                       torch.nn.functional.softplus(base.amp_logits.detach())).to(device)
    opt = torch.optim.Adam([
        {'params': [gs.positions], 'lr': 0.0016},
        {'params': [gs.log_scales], 'lr': 0.005},
        {'params': [gs.quaternions], 'lr': 0.001},
        {'params': [gs.amp_logits], 'lr': 0.05},
        {'params': [gs.t_centers], 'lr': 0.01},
        {'params': [gs.log_t_scales], 'lr': 0.01},
    ])
    vts = [torch.from_numpy(v.astype(np.float32)).to(device) for v in volumes]
    for it in range(iterations):
        t = int(rng.integers(0, T))
        pos, tgt = sample_training_points(vts[t], batch_size)
        loss = mse_loss(gs.query_density(pos, float(t)), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return gs


@torch.no_grad()
def evaluate_4d_psnr(gs4d, volumes, t, subsample=50_000, device=None):
    """Density PSNR of a GaussianSet4D against frame `t` (t may be non-integer)."""
    from .losses import psnr
    if device is None:
        device = next(gs4d.parameters()).device
    vt = volumes[int(round(t))]
    D, H, W = vt.shape
    n = min(subsample, D * H * W)
    x = torch.randint(0, W, (n,), device=device).float()
    y = torch.randint(0, H, (n,), device=device).float()
    z = torch.randint(0, D, (n,), device=device).float()
    pts = torch.stack([x, y, z], -1)
    tgt = torch.from_numpy(vt.astype(np.float32)).to(device)[z.long(), y.long(), x.long()]
    return psnr(gs4d.query_density(pts, float(t)), tgt)


# =============================================================== deformation

class GaussianSetDeform(nn.Module):
    """Canonical Gaussians + a per-primitive polynomial motion field (deformation).

    position(t) = mu0 + v*(t - t_ref) + 0.5*a*(t - t_ref)^2

    Appearance (scale, rotation, amplitude) is shared across time; only positions
    deform. O(N) base + O(N) motion params, renders at continuous t. Smooth by
    construction (a low-order motion basis can't jitter), which is the point of the
    deformation family. t_ref defaults to the sequence midpoint.
    """
    def __init__(self, positions, scales, t_ref, quaternions=None, amplitudes=None):
        super().__init__()
        N = positions.shape[0]
        self.mu0 = nn.Parameter(positions.float())
        self.vel = nn.Parameter(torch.zeros(N, 3))
        self.acc = nn.Parameter(torch.zeros(N, 3))
        self.log_scales = nn.Parameter(torch.log(scales.float().clamp_min(1e-4)))
        if quaternions is None:
            quaternions = torch.zeros(N, 4); quaternions[:, 0] = 1.0
        self.quaternions = nn.Parameter(quaternions.float())
        if amplitudes is None:
            amplitudes = torch.ones(N)
        self.amp_logits = nn.Parameter(torch.log(torch.expm1(amplitudes.float().clamp_min(1e-4))))
        self.t_ref = float(t_ref)

    @property
    def num_gaussians(self): return self.mu0.shape[0]

    def positions_at(self, t: float):
        dt = t - self.t_ref
        return self.mu0 + self.vel * dt + 0.5 * self.acc * (dt * dt)

    def query_density(self, points, t: float, chunk_size: int = 4096):
        mu = self.positions_at(t)
        scales = torch.exp(self.log_scales)
        R = quaternion_to_rotation(self.quaternions)
        amp = torch.nn.functional.softplus(self.amp_logits)
        P = points.shape[0]
        out = torch.zeros(P, device=points.device, dtype=points.dtype)
        for s in range(0, P, chunk_size):
            e = min(s + chunk_size, P)
            diff = points[s:e].unsqueeze(1) - mu.unsqueeze(0)
            local = torch.einsum('gji,pgj->pgi', R, diff)
            quad = ((local / scales.unsqueeze(0)) ** 2).sum(-1)
            out[s:e] = (amp.unsqueeze(0) * torch.exp(-0.5 * quad)).sum(-1)
        return out


def fit_deformation(volumes, num_gaussians: int = 800, iterations: int = 2500,
                    batch_size: int = 2048, device: str = None, seed: int = 0):
    """Fit a canonical GaussianSet + polynomial motion to all frames."""
    from .train import sample_training_points
    from .init import init_gaussians
    from .losses import mse_loss
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed); np.random.seed(seed)
    T = len(volumes); t_ref = (T - 1) / 2.0
    base = init_gaussians(volumes[T // 2], num_gaussians, strategy='intensity_weighted', seed=seed)
    gs = GaussianSetDeform(base.positions.detach(), torch.exp(base.log_scales.detach()),
                           t_ref, base.quaternions.detach(),
                           torch.nn.functional.softplus(base.amp_logits.detach())).to(device)
    opt = torch.optim.Adam([
        {'params': [gs.mu0], 'lr': 0.0016}, {'params': [gs.vel], 'lr': 0.004},
        {'params': [gs.acc], 'lr': 0.002}, {'params': [gs.log_scales], 'lr': 0.005},
        {'params': [gs.quaternions], 'lr': 0.001}, {'params': [gs.amp_logits], 'lr': 0.05},
    ])
    vts = [torch.from_numpy(v.astype(np.float32)).to(device) for v in volumes]
    rng = np.random.default_rng(seed)
    for it in range(iterations):
        t = int(rng.integers(0, T))
        pos, tgt = sample_training_points(vts[t], batch_size)
        loss = mse_loss(gs.query_density(pos, float(t)), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return gs


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

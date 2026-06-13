"""Compression baselines for the RQ2 fidelity-vs-storage comparison (E5).

Each baseline returns (psnr_dB, size_bytes) so 3DGS can be placed on the same
Pareto front as the obvious alternatives:

  * compressed_voxels  - quantize to N bits + zlib. The "do nothing clever" floor.
  * sparse_pointcloud  - threshold, store (coord uint16x3 + value uint8) + zlib.
  * implicit_field     - a small SIREN MLP fit to the density field.

3DGS itself reuses train_static + the on-disk model size (see scripts/run_e5.py).
"""
from __future__ import annotations

import zlib

import numpy as np
import torch
import torch.nn as nn

from .losses import psnr


def compressed_voxels(vol: np.ndarray, bits: int = 8):
    """Quantize to `bits` and zlib-compress. Returns (psnr, bytes)."""
    levels = (1 << bits) - 1
    q = np.round(np.clip(vol, 0, 1) * levels).astype(np.uint16)
    comp = zlib.compress(q.tobytes(), 9)
    deq = q.astype(np.float32) / levels
    return psnr(deq, vol), len(comp)


def sparse_pointcloud(vol: np.ndarray, threshold: float):
    """Keep voxels above `threshold` as (coord uint16x3 + value uint8), zlib'd.

    Reconstruction is the thresholded volume, so PSNR reflects what's dropped
    below the threshold. Returns (psnr, bytes)."""
    mask = vol > threshold
    if not mask.any():
        return psnr(np.zeros_like(vol), vol), 0
    coords = np.argwhere(mask).astype(np.uint16)
    vals = np.round(np.clip(vol[mask], 0, 1) * 255).astype(np.uint8)
    payload = zlib.compress(coords.tobytes() + vals.tobytes(), 9)
    recon = np.zeros_like(vol, dtype=np.float32)
    recon[mask] = vals.astype(np.float32) / 255
    return psnr(recon, vol), len(payload)


class _SIREN(nn.Module):
    """Tiny sine-activated coordinate MLP: (x,y,z) -> density."""
    def __init__(self, hidden: int = 64, layers: int = 3, w0: float = 30.0):
        super().__init__()
        self.w0 = w0
        dims = [3] + [hidden] * layers + [1]
        self.lin = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        with torch.no_grad():
            self.lin[0].weight.uniform_(-1 / 3, 1 / 3)
            for l in self.lin[1:]:
                b = (6 / l.in_features) ** 0.5 / w0
                l.weight.uniform_(-b, b)

    def forward(self, x):
        x = torch.sin(self.w0 * self.lin[0](x))
        for l in self.lin[1:-1]:
            x = torch.sin(self.w0 * l(x))
        return self.lin[-1](x).squeeze(-1)


def implicit_field(vol: np.ndarray, hidden: int = 64, layers: int = 3,
                   iters: int = 1500, batch: int = 4096, device: str = None,
                   seed: int = 0):
    """Fit a SIREN to the volume. Returns (psnr, bytes) with bytes = params * 2 (fp16)."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    D, H, W = vol.shape
    vt = torch.from_numpy(vol.astype(np.float32)).to(device)
    net = _SIREN(hidden, layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    def coords(n):
        x = torch.randint(0, W, (n,), device=device)
        y = torch.randint(0, H, (n,), device=device)
        z = torch.randint(0, D, (n,), device=device)
        # normalize to [-1, 1]
        c = torch.stack([x / (W - 1), y / (H - 1), z / (D - 1)], -1) * 2 - 1
        return c.float(), vt[z, y, x]

    for _ in range(iters):
        c, tgt = coords(batch)
        loss = ((net(c) - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        c, tgt = coords(min(100_000, D * H * W))
        p = psnr(net(c), tgt)
    n_params = sum(prm.numel() for prm in net.parameters())
    return p, n_params * 2  # fp16 storage

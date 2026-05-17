"""Anisotropic 3D Gaussian set as a learnable density field.

The set represents a density

    rho(x) = sum_i  alpha_i * exp(-0.5 * (x - mu_i)^T Sigma_i^{-1} (x - mu_i))

where
    mu_i      : position             (learnable, R^3)
    Sigma_i   : covariance = R_i diag(s_i^2) R_i^T
    R_i       : rotation from unit quaternion  (learnable, R^4 -> SO(3))
    s_i       : anisotropic scales = exp(log_scales)  (learnable, R^3, positive)
    alpha_i   : amplitude = softplus(amp_logit)       (learnable, R, positive)

This implementation prioritizes clarity over speed. For production-scale fitting we'll
wire in a CUDA kernel (gsplat or custom) at Phase 2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
    """Convert (..., 4) quaternions (w, x, y, z) to (..., 3, 3) rotation matrices.

    Inputs are normalized internally.
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)], dim=-1),
        torch.stack([2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
        torch.stack([2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)
    return R


class GaussianSet(nn.Module):
    """A learnable set of anisotropic 3D Gaussians representing a density field."""

    def __init__(self, positions, scales, quaternions=None, amplitudes=None):
        super().__init__()
        N = positions.shape[0]

        self.positions = nn.Parameter(positions.float())

        if scales.ndim == 1:
            scales = scales[:, None].expand(N, 3).contiguous()
        self.log_scales = nn.Parameter(torch.log(scales.float().clamp_min(1e-4)))

        if quaternions is None:
            q = torch.zeros(N, 4)
            q[:, 0] = 1.0  # identity quaternion
            quaternions = q
        self.quaternions = nn.Parameter(quaternions.float())

        if amplitudes is None:
            amplitudes = torch.ones(N)
        # amp = softplus(amp_logit) => amp_logit = log(expm1(amp))
        amp_logits = torch.log(torch.expm1(amplitudes.float().clamp_min(1e-4)))
        self.amp_logits = nn.Parameter(amp_logits)

    # ------------------------------------------------------------------ properties

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    @property
    def amplitudes(self) -> torch.Tensor:
        return F.softplus(self.amp_logits)

    @property
    def rotations(self) -> torch.Tensor:
        return quaternion_to_rotation(self.quaternions)

    # ------------------------------------------------------------------ queries

    def query_density(self, points: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """Evaluate the density field at (P, 3) points in (x, y, z) order. Returns (P,).

        Chunks the (P, G, 3) intermediate to bound peak memory.
        """
        P = points.shape[0]
        out = torch.zeros(P, device=points.device, dtype=points.dtype)
        for start in range(0, P, chunk_size):
            end = min(start + chunk_size, P)
            out[start:end] = self._query_chunk(points[start:end])
        return out

    def _query_chunk(self, points: torch.Tensor) -> torch.Tensor:
        mu = self.positions                                   # (G, 3)
        scales = self.scales                                  # (G, 3)
        R = self.rotations                                    # (G, 3, 3)
        amp = self.amplitudes                                 # (G,)

        diff = points.unsqueeze(1) - mu.unsqueeze(0)          # (P, G, 3)
        # local = R^T @ diff  : transform diff into each Gaussian's local frame
        local = torch.einsum('gji,pgj->pgi', R, diff)         # (P, G, 3)
        normalized = local / scales.unsqueeze(0)              # (P, G, 3)
        quad = (normalized ** 2).sum(-1)                      # (P, G)
        gauss = torch.exp(-0.5 * quad)                        # (P, G)
        density = (amp.unsqueeze(0) * gauss).sum(-1)          # (P,)
        return density

    @torch.no_grad()
    def query_volume(self, shape, chunk_size: int = 4096, device=None) -> torch.Tensor:
        """Evaluate the density field on a dense voxel grid of shape (D, H, W).

        Returns a (D, H, W) tensor. Streams one z-slab at a time so memory cost is
        O(H * W * 3) for coordinates rather than O(D * H * W * 3).
        """
        if device is None:
            device = self.positions.device
        D, H, W = shape
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        # Build the HW grid once and reuse it for every z-slice.
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')          # (H, W) each
        xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2).float()  # (HW, 2)

        slabs = []
        for z in range(D):
            zz = torch.full((H * W, 1), float(z), device=device)
            pts = torch.cat([xy, zz], dim=-1)                   # (HW, 3) as (x, y, z)
            slabs.append(self.query_density(pts, chunk_size=chunk_size).reshape(H, W))
        return torch.stack(slabs, dim=0)                        # (D, H, W)

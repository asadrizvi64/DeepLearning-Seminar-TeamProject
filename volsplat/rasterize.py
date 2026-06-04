"""Orthographic alpha-blending rasterizer for axis-aligned sum-projections.

Phase 2 / E1: this is the "render-and-compare" supervision path. It implements
gsplat-style alpha blending along a chosen axis, which is the operator R^2-Gaussian
identified as introducing systematic intensity bias when used to fit density fields.

For a Gaussian set with parameters (mu, Sigma, alpha) and an orthographic view along
the chosen axis (z by default), the rendered image at pixel p is:

    I(p) = sum_i T_i(p) * a_i(p)
    a_i(p) = alpha_i * exp(-0.5 * (p - mu_2d_i)^T Sigma_2d_i^{-1} (p - mu_2d_i))
    T_i(p) = prod_{j<i} (1 - a_j(p))    # Gaussians sorted front-to-back by depth

where Sigma_2d_i is the 2x2 marginal covariance in the image plane. The integration
bias arises because T < 1 in regions of Gaussian overlap, so I is *not* a linear
function of the underlying densities (which a true sum-projection would be).

Conventions
-----------
Volumes are V[z, y, x] with shape (D, H, W); Gaussian positions are (x, y, z).
gt_projection sums the volume along the chosen axis (a true linear sum-projection,
the bias-free target). project_alpha_blend renders the same view via alpha blending.
"""
import torch

from .gaussians import GaussianSet


# axis -> (img_H, img_W, idx_col, idx_row, idx_depth)
#   idx_col: which component of (x, y, z) becomes the image column dim
#   idx_row: which component of (x, y, z) becomes the image row dim
#   idx_depth: which component is the viewing axis
def _axis_layout(axis: str, shape: tuple):
    D, H, W = shape
    if axis == 'z':
        return H, W, 0, 1, 2   # rows = y, cols = x
    if axis == 'y':
        return D, W, 0, 2, 1   # rows = z, cols = x
    if axis == 'x':
        return D, H, 1, 2, 0   # rows = z, cols = y
    raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")


def gt_projection(volume_t: torch.Tensor, axis: str) -> torch.Tensor:
    """True sum-projection of a (D, H, W) volume along the chosen axis.

    Returns:
        axis='z' -> (H, W) indexed [y, x]
        axis='y' -> (D, W) indexed [z, x]
        axis='x' -> (D, H) indexed [z, y]
    """
    if axis == 'z':
        return volume_t.sum(dim=0)
    if axis == 'y':
        return volume_t.sum(dim=1)
    if axis == 'x':
        return volume_t.sum(dim=2)
    raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")


def project_alpha_blend(
    gs: GaussianSet,
    axis: str,
    shape: tuple,
    pixel_chunk: int = 4096,
    alpha_clamp: float = 0.999,
    cov_reg: float = 1e-6,
) -> torch.Tensor:
    """Render an orthographic alpha-blended projection of the Gaussian set.

    Mirrors what gsplat does for image-plane rendering, restricted to an axis-aligned
    orthographic view and density-only "color" (no view-dependent SH).

    Parameters
    ----------
    gs : GaussianSet
    axis : 'x', 'y', or 'z' - the viewing axis
    shape : (D, H, W) of the source volume; only used to size the output image
    pixel_chunk : pixels processed per chunk (keeps the (Pc, G, *) tensors bounded)
    alpha_clamp : per-Gaussian opacity ceiling for compositing stability
    cov_reg : tiny diagonal added to the 2D marginal before inversion

    Returns
    -------
    image : (img_H, img_W) tensor, same device as the Gaussian set, differentiable
        w.r.t. all Gaussian parameters.
    """
    img_H, img_W, idx_col, idx_row, idx_depth = _axis_layout(axis, shape)

    mu = gs.positions          # (G, 3) in (x, y, z)
    R = gs.rotations           # (G, 3, 3)
    s = gs.scales              # (G, 3)
    amp = gs.amplitudes        # (G,)
    G = mu.shape[0]
    device = mu.device

    # Full 3D covariance, then take the 2x2 marginal in the image plane.
    Sigma = R @ torch.diag_embed(s ** 2) @ R.transpose(-1, -2)   # (G, 3, 3)
    idx_2d = [idx_col, idx_row]                                  # (col, row) order
    Sigma_2d = Sigma[:, idx_2d][:, :, idx_2d]                    # (G, 2, 2)
    Sigma_2d = Sigma_2d + cov_reg * torch.eye(2, device=device).expand(G, 2, 2)
    Sigma_2d_inv = torch.linalg.inv(Sigma_2d)                    # (G, 2, 2)

    mu_col = mu[:, idx_col]
    mu_row = mu[:, idx_row]
    mu_2d = torch.stack([mu_col, mu_row], dim=-1)                # (G, 2) as (col, row)
    mu_depth = mu[:, idx_depth]                                  # (G,)

    # Front-to-back depth sort for compositing.
    order = torch.argsort(mu_depth)
    mu_2d = mu_2d[order]
    Sigma_2d_inv = Sigma_2d_inv[order]
    amp = amp[order]

    # Pixel grid in image coords (col, row) = (x_img, y_img).
    rows = torch.arange(img_H, device=device).float()
    cols = torch.arange(img_W, device=device).float()
    rr, cc = torch.meshgrid(rows, cols, indexing='ij')
    pixels = torch.stack([cc, rr], dim=-1).reshape(-1, 2)        # (P, 2)
    P = pixels.shape[0]

    out = torch.zeros(P, device=device, dtype=mu.dtype)
    for ps in range(0, P, pixel_chunk):
        pe = min(ps + pixel_chunk, P)
        diff = pixels[ps:pe].unsqueeze(1) - mu_2d.unsqueeze(0)   # (Pc, G, 2)
        # quad = diff^T Sigma_inv diff
        Sd = torch.einsum('gij,pgj->pgi', Sigma_2d_inv, diff)    # (Pc, G, 2)
        quad = (Sd * diff).sum(-1)                               # (Pc, G)
        a = amp.unsqueeze(0) * torch.exp(-0.5 * quad)            # (Pc, G) raw splats
        # Clamp opacity for compositing stability (standard 3DGS practice).
        a = a.clamp(min=0.0, max=alpha_clamp)
        # Front-to-back composite: T_i = prod_{j<i} (1 - a_j); I = sum_i T_i a_i.
        one_minus_a = 1.0 - a                                    # (Pc, G), all in (eps, 1]
        T_running = torch.cumprod(one_minus_a, dim=1)            # T_after_i
        # Shift right with leading 1 to get T_before_i (transmittance hitting Gaussian i).
        T_before = torch.cat(
            [torch.ones_like(T_running[:, :1]), T_running[:, :-1]], dim=1
        )
        out[ps:pe] = (a * T_before).sum(-1)

    return out.reshape(img_H, img_W)

"""Loss functions and metrics."""
import math

import numpy as np
import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def psnr(pred, target, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB. Accepts torch tensors or numpy arrays."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    mse = ((pred - target) ** 2).mean()
    if mse <= 0:
        return float('inf')
    return float(20.0 * math.log10(max_val / math.sqrt(float(mse))))


def mae(pred, target) -> float:
    """Mean absolute error. Accepts torch tensors or numpy arrays.

    Complementary to PSNR:
      - In native intensity units [0, 1] (for normalized volumes) — directly
        interpretable as "average per-voxel error".
      - Not log-scaled; large errors do not dominate the summary the way they
        do in PSNR's squared-error base.
      - Always compute MAE and PSNR on the *same* voxel sample (see
        evaluate_metrics in train.py). Computing them on separate random samples
        introduces unnecessary variance in the comparison.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return float(np.abs(
        np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    ).mean())


def ssim3d(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
    win_size: int = 7,
) -> float:
    
    """
    Why SSIM is needed alongside PSNR
    ----------------------------------
    PSNR captures average squared error globally; it is insensitive to whether
    errors are spatially structured (e.g. an entire nucleus is blurred away) or
    scattered (small per-voxel noise). SSIM captures local contrast, luminance,
    and structural similarity — it can detect the case where reconstruction error
    decreases (PSNR rises) while biologically meaningful spatial structure is
    being smeared (SSIM falls). This distinction matters for microscopy data
    where downstream tasks (segmentation, tracking) depend on spatial coherence.

    Implementation
    --------------
    Uses scikit-image's N-dimensional `structural_similarity`, which extends the
    standard 2D sliding-window SSIM to arbitrary dimensions using a 3D window.
    This is the scientifically strongest SSIM variant for volumetric data because
    it captures inter-slice structure that slice-wise 2D SSIM misses.

    Requires full voxelization of the Gaussian field (gs.query_volume(shape)).
    For volumes ≤5M voxels this is fast; see evaluate_ssim in train.py for the
    large-volume fallback.

    Parameters
    ----------
    pred, target : (D, H, W) float32/float64 arrays, values in [0, data_range]
    data_range   : intensity dynamic range (1.0 for [0,1]-normalized volumes)
    win_size     : side length of the 3D sliding window; must be odd and
                   ≤ min(D, H, W). Clamped automatically.

    Notes on win_size
    -----------------
    The default win_size=7 is standard for 2D images (7×7 = 49 pixels) and
    gives a 7×7×7 = 343-voxel 3D window. For very small volumes (D < 7),
    it is automatically reduced to the largest odd number ≤ D. A minimum of
    3 is enforced. Results from volumes where win_size was clamped are still
    valid but compare different window sizes — note the clamped value in
    the report.
    """
    from skimage.metrics import structural_similarity

    min_dim = min(pred.shape)
    ws = min(win_size, min_dim)
    if ws % 2 == 0:
        ws -= 1
    ws = max(ws, 3)

    return float(structural_similarity(
        pred.astype(np.float64),
        target.astype(np.float64),
        data_range=float(data_range),
        win_size=ws,
    ))


def ssim3d_slicewise(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
    win_size: int = 7,
) -> float:
    """Fallback: average 2D SSIM over all z-slices.

    Used automatically by evaluate_ssim when the volume exceeds max_voxels.
    Less accurate than full 3D SSIM (ignores inter-slice structure) but
    tractable for large volumes where 3D sliding window is too slow.
    """
    from skimage.metrics import structural_similarity

    ssims = []
    for z in range(pred.shape[0]):
        s = pred[z].astype(np.float64)
        t = target[z].astype(np.float64)
        min_dim = min(s.shape)
        ws = min(win_size, min_dim)
        if ws % 2 == 0:
            ws -= 1
        ws = max(ws, 3)
        ssims.append(float(structural_similarity(s, t, data_range=float(data_range), win_size=ws)))
    return float(np.mean(ssims))

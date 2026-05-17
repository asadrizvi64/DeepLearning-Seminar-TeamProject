"""Volume loaders and synthetic phantom generation."""
from pathlib import Path

import numpy as np
import torch
import tifffile


# ---------------------------------------------------------------------------- IO

def load_volume(path) -> np.ndarray:
    """Load a TIFF stack or .npy as a (D, H, W) float32 numpy array, normalized to [0, 1].

    Squeezes leading/trailing singleton axes so common microscopy formats like
    (1, Z, Y, X) or (Z, Y, X, 1) work without pre-processing.
    """
    path = Path(path)
    if path.suffix in {'.tif', '.tiff'}:
        vol = tifffile.imread(str(path))
    elif path.suffix == '.npy':
        vol = np.load(str(path))
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")
    vol = vol.astype(np.float32)
    # Squeeze out any size-1 axes (channel, time singletons common in OME-TIFF).
    vol = np.squeeze(vol)
    if vol.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume (D, H, W) but got shape {vol.shape} after squeezing "
            f"singleton axes. For multi-channel data select a single channel before "
            f"loading, e.g.: tifffile.imread(path)[0] for the first channel."
        )
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    return vol


def save_volume(volume: np.ndarray, path) -> None:
    """Save a (D, H, W) volume to TIFF or .npy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {'.tif', '.tiff'}:
        tifffile.imwrite(str(path), volume.astype(np.float32))
    elif path.suffix == '.npy':
        np.save(str(path), volume)
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")


# ------------------------------------------------------------------ phantom gen

def generate_phantom(
    shape=(64, 128, 128),
    num_blobs: int = 50,
    blob_scale_range=(2.0, 5.0),
    blob_amp_range=(0.5, 1.5),
    noise_std: float = 0.0,
    seed: int = 0,
    return_params: bool = False,
):
    """Generate a synthetic 3D microscopy-like volume.

    Sparse anisotropic Gaussian blobs placed at random positions inside the volume,
    rendered to a voxel grid. By construction the ground truth is exactly a Gaussian sum,
    so a correctly-implemented fitter must be able to recover near-zero MSE.

    Parameters
    ----------
    shape : (D, H, W)
    num_blobs : number of blobs
    blob_scale_range : per-axis scale range
    blob_amp_range : amplitude range
    noise_std : Gaussian noise std (0 => exact recoverability sanity test)
    seed : RNG seed
    return_params : also return the ground-truth blob parameters

    Returns
    -------
    volume : (D, H, W) float32 numpy array.
    (optional) params : dict with arrays positions, scales, amplitudes, quaternions.
    """
    rng = np.random.default_rng(seed)
    D, H, W = shape
    margin = max(blob_scale_range) * 2
    if margin >= min(D, H, W) / 2:
        raise ValueError(
            f"blob_scale_range {blob_scale_range} produces a margin of {margin:.1f} voxels "
            f"but the smallest volume dimension is {min(D, H, W)}. Either reduce "
            f"blob_scale_range or increase the volume shape."
        )
    positions = np.stack([
        rng.uniform(margin, W - margin, size=num_blobs),
        rng.uniform(margin, H - margin, size=num_blobs),
        rng.uniform(margin, D - margin, size=num_blobs),
    ], axis=-1).astype(np.float32)  # (N, 3) as (x, y, z)
    scales = rng.uniform(*blob_scale_range, size=(num_blobs, 3)).astype(np.float32)
    amplitudes = rng.uniform(*blob_amp_range, size=num_blobs).astype(np.float32)
    quats = rng.normal(size=(num_blobs, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)

    # Render via the same evaluator we use for fitting, so the phantom is *exactly* a
    # GaussianSet evaluation. This makes P1 a clean sanity check.
    from .gaussians import GaussianSet
    gs = GaussianSet(
        positions=torch.from_numpy(positions),
        scales=torch.from_numpy(scales),
        quaternions=torch.from_numpy(quats),
        amplitudes=torch.from_numpy(amplitudes),
    )
    volume = gs.query_volume(shape).cpu().numpy().astype(np.float32)

    if noise_std > 0:
        volume = volume + rng.normal(scale=noise_std, size=volume.shape).astype(np.float32)
        volume = np.clip(volume, 0, None)

    if return_params:
        params = dict(positions=positions, scales=scales, amplitudes=amplitudes, quaternions=quats)
        return volume, params
    return volume


# ------------------------------------------------------------------ sampling helpers

def intensity_weighted_sample(
    volume: np.ndarray,
    num_points: int,
    eps: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """Sample voxel coordinates with probability proportional to intensity.

    Returns positions in (x, y, z) order with sub-voxel jitter.
    """
    rng = np.random.default_rng(seed)
    D, H, W = volume.shape
    flat = volume.flatten() + eps
    flat = flat / flat.sum()
    idx = rng.choice(flat.size, size=num_points, replace=True, p=flat)
    z = idx // (H * W)
    rem = idx % (H * W)
    y = rem // W
    x = rem % W
    jitter = rng.uniform(-0.5, 0.5, size=(num_points, 3)).astype(np.float32)
    return np.stack([x, y, z], axis=-1).astype(np.float32) + jitter

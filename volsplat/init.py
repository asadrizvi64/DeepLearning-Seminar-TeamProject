"""Initialization strategies for GaussianSet (Phase 3 / E2).

Three strategies, all returning a `GaussianSet` of the requested size:

  * 'random'              - positions uniformly random inside the volume bbox;
                            amplitudes = mean intensity; isotropic init scale.
  * 'intensity_weighted'  - sample positions with probability proportional to intensity;
                            amplitudes = local intensity. (The P1 default.)
  * 'local_maxima'        - smooth the volume, find local maxima, place Gaussians at
                            the top-`num_gaussians` peaks. The "blob/nuclei detection"
                            counterpart 3DGS would use SfM for, adapted to volumes.

E2 will compare these on (final PSNR, fit time, #Gaussians to reach a target PSNR).
"""
from __future__ import annotations

import numpy as np
import torch

from .gaussians import GaussianSet
from .data import intensity_weighted_sample


# ----------------------------------------------------------- shared helper

def _build_gaussian_set(
    positions: np.ndarray,
    amplitudes: np.ndarray,
    num_gaussians: int,
    init_scale,
) -> GaussianSet:
    if np.isscalar(init_scale):
        scales = np.full((num_gaussians, 3), float(init_scale), dtype=np.float32)
    else:
        s = np.asarray(init_scale, dtype=np.float32).reshape(3)
        scales = np.tile(s[None, :], (num_gaussians, 1))
    quats = np.zeros((num_gaussians, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    return GaussianSet(
        positions=torch.from_numpy(positions.astype(np.float32)),
        scales=torch.from_numpy(scales),
        quaternions=torch.from_numpy(quats),
        amplitudes=torch.from_numpy(amplitudes.astype(np.float32)),
    )


# ----------------------------------------------------------- strategies

def random_init(
    volume: np.ndarray,
    num_gaussians: int,
    init_scale=2.0,
    seed: int = 0,
    margin: int = 2,
) -> GaussianSet:
    """Uniformly-random positions inside [margin, dim - margin]. Constant amplitude
    set to the volume's mean intensity (clipped at 0.05)."""
    rng = np.random.default_rng(seed)
    D, H, W = volume.shape
    positions = np.stack([
        rng.uniform(margin, W - margin, size=num_gaussians),
        rng.uniform(margin, H - margin, size=num_gaussians),
        rng.uniform(margin, D - margin, size=num_gaussians),
    ], axis=-1)
    mean_int = float(np.maximum(volume.mean(), 0.05))
    amps = np.full(num_gaussians, mean_int, dtype=np.float32)
    return _build_gaussian_set(positions, amps, num_gaussians, init_scale)


def intensity_weighted_init(
    volume: np.ndarray,
    num_gaussians: int,
    init_scale=2.0,
    seed: int = 0,
) -> GaussianSet:
    """Intensity-weighted positions; amplitudes = local intensity. P1 default."""
    positions = intensity_weighted_sample(volume, num_gaussians, seed=seed)
    D, H, W = volume.shape
    ix = np.clip(positions[:, 0].round().astype(np.int64), 0, W - 1)
    iy = np.clip(positions[:, 1].round().astype(np.int64), 0, H - 1)
    iz = np.clip(positions[:, 2].round().astype(np.int64), 0, D - 1)
    amps = np.clip(volume[iz, iy, ix], 0.05, None).astype(np.float32)
    return _build_gaussian_set(positions, amps, num_gaussians, init_scale)


def local_maxima_init(
    volume: np.ndarray,
    num_gaussians: int,
    init_scale=2.0,
    seed: int = 0,
    smooth_sigma: float = 1.5,
    nms_size: int = 3,
) -> GaussianSet:
    """Smooth volume, find local maxima via max-filter NMS, take the top-`num_gaussians`
    peaks by intensity. Falls back to intensity-weighted sampling if not enough peaks
    are found (typical for very dense / unstructured volumes).
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    smoothed = gaussian_filter(volume.astype(np.float32), sigma=smooth_sigma)
    local_max = (maximum_filter(smoothed, size=nms_size) == smoothed) & (smoothed > 0.0)
    coords = np.argwhere(local_max)                  # (N_peaks, 3) as (z, y, x)
    if coords.shape[0] == 0:
        return intensity_weighted_init(volume, num_gaussians, init_scale, seed)

    intensities = smoothed[coords[:, 0], coords[:, 1], coords[:, 2]]
    order = np.argsort(-intensities)                 # highest first
    coords = coords[order]
    intensities = intensities[order]

    if coords.shape[0] >= num_gaussians:
        coords = coords[:num_gaussians]
        intensities = intensities[:num_gaussians]
        positions = np.stack(
            [coords[:, 2], coords[:, 1], coords[:, 0]], axis=-1   # -> (x, y, z)
        ).astype(np.float32)
        amps = np.clip(intensities, 0.05, None).astype(np.float32)
        return _build_gaussian_set(positions, amps, num_gaussians, init_scale)

    # Fewer peaks than requested: keep all peaks, top up with intensity-weighted sampling.
    n_peaks = coords.shape[0]
    n_fill = num_gaussians - n_peaks
    peak_pos = np.stack(
        [coords[:, 2], coords[:, 1], coords[:, 0]], axis=-1
    ).astype(np.float32)
    fill_pos = intensity_weighted_sample(volume, n_fill, seed=seed)
    positions = np.concatenate([peak_pos, fill_pos], axis=0)
    D, H, W = volume.shape
    ix = np.clip(fill_pos[:, 0].round().astype(np.int64), 0, W - 1)
    iy = np.clip(fill_pos[:, 1].round().astype(np.int64), 0, H - 1)
    iz = np.clip(fill_pos[:, 2].round().astype(np.int64), 0, D - 1)
    fill_amps = np.clip(volume[iz, iy, ix], 0.05, None).astype(np.float32)
    amps = np.concatenate(
        [np.clip(intensities, 0.05, None).astype(np.float32), fill_amps], axis=0
    )
    return _build_gaussian_set(positions, amps, num_gaussians, init_scale)


# ----------------------------------------------------------- dispatcher

INIT_STRATEGIES = {
    'random':              random_init,
    'intensity_weighted':  intensity_weighted_init,
    'local_maxima':        local_maxima_init,
}


def init_gaussians(
    volume: np.ndarray,
    num_gaussians: int,
    strategy: str = 'intensity_weighted',
    init_scale=2.0,
    seed: int = 0,
    **kwargs,
) -> GaussianSet:
    """Dispatch initializer. `kwargs` are forwarded to the chosen strategy."""
    if strategy not in INIT_STRATEGIES:
        raise ValueError(
            f"Unknown init strategy {strategy!r}. "
            f"Available: {sorted(INIT_STRATEGIES)}"
        )
    return INIT_STRATEGIES[strategy](
        volume, num_gaussians, init_scale=init_scale, seed=seed, **kwargs
    )

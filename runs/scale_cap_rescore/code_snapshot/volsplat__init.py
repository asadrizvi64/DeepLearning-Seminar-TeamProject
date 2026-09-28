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


def coverage_init(
    volume: np.ndarray,
    num_gaussians: int,
    init_scale=2.0,
    seed: int = 0,
    smooth_sigma: float = 2.0,
    min_distance: int = 13,
    threshold_rel: float = 0.12,
    max_candidates: int = 20000,
) -> GaussianSet:
    """Cover every detected nucleus first, then spend what's left on support.

    `local_maxima_init` ranks candidate peaks by intensity and keeps the top
    `num_gaussians`. On real embryo data that systematically starves DIM nuclei: they
    are never seeded no matter how large the budget, because brighter regions consume
    the whole ranking (measured: an 8x surplus budget still covered only half the
    nuclei, and 58% of missed nuclei had no Gaussian within 3 voxels).

    This strategy instead:
      1. detects candidate nuclei with non-maximum suppression at the NUCLEUS SPACING
         (`min_distance`), using a LOW relative threshold so faint nuclei survive;
      2. places exactly one Gaussian on each candidate -- coverage before budget;
      3. allocates any remaining budget by intensity-weighted sampling, which supplies
         broad background/support structure.

    If the budget is smaller than the candidate count, candidates are taken in order of
    intensity (coverage is then impossible and the caller should raise `num_gaussians`).
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    smoothed = gaussian_filter(volume.astype(np.float32), sigma=smooth_sigma)
    thr = float(smoothed.max()) * threshold_rel
    size = int(2 * min_distance + 1)
    is_peak = (maximum_filter(smoothed, size=size) == smoothed) & (smoothed > thr)
    coords = np.argwhere(is_peak)                          # (N, 3) as (z, y, x)

    if coords.shape[0] == 0:
        return intensity_weighted_init(volume, num_gaussians, init_scale, seed)

    inten = smoothed[coords[:, 0], coords[:, 1], coords[:, 2]]
    order = np.argsort(-inten)
    coords, inten = coords[order][:max_candidates], inten[order][:max_candidates]

    n_nuc = min(coords.shape[0], num_gaussians)
    nuc_pos = np.stack(
        [coords[:n_nuc, 2], coords[:n_nuc, 1], coords[:n_nuc, 0]], axis=-1
    ).astype(np.float32)                                   # -> (x, y, z)
    nuc_amp = np.clip(inten[:n_nuc], 0.05, None).astype(np.float32)

    n_fill = num_gaussians - n_nuc
    if n_fill <= 0:
        return _build_gaussian_set(nuc_pos, nuc_amp, num_gaussians, init_scale)

    fill_pos = intensity_weighted_sample(volume, n_fill, seed=seed)
    D, H, W = volume.shape
    ix = np.clip(fill_pos[:, 0].round().astype(np.int64), 0, W - 1)
    iy = np.clip(fill_pos[:, 1].round().astype(np.int64), 0, H - 1)
    iz = np.clip(fill_pos[:, 2].round().astype(np.int64), 0, D - 1)
    fill_amp = np.clip(volume[iz, iy, ix], 0.05, None).astype(np.float32)

    positions = np.concatenate([nuc_pos, fill_pos], axis=0)
    amps = np.concatenate([nuc_amp, fill_amp], axis=0)
    return _build_gaussian_set(positions, amps, num_gaussians, init_scale)


def suppressed_topk_init(
    volume: np.ndarray,
    num_gaussians: int,
    init_scale=2.0,
    seed: int = 0,
    smooth_sigma: float = 2.0,
    min_distance: int = 13,
) -> GaussianSet:
    """Control for `coverage_init`: spatial suppression at the nucleus spacing, but
    still ranked GLOBALLY BY INTENSITY with no coverage guarantee.

    `local_maxima_init` suppresses at only 3 voxels, so a single bright nucleus can
    absorb many redundant seeds; `coverage_init` changes BOTH the suppression radius
    and the allocation policy. This strategy changes only the radius, isolating how
    much of the improvement comes from de-duplication alone versus from covering
    faint candidates.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    smoothed = gaussian_filter(volume.astype(np.float32), sigma=smooth_sigma)
    size = int(2 * min_distance + 1)
    is_peak = (maximum_filter(smoothed, size=size) == smoothed) & (smoothed > 0.0)
    coords = np.argwhere(is_peak)
    if coords.shape[0] == 0:
        return intensity_weighted_init(volume, num_gaussians, init_scale, seed)

    inten = smoothed[coords[:, 0], coords[:, 1], coords[:, 2]]
    order = np.argsort(-inten)
    coords, inten = coords[order], inten[order]

    n = min(coords.shape[0], num_gaussians)
    pos = np.stack([coords[:n, 2], coords[:n, 1], coords[:n, 0]], axis=-1).astype(np.float32)
    amp = np.clip(inten[:n], 0.05, None).astype(np.float32)
    if n == num_gaussians:
        return _build_gaussian_set(pos, amp, num_gaussians, init_scale)

    n_fill = num_gaussians - n
    fill_pos = intensity_weighted_sample(volume, n_fill, seed=seed)
    D, H, W = volume.shape
    ix = np.clip(fill_pos[:, 0].round().astype(np.int64), 0, W - 1)
    iy = np.clip(fill_pos[:, 1].round().astype(np.int64), 0, H - 1)
    iz = np.clip(fill_pos[:, 2].round().astype(np.int64), 0, D - 1)
    fill_amp = np.clip(volume[iz, iy, ix], 0.05, None).astype(np.float32)
    return _build_gaussian_set(np.concatenate([pos, fill_pos]),
                               np.concatenate([amp, fill_amp]),
                               num_gaussians, init_scale)


def oracle_coverage_init(
    volume: np.ndarray,
    num_gaussians: int,
    init_scale=2.0,
    seed: int = 0,
    nuclei=None,
) -> GaussianSet:
    """EXPERIMENTAL UPPER BOUND -- NOT A DEPLOYABLE METHOD.

    Places one Gaussian on each supplied target nucleus, then fills the remaining
    budget with intensity-weighted support. Because it consumes the evaluation target,
    it must never be reported as a method; its only purpose is to isolate stage 3 by
    answering: if EVERY nucleus starts covered, how many survive fitting?

    `nuclei` is an (N, 3) array of (z, y, x) target positions.
    """
    if nuclei is None or len(nuclei) == 0:
        raise ValueError("oracle_coverage_init requires `nuclei` (N,3) in (z,y,x). "
                         "It is an experimental upper bound, not a real initializer.")
    nuclei = np.asarray(nuclei)
    n = min(len(nuclei), num_gaussians)
    pos = np.stack([nuclei[:n, 2], nuclei[:n, 1], nuclei[:n, 0]], axis=-1).astype(np.float32)

    D, H, W = volume.shape
    iz = np.clip(nuclei[:n, 0].round().astype(np.int64), 0, D - 1)
    iy = np.clip(nuclei[:n, 1].round().astype(np.int64), 0, H - 1)
    ix = np.clip(nuclei[:n, 2].round().astype(np.int64), 0, W - 1)
    amp = np.clip(volume[iz, iy, ix], 0.05, None).astype(np.float32)

    n_fill = num_gaussians - n
    if n_fill <= 0:
        return _build_gaussian_set(pos, amp, num_gaussians, init_scale)

    fill_pos = intensity_weighted_sample(volume, n_fill, seed=seed)
    fx = np.clip(fill_pos[:, 0].round().astype(np.int64), 0, W - 1)
    fy = np.clip(fill_pos[:, 1].round().astype(np.int64), 0, H - 1)
    fz = np.clip(fill_pos[:, 2].round().astype(np.int64), 0, D - 1)
    fill_amp = np.clip(volume[fz, fy, fx], 0.05, None).astype(np.float32)
    return _build_gaussian_set(np.concatenate([pos, fill_pos]),
                               np.concatenate([amp, fill_amp]),
                               num_gaussians, init_scale)


# ----------------------------------------------------------- dispatcher

INIT_STRATEGIES = {
    'random':              random_init,
    'intensity_weighted':  intensity_weighted_init,
    'local_maxima':        local_maxima_init,
    'suppressed_topk':     suppressed_topk_init,
    'coverage':            coverage_init,
    # experimental upper bound; consumes the evaluation target, never a method
    'oracle_coverage':     oracle_coverage_init,
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

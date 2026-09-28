"""Cell-detection metrics: does the reconstruction recover the NUCLEI, not just the intensities?

PSNR measures photometric agreement. On this data it is nearly uncorrelated with whether
the fit actually recovers individual nuclei (r = 0.199 across the colleague's 19
cell-matching runs). These metrics score the biologically meaningful question directly.

Protocol matches the colleague's `cell_count_matching.json` so numbers are comparable:
detect peaks on a smoothed volume (or its MIP), then match predicted peaks to target
peaks within a radius and report precision / recall / F1 / localization RMSE.

Detection defaults (his `cell_count_parameters.json`):
    mode='mip', threshold_abs=0.35, smoothing_sigma=2.0, min_distance=13

One honest caveat carried over from his notes: the target peak set is an IMAGE-DERIVED
estimate, not manual ground-truth annotation. F1 here measures agreement with a detector
run on the target volume, not with a human-annotated cell census.

Difference from his implementation: matching here is OPTIMAL (Hungarian assignment)
rather than greedy nearest-neighbour, so scores are a slight upper bound on his.
"""
from __future__ import annotations

import numpy as np


def detect_cells(
    volume: np.ndarray,
    mode: str = 'mip',
    threshold_abs: float = 0.35,
    smoothing_sigma: float = 2.0,
    min_distance: int = 13,
    mip_axis: int = 0,
    max_peaks: int = 20000,
    prominence: float = 1e-4,
    voxel_size_zyx=None,
) -> np.ndarray:
    """Detect cell centres. Returns (N, 2) coords for mode='mip', (N, 3) for mode='3d'.

    'mip' projects along `mip_axis` first (fast, robust on large volumes -- his default
    for the full 37M-voxel volume); '3d' detects directly in the volume.

    `voxel_size_zyx` -- pass (dz, dy, dx) in microns on ANISOTROPIC data. Then
    `min_distance` and `smoothing_sigma` are read as MICRONS and the suppression window
    is sized per axis so it spans the same physical distance in every direction.

    Leaving it None keeps the legacy voxel-space cubic behaviour, which is WRONG on
    anisotropic data and silently so: on Fluo-N3DL-DRO (2.03 x 0.406 x 0.406 um) a
    13-voxel cube spans 26 um in z against 5.3 um in xy, while nuclei sit 7.6 um apart,
    so it merges nuclei along z. Measured against 29 human-annotated nuclei, the cubic
    detector recovered 5/29 (recall 0.172) from the RAW TARGET volume, versus 22/29
    (0.759) with a physically isotropic window. Any score computed with the cubic
    default on anisotropic data is capped by that, not by the model under test.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

    img = volume.max(axis=mip_axis) if mode == 'mip' else volume

    if voxel_size_zyx is None:
        sig = smoothing_sigma
        size = int(2 * min_distance + 1)
    else:
        vs = np.asarray(voxel_size_zyx, dtype=np.float64)
        if mode == 'mip':
            vs = np.delete(vs, mip_axis)
        sig = tuple(smoothing_sigma / vs)
        size = [max(1, int(2 * round(min_distance / s)) + 1) for s in vs]

    if smoothing_sigma and smoothing_sigma > 0:
        img = gaussian_filter(img.astype(np.float32), sigma=sig)
    # A FLAT region satisfies `maximum_filter(img) == img` at every voxel, so a
    # reconstruction containing genuinely constant areas (e.g. one carrying an explicit
    # background term, where far-from-any-Gaussian regions equal b exactly) would report
    # a "peak" at every voxel of the plateau. Requiring the window to actually vary
    # removes that artefact without affecting real peaks.
    is_peak = (
        (maximum_filter(img, size=size) == img)
        & (img >= threshold_abs)
        & (maximum_filter(img, size=size) > minimum_filter(img, size=size) + prominence)
    )
    coords = np.argwhere(is_peak)
    if coords.shape[0] == 0:
        return coords.astype(np.float32)

    # strongest peaks first, capped
    vals = img[tuple(coords.T)]
    order = np.argsort(-vals)
    coords = coords[order][:max_peaks]
    return coords.astype(np.float32)


def match_indices(
    pred_peaks: np.ndarray,
    target_peaks: np.ndarray,
    match_radius: float,
):
    """One-to-one matching within `match_radius`: MAXIMIZE the number of matched pairs
    first, and only among maximum-cardinality solutions minimize total distance.

    BUG this replaces: running `linear_sum_assignment` directly on the raw distance
    matrix and THEN discarding pairs beyond `match_radius` minimizes total assignment
    cost, not match count -- it can report zero matches when a valid one exists.
    Counterexample (reproduced): pred=[10,30], target=[22,50], radius=9.945.
    Distances [[12,40],[8,20]]; the min-cost assignment is (0,0)+(1,1) at cost 32,
    both legs exceed the radius -> 0 matches. But 30->22 alone is a valid match at
    distance 8. The min-cost assignment doesn't know about the radius filter.

    Fix: penalize every out-of-radius edge with a cost that provably dominates the sum
    of all in-radius edges (`big = (match_radius + 1) * (n_pred + n_tgt) + 1`, larger
    than n*match_radius for any n <= n_pred+n_tgt). Hungarian on this reweighted matrix
    then strictly prefers using more in-radius edges over any number of out-of-radius
    ones -- i.e. it maximizes cardinality within the radius first, and minimizes
    distance second, among in-radius edges only. Verified against the counterexample
    above: it returns the correct single match (30->22, distance 8).

    Returns (pred_idx, target_idx, distances) for the ACCEPTED (within-radius) pairs.
    """
    n_pred, n_tgt = len(pred_peaks), len(target_peaks)
    if n_pred == 0 or n_tgt == 0:
        empty_i = np.empty(0, dtype=int)
        return empty_i, empty_i, np.empty(0, dtype=float)

    from scipy.optimize import linear_sum_assignment
    d = np.linalg.norm(pred_peaks[:, None, :] - target_peaks[None, :, :], axis=-1)
    big = (float(match_radius) + 1.0) * (n_pred + n_tgt) + 1.0
    cost = np.where(d <= match_radius, d, big)
    ri, ci = linear_sum_assignment(cost)
    keep = d[ri, ci] <= match_radius
    return ri[keep], ci[keep], d[ri, ci][keep]


def match_cells(
    pred_peaks: np.ndarray,
    target_peaks: np.ndarray,
    match_radius: float,
) -> dict:
    """Match predicted to target peaks within `match_radius` (see `match_indices` for
    the matching rule: maximum cardinality within the radius, minimum distance among
    max-cardinality solutions). Returns precision / recall / f1 / localization RMSE / counts.
    """
    n_pred, n_tgt = len(pred_peaks), len(target_peaks)
    base = {
        'pred_cell_count': int(n_pred),
        'target_cell_count': int(n_tgt),
        'cell_count_error': int(abs(n_pred - n_tgt)),
        'cell_count_relative_error': (abs(n_pred - n_tgt) / n_tgt) if n_tgt else float('nan'),
        'match_radius_voxels': float(match_radius),
    }
    if n_pred == 0 or n_tgt == 0:
        base.update(matched_peaks=0, precision=0.0, recall=0.0, cell_f1=0.0,
                    localization_rmse_voxels=float('nan'),
                    mean_match_distance_voxels=float('nan'))
        return base

    ri, ci, dist = match_indices(pred_peaks, target_peaks, match_radius)
    matched = len(ri)

    precision = matched / n_pred
    recall = matched / n_tgt
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    base.update(
        matched_peaks=matched,
        precision=float(precision),
        recall=float(recall),
        cell_f1=float(f1),
        localization_rmse_voxels=float(np.sqrt((dist ** 2).mean())) if matched else float('nan'),
        mean_match_distance_voxels=float(dist.mean()) if matched else float('nan'),
    )
    return base


def score_cells(
    recon: np.ndarray,
    target: np.ndarray,
    mode: str = 'mip',
    threshold_abs: float = 0.35,
    smoothing_sigma: float = 2.0,
    min_distance: int = 13,
    match_radius: float = None,
    mip_axis: int = 0,
    prominence: float = 1e-4,
    voxel_size_zyx=None,
) -> dict:
    """Detect cells in both volumes with identical settings and score the match.

    `match_radius` defaults to 0.765 * min_distance (matches his auto radius of
    9.94 voxels at min_distance=13). `voxel_size_zyx` is forwarded to `detect_cells`
    for physically-correct detection on anisotropic data (see its docstring); when set,
    `min_distance` is interpreted in the same units passed there (microns), and both
    peak sets and `match_radius` must then be in the SAME units -- this function does
    NOT itself rescale peak coordinates for matching, since that depends on whether the
    caller wants voxel-space or physical-space match distance (see
    scripts/test_metrics.py and scripts/capstone_transfer.py for the established
    physical-matching pattern: scale coordinates by voxel size before matching).
    """
    if match_radius is None:
        match_radius = 0.765 * min_distance
    kw = dict(mode=mode, threshold_abs=threshold_abs, smoothing_sigma=smoothing_sigma,
              min_distance=min_distance, mip_axis=mip_axis, prominence=prominence,
              voxel_size_zyx=voxel_size_zyx)
    tgt_peaks = detect_cells(target, **kw)
    pred_peaks = detect_cells(recon, **kw)
    out = match_cells(pred_peaks, tgt_peaks, match_radius)
    out['detection_mode'] = mode
    return out

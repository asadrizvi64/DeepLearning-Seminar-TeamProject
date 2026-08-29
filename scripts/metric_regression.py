"""Regression tests for the evaluation metrics themselves.

Metric bugs have already produced several false intermediate conclusions in this project
(MIP-vs-3D target counts, an absolute detection threshold, flat regions registering as
peaks, budget-dependent ranking). Each test below pins one of those failure classes so
it cannot silently return.

No pytest in this environment, so this is a self-contained runner:
    PASS  behaviour is correct and pinned
    WARN  behaviour is a known limitation that callers must handle explicitly
    FAIL  a real bug

Usage:
    python scripts/metric_regression.py
"""
import sys

import numpy as np

from volsplat.cellmetrics import detect_cells, match_cells, score_cells

RESULTS = []


def record(name, status, detail):
    RESULTS.append((name, status, detail))
    tag = {'PASS': 'PASS', 'WARN': 'WARN', 'FAIL': 'FAIL'}[status]
    print(f'  [{tag}] {name}\n         {detail}')


def blob_volume(shape=(40, 64, 64), centres=None, sigma=3.0, amp=1.0, bg=0.0):
    """Synthetic volume with isotropic Gaussian blobs on a flat background."""
    if centres is None:
        centres = [(10, 16, 16), (10, 16, 48), (10, 48, 16), (30, 32, 32)]
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
    vol = np.full(shape, bg, dtype=np.float32)
    for (cz, cy, cx) in centres:
        d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        vol += amp * np.exp(-d2 / (2 * sigma ** 2))
    return np.clip(vol, 0, None), centres


# ------------------------------------------------------------------ 1. identity

def test_perfect_reconstruction():
    vol, _ = blob_volume()
    out = score_cells(vol, vol, mode='3d', threshold_abs=0.3, min_distance=8)
    ok = np.isclose(out['cell_f1'], 1.0)
    record('perfect reconstruction scores F1 = 1',
           'PASS' if ok else 'FAIL',
           f"F1={out['cell_f1']:.3f}  pred={out['pred_cell_count']} "
           f"target={out['target_cell_count']}")


# ------------------------------------------------------------------ 2. flat regions

def test_flat_region_no_peaks():
    """A constant volume must yield NO peaks. Without a prominence guard,
    maximum_filter(img)==img holds at every voxel of a plateau, so a reconstruction
    carrying an explicit background term would report a peak everywhere."""
    flat = np.full((32, 48, 48), 0.6, dtype=np.float32)
    peaks = detect_cells(flat, mode='3d', threshold_abs=0.3, min_distance=8)
    ok = len(peaks) == 0
    record('constant volume yields no peaks',
           'PASS' if ok else 'FAIL',
           f'{len(peaks)} peaks found in a flat volume (expected 0)')


def test_blobs_on_plateau():
    """Real blobs sitting on a plateau must still be found once the guard is in."""
    vol, centres = blob_volume(bg=0.5)
    peaks = detect_cells(vol, mode='3d', threshold_abs=0.6, min_distance=8)
    ok = len(peaks) == len(centres)
    record('blobs on a plateau are still detected',
           'PASS' if ok else 'FAIL',
           f'{len(peaks)} peaks for {len(centres)} blobs on a 0.5 pedestal')


# ------------------------------------------------------------------ 3. background offset

def test_background_offset_invariance():
    """Adding a constant to the whole volume must not change WHICH peaks are found.

    With an ABSOLUTE threshold it does, and the failure only shows up for blobs whose
    peak straddles the threshold -- exactly the low-contrast population this project
    cares about. This is the confound that made the Stage 0 object-only reconstruction
    (near-zero background) unscoreable against the full one (pedestal ~0.65): the same
    nucleus clears an absolute threshold in one and not the other.
    """
    # dim blobs: peak amplitude 0.30, threshold 0.35 -> below without a pedestal,
    # above with one. Bright blobs would mask the bug entirely.
    vol, centres = blob_volume(amp=0.30, bg=0.0)
    a = detect_cells(vol, mode='3d', threshold_abs=0.35, min_distance=8)
    b = detect_cells(vol + 0.40, mode='3d', threshold_abs=0.35, min_distance=8)
    same = len(a) == len(b)
    record('detection is invariant to a background offset',
           'PASS' if same else 'WARN',
           f'dim blobs (amp 0.30), threshold_abs=0.35: {len(a)} peaks at bg=0.0 vs '
           f'{len(b)} at bg=0.40 (of {len(centres)} true). '
           + ('' if same else 'Absolute thresholds are NOT offset-invariant. Never '
              'compare reconstructions with different DC levels on an absolute '
              'threshold -- use a relative threshold, or add the background back '
              'before scoring.'))


# ------------------------------------------------------------------ 4. 3D vs MIP

def test_3d_vs_mip_disagree():
    """MIP scoring collapses the z axis, so overlapping nuclei merge and the target
    count drops. Pinning the discrepancy stops it being read as a model difference."""
    vol, centres = blob_volume(centres=[(10, 32, 32), (30, 32, 32)], sigma=3.0)
    n3 = len(detect_cells(vol, mode='3d', threshold_abs=0.3, min_distance=8))
    nm = len(detect_cells(vol, mode='mip', threshold_abs=0.3, min_distance=8))
    ok = (n3 == 2 and nm == 1)
    record('MIP merges z-stacked nuclei that 3D separates',
           'PASS' if ok else 'WARN',
           f'two nuclei stacked in z: 3D finds {n3}, MIP finds {nm}. '
           'Never compare an F1 computed in one mode against the other.')


# ------------------------------------------------------------------ 5. anisotropy

def test_anisotropic_matching():
    """Matching in raw voxel units is wrong on anisotropic data. With DRO spacing
    (z 5x coarser), two points 3 voxels apart in z are 6.09 um apart, while 3 voxels
    apart in x are 1.22 um -- a voxel-space radius treats them identically."""
    V = np.array([2.03, 0.406, 0.406])
    tgt = np.array([[10.0, 20.0, 20.0]])
    off_z = np.array([[13.0, 20.0, 20.0]])       # 3 voxels in z  = 6.09 um
    off_x = np.array([[10.0, 20.0, 23.0]])       # 3 voxels in x  = 1.22 um

    vox_z = match_cells(off_z, tgt, match_radius=4.0)['recall']
    vox_x = match_cells(off_x, tgt, match_radius=4.0)['recall']
    phys_z = match_cells(off_z * V, tgt * V, match_radius=3.0)['recall']
    phys_x = match_cells(off_x * V, tgt * V, match_radius=3.0)['recall']

    voxel_blind = (vox_z == vox_x)
    physical_separates = (phys_z == 0.0 and phys_x == 1.0)
    ok = voxel_blind and physical_separates
    record('physical-distance matching separates what voxel matching cannot',
           'PASS' if ok else 'FAIL',
           f'voxel radius 4: z-offset recall {vox_z:.0f}, x-offset {vox_x:.0f} '
           f'(identical, wrong). physical radius 3um: z {phys_z:.0f}, x {phys_x:.0f}. '
           'Always match in microns on anisotropic data.')


# ------------------------------------------------------------------ 6. ranking / budget

def test_recall_monotonic_in_budget():
    """Recall at a fixed detector must be non-decreasing in budget. A drop means the
    ranking or the top-k truncation is inconsistent."""
    rng = np.random.default_rng(0)
    vol, centres = blob_volume(sigma=2.5)
    vol = vol + rng.normal(0, 0.02, vol.shape).astype(np.float32)
    peaks = detect_cells(vol, mode='3d', threshold_abs=0.2, min_distance=4,
                         max_peaks=100000)
    tgt = np.array(centres, dtype=np.float32)
    recalls = []
    for budget in [1, 2, 4, 8, 16, 64]:
        r = match_cells(peaks[:budget], tgt, match_radius=3.0)['recall']
        recalls.append(r)
    ok = all(b >= a - 1e-9 for a, b in zip(recalls, recalls[1:]))
    record('recall is non-decreasing in detection budget',
           'PASS' if ok else 'FAIL',
           'budgets 1,2,4,8,16,64 -> ' + ', '.join(f'{r:.2f}' for r in recalls))


def test_match_radius_sanity():
    """A pair further apart than the radius must not match."""
    tgt = np.array([[0.0, 0.0, 0.0]])
    near = np.array([[0.0, 0.0, 1.0]])
    far = np.array([[0.0, 0.0, 9.0]])
    ok = (match_cells(near, tgt, 3.0)['matched_peaks'] == 1 and
          match_cells(far, tgt, 3.0)['matched_peaks'] == 0)
    record('match radius is enforced',
           'PASS' if ok else 'FAIL',
           'a pair at distance 9 must not match within radius 3')


def main():
    print('Metric regression suite\n' + '=' * 72)
    for fn in [test_perfect_reconstruction,
               test_flat_region_no_peaks,
               test_blobs_on_plateau,
               test_background_offset_invariance,
               test_3d_vs_mip_disagree,
               test_anisotropic_matching,
               test_recall_monotonic_in_budget,
               test_match_radius_sanity]:
        try:
            fn()
        except Exception as e:
            record(fn.__name__, 'FAIL', f'{type(e).__name__}: {e}')

    n_fail = sum(1 for _, s, _ in RESULTS if s == 'FAIL')
    n_warn = sum(1 for _, s, _ in RESULTS if s == 'WARN')
    print('=' * 72)
    print(f'{len(RESULTS)} checks: {len(RESULTS)-n_fail-n_warn} pass, '
          f'{n_warn} warn (known limitations), {n_fail} FAIL')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())

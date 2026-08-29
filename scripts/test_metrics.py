"""Regression suite for the evaluation code.

Every test here encodes a bug that ACTUALLY occurred in this project and changed a
conclusion before it was caught. Evaluator faults have reversed findings repeatedly, so
these run before trusting any new experiment.

    python scripts/test_metrics.py

No pytest dependency -- plain asserts, prints a pass/fail table, exits non-zero on
failure so it can gate a run.
"""
import sys
import traceback

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from volsplat.cellmetrics import detect_cells, match_cells, score_cells
from volsplat import metrics as M

RESULTS = []


def test(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, True, ''))
        except Exception as e:
            RESULTS.append((name, False, f'{type(e).__name__}: {e}'))
            if '-v' in sys.argv:
                traceback.print_exc()
        return fn
    return deco


# ---------------------------------------------------------------- detection

@test('flat field yields zero peaks')
def _():
    """BUG: maximum_filter(img)==img holds at EVERY voxel of a plateau, so a constant
    region reported a peak per voxel. Surfaced once an explicit background term made
    far-from-Gaussian regions exactly constant."""
    flat = np.full((32, 48, 48), 0.7, dtype=np.float32)
    assert len(detect_cells(flat, mode='3d')) == 0, 'constant volume produced peaks'
    assert len(detect_cells(flat, mode='mip')) == 0, 'constant MIP produced peaks'


@test('single blob is found exactly once')
def _():
    v = np.zeros((32, 48, 48), np.float32)
    zz, yy, xx = np.mgrid[0:32, 0:48, 0:48]
    v += np.exp(-(((zz - 16) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2) / (2 * 3.0 ** 2)))
    pk = detect_cells(v, mode='3d', threshold_abs=0.35)
    assert len(pk) == 1, f'expected 1 peak, got {len(pk)}'
    assert np.allclose(pk[0], [16, 24, 24], atol=1.5), f'peak misplaced: {pk[0]}'


@test('absolute threshold is NOT background-offset invariant (documented sensitivity)')
def _():
    """threshold_abs compares to a fixed value, so adding a DC pedestal changes what is
    detected. Any comparison between a fit WITH an explicit background term and one
    WITHOUT must therefore not rely on an absolute threshold."""
    v = np.zeros((32, 48, 48), np.float32)
    zz, yy, xx = np.mgrid[0:32, 0:48, 0:48]
    for c in [(10, 16, 16), (22, 32, 32)]:
        v += 0.5 * np.exp(-(((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / 18.0))
    low = len(detect_cells(v, mode='3d', threshold_abs=0.35))
    high = len(detect_cells(v + 0.3, mode='3d', threshold_abs=0.35))
    assert low != high, ('offset invariance would be a surprise here; if this test '
                         'starts passing trivially the threshold semantics changed')


@test('mip and 3d detection are not interchangeable')
def _():
    """BUG: score_cells defaulted to mode='mip', giving 15 targets where 3d gave 24.
    F1 granularity became 1/15 and seed noise swamped the effect under study."""
    rng = np.random.default_rng(0)
    v = np.zeros((32, 64, 64), np.float32)
    zz, yy, xx = np.mgrid[0:32, 0:64, 0:64]
    # two blobs stacked along z -> merge under MIP, separate in 3D
    for c in [(10, 32, 32), (22, 32, 32)]:
        v += np.exp(-(((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / 12.0))
    n_mip = len(detect_cells(v, mode='mip', min_distance=5))
    n_3d = len(detect_cells(v, mode='3d', min_distance=5))
    assert n_3d > n_mip, (f'3d should separate z-stacked blobs that MIP merges '
                          f'(3d={n_3d}, mip={n_mip})')


# ---------------------------------------------------------------- matching

@test('matching is symmetric and perfect on identical sets')
def _():
    pts = np.array([[1., 2., 3.], [10., 12., 14.], [20., 5., 7.]])
    r = match_cells(pts, pts, match_radius=1.0)
    assert r['recall'] == 1.0 and r['precision'] == 1.0, r
    assert r['cell_f1'] == 1.0


@test('matching is one-to-one, not greedy double-counting')
def _():
    """Two predictions near one target must not both count as matches."""
    tgt = np.array([[10., 10., 10.]])
    pred = np.array([[10., 10., 10.], [10.5, 10., 10.]])
    r = match_cells(pred, tgt, match_radius=2.0)
    assert r['matched_peaks'] == 1, f"expected 1 match, got {r['matched_peaks']}"
    assert abs(r['precision'] - 0.5) < 1e-9, r['precision']


@test('anisotropic voxels require physical-distance matching')
def _():
    """BUG: a 5-voxel Euclidean radius spans 2.03 um in xy but 10.15 um in z on DRO
    voxels, so voxel-space matching silently accepts far-apart pairs along z."""
    voxel_zyx = np.array([2.03, 0.406, 0.406])
    tgt = np.array([[10., 10., 10.]])
    far_z = np.array([[13., 10., 10.]])      # 3 voxels in z = 6.09 um
    d_vox = np.linalg.norm(far_z - tgt)
    d_um = np.linalg.norm((far_z - tgt) * voxel_zyx)
    assert d_vox < 4.0, d_vox
    assert d_um > 5.0, d_um
    # voxel-space matching accepts it; physical-space matching rejects it
    assert match_cells(far_z, tgt, match_radius=4.0)['matched_peaks'] == 1
    assert match_cells(far_z * voxel_zyx, tgt * voxel_zyx,
                       match_radius=4.0)['matched_peaks'] == 0


@test('detection footprint must be physical on anisotropic voxels')
def _():
    """BUG (third instance of this class): detect_cells used a CUBIC suppression window
    sized in voxels. On DRO voxels (2.03, 0.406, 0.406 um) a 13-voxel cube spans 26 um
    in z against 5.3 um in xy, so it merges nuclei along z. Measured on the RAW TARGET
    against 29 human-annotated nuclei: cubic recovered 5/29, physical 22/29. Scores
    computed with the cubic default on anisotropic data are capped by the SCORER."""
    voxel = (2.03, 0.406, 0.406)
    v = np.zeros((40, 64, 64), np.float32)
    zz, yy, xx = np.mgrid[0:40, 0:64, 0:64]
    # Two nuclei ~6 um apart along z == 3 voxels apart. Amplitudes must DIFFER: with
    # equal peaks both satisfy `maximum_filter == img` and survive the tie, which would
    # hide the merging the test is looking for.
    for cz, amp in ((18, 1.0), (21, 0.8)):
        v += amp * np.exp(-(((zz - cz) * voxel[0]) ** 2 + ((yy - 32) * voxel[1]) ** 2
                            + ((xx - 32) * voxel[2]) ** 2) / (2 * 1.6 ** 2))
    cubic = detect_cells(v, mode='3d', threshold_abs=0.0, min_distance=13,
                         smoothing_sigma=1.0, prominence=1e-3)
    phys = detect_cells(v, mode='3d', threshold_abs=0.0, min_distance=4.0,
                        smoothing_sigma=0.8, prominence=1e-3,
                        voxel_size_zyx=voxel)
    assert len(cubic) < len(phys), (
        f'cubic window should merge the z-separated pair that a physical window '
        f'resolves (cubic={len(cubic)}, physical={len(phys)})')


@test('empty prediction scores zero, not NaN or crash')
def _():
    tgt = np.array([[1., 1., 1.]])
    r = match_cells(np.empty((0, 3)), tgt, match_radius=3.0)
    assert r['recall'] == 0.0 and r['cell_f1'] == 0.0, r


# ---------------------------------------------------------------- ranking

@test('recall is monotone in budget, so budgets must be matched')
def _():
    """BUG: LoG was called 'the better ranking' from small-budget numbers, then used at
    a large budget where raw intensity wins (R@5000 0.868 vs 0.540). Comparing rankings
    at different budgets is meaningless because recall only ever increases."""
    rng = np.random.default_rng(1)
    tgt = rng.uniform(0, 100, size=(50, 3))
    det = np.vstack([tgt + rng.normal(0, 0.3, tgt.shape), rng.uniform(0, 100, (200, 3))])
    rng.shuffle(det)
    prev = -1.0
    for n in [10, 50, 100, 250]:
        d = np.linalg.norm(tgt[:, None, :] - det[None, :n, :], axis=-1)
        rec = float((d.min(axis=1) <= 2.0).mean())
        assert rec >= prev - 1e-12, 'recall decreased with a larger budget'
        prev = rec


# ---------------------------------------------------------------- fidelity

@test('psnr of a perfect reconstruction is infinite')
def _():
    a = np.random.default_rng(0).uniform(0, 1, (8, 16, 16)).astype(np.float32)
    assert np.isinf(M.psnr(a, a))


@test('psnr matches the closed form for a known error')
def _():
    a = np.zeros((8, 16, 16), np.float32)
    b = np.full_like(a, 0.1)
    assert abs(M.psnr(b, a) - 20.0) < 1e-6, M.psnr(b, a)


@test('parameter counts follow the stated DOF ladder')
def _():
    assert M.param_count(100, 'isotropic') == 500
    assert M.param_count(100, 'diagonal') == 700
    assert M.param_count(100, 'full') == 1000


# ---------------------------------------------------------------- fitting invariants

@test('isotropic parameterization really ties the three scales')
def _():
    import torch
    from volsplat.init import init_gaussians
    from volsplat.densify import project_parameterization
    v = np.random.default_rng(0).uniform(0, 1, (16, 32, 32)).astype(np.float32)
    gs = init_gaussians(v, 20, strategy='intensity_weighted', init_scale=2.0, seed=0)
    with torch.no_grad():
        gs.log_scales += torch.randn_like(gs.log_scales)     # break the tie
    project_parameterization(gs, 'isotropic')
    sc = gs.scales.detach().cpu().numpy()
    spread = float(np.abs(sc - sc.mean(1, keepdims=True)).max())
    assert spread < 1e-6, f'isotropic scales not tied: spread={spread}'


@test('validation split leaks nothing and covers everything')
def _():
    import torch
    from volsplat.ablation import make_val_split
    n = 10000
    tr, va = make_val_split(n, 0.1, seed=1234)
    assert len(set(tr.tolist()) & set(va.tolist())) == 0, 'train/val overlap'
    assert len(tr) + len(va) == n
    tr2, va2 = make_val_split(n, 0.1, seed=1234)
    assert torch.equal(va, va2), 'split is not reproducible'


# ---------------------------------------------------------------- report

def main():
    print(f'{"test":58s} {"result":>8s}')
    print('-' * 70)
    for name, ok, err in RESULTS:
        print(f'{name:58s} {"PASS" if ok else "FAIL":>8s}')
        if not ok:
            print(f'    {err}')
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print('-' * 70)
    print(f'{len(RESULTS) - n_fail}/{len(RESULTS)} passed')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())

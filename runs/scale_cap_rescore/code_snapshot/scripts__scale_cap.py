"""Causal test: does constraining spatial scale rescue faint nuclei?

Stage 3 trajectory analysis established the failure SIGNATURE, not yet the cause:

    lost nuclei final scale 11.72  vs  kept 9.07   (p = 0.0010)
    amplitude indistinguishable                    (p = 0.63)
    post-Adam effective updates indistinguishable  (not starvation)
    neighbour capture / detector-only loss         (0 cases)
    crowding REVERSED -- lost nuclei are more isolated (p = 0.025)

HYPOTHESIS
    Low-contrast seeded nuclei are lost because MSE permits their Gaussians to grow in
    spatial scale and explain surrounding low-frequency intensity, rather than
    preserving a localized nucleus peak.

INTERVENTION -- a hard cap on scale for the NUCLEUS-SEEDED Gaussians only, and nothing
else. The cap must NOT be global: measured, a global cap at 10 voxels pins the entire
population (median = max = 10.00) and costs 9.6 dB on its own, because SUPPORT Gaussians
legitimately need large scales (max observed 25.56) to cover broad tissue. Capping them
would confound S_B with a global reconstruction collapse. With the coverage initializer
the nucleus Gaussians are exactly indices 0..n_nuclei-1, so the cap targets precisely the
population the hypothesis is about. Anisotropy regularization is
deliberately NOT enabled: that is a second, separate manipulation.

The cap ladder is set from the MEASURED distribution rather than as a fraction of the
init scale. Kept nuclei naturally reach 4.5x their init scale (9.07 voxels from
init_scale=2.0), so a cap at 2x init (4 voxels) would sit far below what SUCCESSFUL
nuclei need and would crush every arm uninformatively. The discriminating range is
between 4.5x (kept) and 6.2x (lost):

    control  no cap
    mild     12 vox  (6.0x init) -- just below the B-lost median
    medium   10 vox  (5.0x init) -- between kept and lost
    strict    8 vox  (4.0x init) -- just below the A-kept median

PRE-REGISTERED
    primary     S_B increases by > 0.05
    guard       |delta S_A| < 0.05
    manip check s(B-lost) decreases, and the kept/lost scale gap shrinks

An accompanying PSNR drop is EXPECTED and is not a failure: the archive already records
that anisotropy regularization cost 1.15 dB, and nobody measured its effect on nucleus
recovery. Localization and reconstruction fidelity are hypothesized to compete through
scale.

Everything else is held at the Stage 3 configuration, including the ORIGINAL position
learning rate -- the point is to explain the regime in which Stage 3 was measured, not
to mix in the corrected-migration regime.

Usage:
    python scripts/scale_cap.py --regions 4 --k 250 --iters 3000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from volsplat.init import init_gaussians
from volsplat.ablation import fit_with_validation
from volsplat.cellmetrics import detect_cells, match_indices
from volsplat.provenance import run_metadata
from volsplat.tribolium import VOXEL_SIZE_LUND_UM
from volsplat import metrics as M

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/scale_cap'
SHAPE = (64, 128, 128)
REGIONS = [(0, 528, 144), (0, 624, 144), (0, 240, 336), (0, 528, 240)]
SEED_R = 3.0
NUC_VOXEL_R = 8.0
ARMS = [('control', None), ('mild', 12.0), ('medium', 10.0), ('strict', 8.0)]

# --- Physical-unit correction (previously missing entirely on this script) ---
# Tribolium/Lund voxels are anisotropic: 0.6934 x 0.6934 x 3.0 um -- z is 4.33x coarser
# than xy. `detect_cells(..., mode='3d')` was being called with NO voxel_size_zyx, so
# its NMS window was a cube in VOXEL space, i.e. physically a squashed ellipsoid (39 um
# tall in z vs 9.0 um wide in xy at min_distance=13) -- the exact bug class already
# fixed for Drosophila (cellmetrics.py docstring, +0.587 recall on raw-target detection
# there).
#
# `volsplat.tribolium.VOXEL_SIZE_LUND_UM` is stored (x, y, z) per its own docstring --
# `detect_cells`'s `voxel_size_zyx` parameter wants (z, y, x). Passing it unreversed
# would swap the finest and coarsest axes (apply 3.0 um to x, 0.6934 um to z) -- the
# reverse must be taken explicitly:
VOXEL_ZYX = tuple(reversed(VOXEL_SIZE_LUND_UM))  # (3.0, 0.6934, 0.6934) = (dz, dy, dx)

# Two numbers must be re-expressed in microns to use `voxel_size_zyx`, and both are
# read off the ALREADY-CALIBRATED voxel values rather than invented fresh:
# `min_distance=13` and the resulting `MATCH_R=0.765*13` were tuned against the
# colleague's `cell_count_parameters.json`, itself set by eye on MIP (i.e. XY-plane)
# images -- so the physically-intended spacing is 13 voxels of XY, not 13 voxels of an
# isotropic cube. Converting via the XY pixel size preserves that calibration instead
# of silently changing it:
_DZ, _DY, _DX = VOXEL_ZYX
MIN_DIST_UM = 13 * _DX            # 13 xy-voxels -> ~9.01 um (matches XY calibration)
SMOOTH_SIGMA_UM = 2.0 * _DX       # 2 xy-voxels -> ~1.39 um
MATCH_R_UM = 0.765 * MIN_DIST_UM  # ~6.89 um -- same 0.765 ratio, now in physical units
DETECT_KW = dict(mode='3d', min_distance=MIN_DIST_UM, smoothing_sigma=SMOOTH_SIGMA_UM,
                 voxel_size_zyx=VOXEL_ZYX)


def nucleus_mask(shape, nuclei, radius=NUC_VOXEL_R):
    D, H, W = shape
    m = np.zeros(shape, dtype=bool)
    r = int(np.ceil(radius))
    for z, y, x in nuclei.astype(int):
        z0, z1 = max(0, z - r), min(D, z + r + 1)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
        m[z0:z1, y0:y1, x0:x1] |= ((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) <= radius ** 2
    return m


def recovered(recon, nuclei):
    """Which target `nuclei` (voxel zyx) are recovered in `recon`.

    Uses the shared `match_indices` (maximum-cardinality matching within the radius,
    not Hungarian-then-threshold -- see cellmetrics.py docstring for the counterexample
    that motivated this) so this path and `cellmetrics.match_cells` cannot silently
    diverge, per instruction: both call paths must use the same implementation.
    Matching is done in PHYSICAL microns (coordinates scaled by VOXEL_SIZE_LUND_UM),
    consistent with `DETECT_KW` sizing detection physically too -- see
    scripts/test_metrics.py / scripts/capstone_transfer.py for the same pattern.
    """
    pred = detect_cells(recon, **DETECT_KW)
    got = np.zeros(len(nuclei), dtype=bool)
    if len(pred) and len(nuclei):
        voxel = np.asarray(VOXEL_ZYX)  # pred/nuclei are (z,y,x) coords -> scale (z,y,x)
        _, ci, _ = match_indices(pred * voxel, nuclei * voxel, MATCH_R_UM)
        got[ci] = True
    return got


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--regions', type=int, default=4)
    p.add_argument('--k', type=int, default=250)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    vol = np.load(REPO / 'runs/colleague_full_volume.npy').astype(np.float32)
    rows, per_nuc = [], []

    for (z, y, x) in REGIONS[:args.regions]:
        tag = f'z{z}y{y}x{x}'
        roi = np.ascontiguousarray(vol[z:z + SHAPE[0], y:y + SHAPE[1], x:x + SHAPE[2]])
        nuclei = detect_cells(roi, **DETECT_KW)
        nmask = nucleus_mask(roi.shape, nuclei)

        old = init_gaussians(roi, args.k, strategy='local_maxima', init_scale=2.0, seed=0)
        p_old = old.positions.detach().cpu().numpy()[:, ::-1]
        d_old = np.linalg.norm(nuclei[:, None, :] - p_old[None, :, :], axis=-1).min(axis=1)
        in_A = d_old <= SEED_R
        print(f'\n=== {tag}: {len(nuclei)} nuclei (A={int(in_A.sum())}, B={int((~in_A).sum())}) ===')

        for arm, cap in ARMS:
            gs, _, res = fit_with_validation(
                roi, num_gaussians=args.k, iterations=args.iters,
                init_strategy='coverage', parameterization='full', init_scale=2.0,
                seed=args.seed, max_scale=cap, max_scale_n=len(nuclei),
                full_recon=False)
            recon = gs.query_volume(roi.shape).cpu().numpy().astype(np.float32)
            got = recovered(recon, nuclei)

            pos = gs.positions.detach().cpu().numpy()
            sc = gs.scales.detach().cpu().numpy().mean(axis=1)
            amp = gs.amplitudes.detach().cpu().numpy()
            n_nuc = len(nuclei)                      # coverage init: 0..n-1 are nuclei

            rows.append(dict(
                region=tag, arm=arm, cap=cap,
                S_A=float(got[in_A].mean()) if in_A.any() else np.nan,
                S_B=float(got[~in_A].mean()) if (~in_A).any() else np.nan,
                recall=float(got.mean()),
                scale_kept=float(np.median(sc[:n_nuc][got])) if got.any() else np.nan,
                scale_lost=float(np.median(sc[:n_nuc][~got])) if (~got).any() else np.nan,
                scale_B_lost=float(np.median(sc[:n_nuc][(~in_A) & (~got)]))
                    if ((~in_A) & (~got)).any() else np.nan,
                amp_final=float(np.median(amp[:n_nuc])),
                psnr_nucleus=M.psnr(recon[nmask], roi[nmask]),
                psnr_background=M.psnr(recon[~nmask], roi[~nmask]),
                psnr_global=M.psnr(recon, roi),
                val_psnr=res['val_psnr'],
                **run_metadata(experiment='scale_cap', arm=arm, max_scale=cap,
                               seed=args.seed, region=tag, k=args.k, iters=args.iters,
                               init='coverage', parameterization='full',
                               init_scale=2.0, lr_position=0.0016)))
            for i in range(n_nuc):
                per_nuc.append(dict(region=tag, arm=arm, nucleus=i,
                                    group='A' if in_A[i] else 'B',
                                    recovered=bool(got[i]), scale=float(sc[i]),
                                    amp=float(amp[i])))
            # Persist after EVERY fit. Two runs have now been lost to a job being
            # killed partway through a script that only wrote its CSV at the end;
            # a partial sweep is still analysable, nothing is not.
            pd.DataFrame(rows).to_csv(OUT / 'scale_cap.csv', index=False)
            pd.DataFrame(per_nuc).to_csv(OUT / 'per_nucleus.csv', index=False)
            print(f'  {arm:8s} cap={str(cap):5s}  S_A={rows[-1]["S_A"]:.3f} '
                  f'S_B={rows[-1]["S_B"]:.3f}  scale kept/lost='
                  f'{rows[-1]["scale_kept"]:.2f}/{rows[-1]["scale_lost"]:.2f}  '
                  f'nucPSNR={rows[-1]["psnr_nucleus"]:.2f}', flush=True)

    df = pd.DataFrame(rows)

    g = df.groupby('arm').agg(
        S_A=('S_A', 'mean'), S_B=('S_B', 'mean'), recall=('recall', 'mean'),
        scale_kept=('scale_kept', 'mean'), scale_lost=('scale_lost', 'mean'),
        scale_B_lost=('scale_B_lost', 'mean'), amp=('amp_final', 'mean'),
        psnr_nuc=('psnr_nucleus', 'mean'), psnr_bg=('psnr_background', 'mean'),
        psnr_glob=('psnr_global', 'mean')).reindex([a for a, _ in ARMS])

    print('\n' + '=' * 78)
    print('SCALE CAP SWEEP (mean over regions)')
    print('=' * 78)
    print(f'{"arm":9s} {"S_A":>7s} {"S_B":>7s} {"recall":>7s} {"s_kept":>7s} '
          f'{"s_lost":>7s} {"s_Blost":>8s} {"nucPSNR":>8s} {"global":>7s}')
    for a in g.index:
        r = g.loc[a]
        print(f'{a:9s} {r.S_A:7.3f} {r.S_B:7.3f} {r.recall:7.3f} {r.scale_kept:7.2f} '
              f'{r.scale_lost:7.2f} {r.scale_B_lost:8.2f} {r.psnr_nuc:8.2f} '
              f'{r.psnr_glob:7.2f}')

    ref = g.loc['control']
    print('\n=== PRE-REGISTERED CRITERIA (each arm vs control) ===')
    verdicts = {}
    for a in g.index:
        if a == 'control':
            continue
        r = g.loc[a]
        dB, dA = r.S_B - ref.S_B, r.S_A - ref.S_A
        gap_ref = ref.scale_lost - ref.scale_kept
        gap_arm = r.scale_lost - r.scale_kept
        c1, c2 = dB > 0.05, abs(dA) < 0.05
        c3 = r.scale_B_lost < ref.scale_B_lost
        c4 = gap_arm < gap_ref
        # BUG this replaces: `causal=bool(c1 and c2)` dropped c3/c4 from the verdict
        # even though they are the pre-registered MANIPULATION checks (the intervention
        # did what it claims) -- the script could print "CAUSAL" while its own stated
        # manipulation criteria failed. All four pre-registered conditions now gate it.
        causal = bool(c1 and c2 and c3 and c4)
        verdicts[a] = dict(dS_B=float(dB), dS_A=float(dA),
                           manip_scale=bool(c3), manip_gap=bool(c4),
                           primary=bool(c1), guard=bool(c2),
                           causal=causal)
        print(f'  {a:8s} S_B {dB:+.3f} [{"PASS" if c1 else "FAIL"}]  '
              f'S_A {dA:+.3f} [{"PASS" if c2 else "FAIL"}]  '
              f'manip: s_Blost {ref.scale_B_lost:.2f}->{r.scale_B_lost:.2f} '
              f'[{"ok" if c3 else "no"}]  gap {gap_ref:.2f}->{gap_arm:.2f} '
              f'[{"ok" if c4 else "no"}]  '
              f'=> {"CAUSAL" if causal else "not established"}')

    best = max(verdicts, key=lambda a: verdicts[a]['dS_B']) if verdicts else None
    print(f'\n  PSNR trade: control {ref.psnr_glob:.2f} dB'
          + (f'  ->  {best} {g.loc[best].psnr_glob:.2f} dB '
             f'({g.loc[best].psnr_glob - ref.psnr_glob:+.2f})' if best else ''))
    json.dump(dict(summary=g.reset_index().to_dict('records'), verdicts=verdicts),
              open(OUT / 'verdict.json', 'w'), indent=2)
    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

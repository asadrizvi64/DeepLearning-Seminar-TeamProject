"""Does Gaussian fitting RE-RANK nucleus candidates better than the raw image?

WHY NOT THE OBVIOUS EXPERIMENT. "Do Gaussians beat a detector at finding nuclei" is
predetermined and worthless here: the fit is SEEDED by a detector, and we measured that
100% of recovered nuclei had a seed within 3 voxels while nothing unseeded was ever
recovered. So recall_fit <= recall_raw by construction; running it would re-measure the
initializer.

THE QUESTION THAT IS ACTUALLY OPEN. The non-circular manual-label result showed that
candidate GENERATION is adequate and RANKING is the bottleneck: 87% of manual nuclei have
a candidate within 3 um, but their median intensity rank is 1912 of 6568 and only 4.9%
reach the top 189. So:

    given the SAME candidate set, does fitting promote true nuclei up the ranking?

Every arm is therefore scored at a MATCHED detection count N. That also repairs the
sparse-annotation problem: Fluo-N3DL-DRO annotates a selected subset, so absolute
precision is uninterpretable -- but at fixed N,

    precision@N = matched/N        recall@N = matched/n_manual

are proportional, so the COMPARISON between arms is valid even though the absolute
precision is not. The sparse-annotation bias is identical across arms.

ARMS
    raw          detect on the target volume, rank, take top N          (deterministic)
    downsample   downsample + upsample at MATCHED STORAGE, detect       (deterministic)
    gauss_lr*    seed K Gaussians at the top-K RAW candidates, fit,
                 detect in the reconstruction                           (3 seeds)

The Gaussian arms are seeded from the SAME detector output the `raw` arm is scored on,
so the only thing that differs is the fit. Two position learning rates are run: the one
the Tribolium results were measured at, and the corrected value -- their combination with
anything else has never been tested, so they are reported separately, never merged.

RESOLUTION LIMIT, stated before the run: this crop holds ~29 annotated nuclei, so one
nucleus is ~3.4 recall points. Differences below ~10 points are not resolvable here.

Usage:
    python scripts/rerank_vs_baselines.py --k 500 --iters 3000 --seeds 0 1 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import zoom
from scipy.optimize import linear_sum_assignment

from volsplat.ctc import percentile_normalize
from volsplat.ablation import fit_with_validation
from volsplat.cellmetrics import detect_cells

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/rerank'
DRO = Path(r'D:/Download/train/Fluo-N3DL-DRO (1)/Fluo-N3DL-DRO')
SHAPE = (64, 128, 128)
V = np.array([2.03, 0.406, 0.406])          # um per voxel (z, y, x)
MATCH_UM = 3.0
PROMINENCE = 0.02
SCALARS_PER_GAUSSIAN = 11                   # 3 pos + 3 log-scale + 4 quat + 1 amp
BUDGETS = [29, 50, 100, 200, 300, 500]


def detect(vol):
    """Candidates, strongest first. Physically isotropic window -- the legacy cubic
    default recovers only 5/29 manual nuclei from the RAW target on this data."""
    return detect_cells(vol, mode='3d', threshold_abs=0.0, smoothing_sigma=0.8,
                        min_distance=2.0, prominence=PROMINENCE,
                        voxel_size_zyx=tuple(V))


def recall_at(pred_zyx, manual_zyx, n):
    """Recall using only the top-n predictions, matched in MICRONS."""
    pred = pred_zyx[:n]
    if len(pred) == 0 or len(manual_zyx) == 0:
        return 0.0, 0
    d = np.linalg.norm((manual_zyx * V)[:, None, :] - (pred * V)[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(d)
    matched = int(sum(1 for a, b in zip(ri, ci) if d[a, b] <= MATCH_UM))
    return matched / len(manual_zyx), matched


def downsample_matched(vol, n_scalars):
    """Uniform voxel downsample + trilinear upsample, at the integer factor whose coarse
    grid is closest to `n_scalars` stored values."""
    best = None
    for f in range(2, 17):
        dims = tuple(max(1, s // f) for s in vol.shape)
        n = int(np.prod(dims))
        if best is None or abs(n - n_scalars) < abs(best[1] - n_scalars):
            best = (f, n, dims)
    f, n, dims = best
    coarse = zoom(vol, (dims[0] / vol.shape[0], dims[1] / vol.shape[1],
                        dims[2] / vol.shape[2]), order=1)
    back = zoom(coarse, (vol.shape[0] / coarse.shape[0], vol.shape[1] / coarse.shape[1],
                         vol.shape[2] / coarse.shape[2]), order=1)
    back = back[:vol.shape[0], :vol.shape[1], :vol.shape[2]].astype(np.float32)
    return back, f, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=500)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--lrs', type=float, nargs='+', default=[0.0016, 0.05])
    p.add_argument('--origin', type=int, nargs=3, default=None,
                   help='crop origin z y x; default = the placement holding most nuclei')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- target volume + HUMAN annotation
    vol_full = percentile_normalize(
        tifffile.imread(str(DRO / '01/t000.tif')).astype(np.float32))
    cents = np.load(REPO / 'runs/dro_manual_centroids.npy')

    if args.origin is None:
        best = (0, None)
        for z in range(0, vol_full.shape[0] - SHAPE[0], 16):
            for y in range(0, vol_full.shape[1] - SHAPE[1], 32):
                for x in range(0, vol_full.shape[2] - SHAPE[2], 32):
                    m = ((cents[:, 0] >= z) & (cents[:, 0] < z + SHAPE[0]) &
                         (cents[:, 1] >= y) & (cents[:, 1] < y + SHAPE[1]) &
                         (cents[:, 2] >= x) & (cents[:, 2] < x + SHAPE[2]))
                    if m.sum() > best[0]:
                        best = (int(m.sum()), (z, y, x))
        origin = best[1]
    else:
        origin = tuple(args.origin)
    z, y, x = origin
    roi = np.ascontiguousarray(vol_full[z:z + SHAPE[0], y:y + SHAPE[1], x:x + SHAPE[2]])
    m = ((cents[:, 0] >= z) & (cents[:, 0] < z + SHAPE[0]) &
         (cents[:, 1] >= y) & (cents[:, 1] < y + SHAPE[1]) &
         (cents[:, 2] >= x) & (cents[:, 2] < x + SHAPE[2]))
    manual = cents[m] - np.array([z, y, x])
    print(f'ROI origin {origin}  shape {roi.shape}  |  {len(manual)} HUMAN-annotated nuclei')
    print(f'one nucleus = {100/len(manual):.1f} recall points -> differences below '
          f'~{3*100/len(manual):.0f} points are not resolvable here\n')

    rows = []

    # ---- arm 1: raw
    raw_cand = detect(roi)
    print(f'raw detector: {len(raw_cand)} candidates')
    for n in BUDGETS:
        r, mt = recall_at(raw_cand, manual, n)
        rows.append(dict(arm='raw', seed=-1, N=n, recall=r, matched=mt,
                         n_cand=len(raw_cand)))
        print(f'   recall@{n:<4d} = {r:.3f}  ({mt}/{len(manual)})')

    # ---- arm 2: downsample at matched storage
    n_scalars = args.k * SCALARS_PER_GAUSSIAN
    ds, f, n_ds = downsample_matched(roi, n_scalars)
    ds_cand = detect(ds)
    print(f'\ndownsample x{f} -> {n_ds} stored values (Gaussian model: {n_scalars}), '
          f'{len(ds_cand)} candidates')
    for n in BUDGETS:
        r, mt = recall_at(ds_cand, manual, n)
        rows.append(dict(arm=f'downsample_x{f}', seed=-1, N=n, recall=r, matched=mt,
                         n_cand=len(ds_cand)))
        print(f'   recall@{n:<4d} = {r:.3f}  ({mt}/{len(manual)})')

    # ---- arm 3: Gaussian fit, seeded from the SAME raw candidates
    seed_pts = raw_cand[:args.k]
    print(f'\nseeding {len(seed_pts)} Gaussians at the top-{args.k} RAW candidates '
          f'(detector output, NOT annotation)')
    for lr in args.lrs:
        for sd in args.seeds:
            gs, _, res = fit_with_validation(
                roi, num_gaussians=args.k, iterations=args.iters,
                init_strategy='oracle_coverage',
                init_kwargs={'nuclei': seed_pts},
                parameterization='full', init_scale=2.0, seed=sd,
                lr_position=lr, full_recon=False)
            recon = gs.query_volume(roi.shape).cpu().numpy().astype(np.float32)
            cand = detect(recon)
            line = []
            for n in BUDGETS:
                r, mt = recall_at(cand, manual, n)
                rows.append(dict(arm=f'gauss_lr{lr:g}', seed=sd, N=n, recall=r,
                                 matched=mt, n_cand=len(cand),
                                 val_psnr=res['val_psnr']))
                line.append(f'@{n}={r:.3f}')
            import pandas as pd
            pd.DataFrame(rows).to_csv(OUT / 'rerank.csv', index=False)
            print(f'  lr={lr:<7g} seed={sd}  ' + ' '.join(line) +
                  f'   ({len(cand)} cand)', flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'rerank.csv', index=False)

    print('\n=== recall@N, mean over seeds ===')
    piv = df.pivot_table(index='arm', columns='N', values='recall')
    print(piv.round(3).to_string())

    # A reconstruction built from K Gaussians can yield at most ~K detectable peaks, so
    # its recall SATURATES while `raw` keeps climbing on thousands of candidates.
    # Comparing at an N larger than an arm can supply measures candidate SUPPLY, not
    # ranking quality. Each arm is therefore compared only at N <= its own candidate
    # count, and the headline is read at that operating point.
    ncand = df.groupby('arm').n_cand.mean()
    raw_row = piv.loc['raw']
    res = 3 * 1.0 / len(manual)          # ~3 nuclei, as a recall fraction

    print('\n=== candidates supplied by each arm ===')
    for arm in piv.index:
        print(f'  {arm:16s} {ncand[arm]:7.0f}')

    print('\n=== re-ranking quality vs raw, at MATCHED N ===')
    print(f"  (N above an arm's candidate count is not meaningful; "
          f"resolution limit ~{res:.3f})")
    verdict = {}
    for arm in piv.index:
        if arm == 'raw':
            continue
        valid = [n for n in piv.columns if n <= ncand[arm]]
        if not valid:
            print(f'  {arm:16s} supplies fewer than {min(piv.columns)} candidates')
            continue
        d = piv.loc[arm][valid] - raw_row[valid]
        op_n = int(max(valid))
        op_d = float(d[op_n])
        tag = ('ABOVE raw' if op_d > res else
               'BELOW raw' if op_d < -res else 'tied within resolution')
        verdict[arm] = {'valid_N': [int(v) for v in valid],
                        'delta_by_N': {int(k): float(v) for k, v in d.items()},
                        'operating_N': op_n, 'delta_at_operating_N': op_d,
                        'verdict': tag}
        print(f'  {arm:16s} delta ' +
              ' '.join(f'@{int(k)}={v:+.3f}' for k, v in d.items()) +
              f'   | operating point N={op_n}: {op_d:+.3f} -> {tag}')

    json.dump({'origin': list(origin), 'n_manual': int(len(manual)),
               'k': args.k, 'iters': args.iters, 'seeds': args.seeds,
               'resolution_points': float(100 / len(manual)),
               'recall_at_N': piv.round(4).to_dict(), 'verdict': verdict},
              open(OUT / 'summary.json', 'w'), indent=2)
    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

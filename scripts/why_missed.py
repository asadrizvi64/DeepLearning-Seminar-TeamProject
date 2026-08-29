"""Why are ~20% of nuclei missed?

Takes fitted models, splits the TARGET nuclei into those the fit recovered and those
it did not, and compares the two groups on every property that could plausibly explain
the failure. Each property maps to a different remedy, so the comparison discriminates
between them rather than just describing the gap.

    dimmer          -> MSE weights bright regions more; the loss is the problem
    more crowded    -> two nuclei share one Gaussian; an assignment problem
    deeper in z     -> light-sheet attenuation; an imaging limit, not a model limit
    smaller         -> below the representable scale
    far from any
    initial Gaussian-> initialization never seeded them; a placement problem

Pools over several non-overlapping regions of the embryo so the sample is not one crop.

Usage:
    python scripts/why_missed.py --regions 4 --k 250 --iters 3000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

from volsplat.init import init_gaussians
from volsplat.ablation import fit_with_validation
from volsplat.cellmetrics import detect_cells

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/why_missed'
SHAPE = (64, 128, 128)


def pick_regions(vol, n_regions, shape=SHAPE, min_sep=64):
    """Non-overlapping windows richest in detected nuclei."""
    peaks = detect_cells(vol, mode='3d')
    D, H, W = vol.shape
    tz, ty, tx = shape
    cands = []
    for z in range(0, D - tz + 1, 16):
        for y in range(0, H - ty + 1, 48):
            for x in range(0, W - tx + 1, 48):
                m = ((peaks[:, 0] >= z) & (peaks[:, 0] < z + tz) &
                     (peaks[:, 1] >= y) & (peaks[:, 1] < y + ty) &
                     (peaks[:, 2] >= x) & (peaks[:, 2] < x + tx))
                n = int(m.sum())
                if n >= 8:
                    cands.append((n, z, y, x))
    cands.sort(reverse=True)
    chosen = []
    for n, z, y, x in cands:
        if all(abs(y - cy) >= min_sep or abs(x - cx) >= min_sep
               for _, _, cy, cx in chosen):
            chosen.append((n, z, y, x))
        if len(chosen) >= n_regions:
            break
    return chosen


def nucleus_properties(vol, peaks, half=10):
    """Per-nucleus brightness, contrast, size, crowding, depth."""
    sm = gaussian_filter(vol.astype(np.float32), 1.5)
    D, H, W = vol.shape
    props = []
    # crowding: distance to nearest OTHER target nucleus
    if len(peaks) > 1:
        dd = np.linalg.norm(peaks[:, None, :] - peaks[None, :, :], axis=-1)
        np.fill_diagonal(dd, np.inf)
        nn = dd.min(axis=1)
    else:
        nn = np.full(len(peaks), np.inf)

    for i, (z, y, x) in enumerate(peaks.astype(int)):
        z0, z1 = max(0, z - half), min(D, z + half + 1)
        y0, y1 = max(0, y - half), min(H, y + half + 1)
        x0, x1 = max(0, x - half), min(W, x + half + 1)
        patch = sm[z0:z1, y0:y1, x0:x1]
        peak_val = float(sm[z, y, x])
        bg = float(np.percentile(patch, 20))
        # second-moment size above half-max
        thr = patch.max() * 0.5
        w = np.clip(patch - thr, 0, None)
        if w.sum() > 0:
            zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
            tot = w.sum()
            cz = (w * zz).sum() / tot; cy = (w * yy).sum() / tot; cx = (w * xx).sum() / tot
            sz = np.sqrt((w * (zz - cz) ** 2).sum() / tot)
            sy = np.sqrt((w * (yy - cy) ** 2).sum() / tot)
            sx = np.sqrt((w * (xx - cx) ** 2).sum() / tot)
            size = float((sx + sy + sz) / 3)
        else:
            size = np.nan
        props.append(dict(peak=peak_val, contrast=peak_val - bg, size=size,
                          nn_dist=float(nn[i]), depth_z=int(z)))
    return props


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--regions', type=int, default=4)
    p.add_argument('--k', type=int, default=250)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--inits', nargs='+', default=['local_maxima'],
                   help='one or more init strategies to compare')
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    vol = np.load(REPO / 'runs/colleague_full_volume.npy').astype(np.float32)
    regions = pick_regions(vol, args.regions)
    print(f'Selected {len(regions)} regions: ' +
          ', '.join(f'(z{z},y{y},x{x}) n={n}' for n, z, y, x in regions) + '\n')

    rows = []
    for init_strategy in args.inits:
        print(f'--- init: {init_strategy} ---')
        for ri, (npk, z, y, x) in enumerate(regions):
            roi = np.ascontiguousarray(vol[z:z + SHAPE[0], y:y + SHAPE[1], x:x + SHAPE[2]])
            tgt = detect_cells(roi, mode='3d')

            # initial Gaussian positions (same seed/strategy the fit will use)
            gs0 = init_gaussians(roi, args.k, strategy=init_strategy, init_scale=2.0,
                                 seed=args.seed)
            p0 = gs0.positions.detach().cpu().numpy()[:, ::-1]    # (x,y,z)->(z,y,x)

            gs, _, res = fit_with_validation(
                roi, num_gaussians=args.k, iterations=args.iters,
                init_strategy=init_strategy, parameterization='full',
                init_scale=2.0, seed=args.seed, cell_metrics=True,
                cell_kwargs={'mode': '3d'})
            recon = gs.query_volume(roi.shape).cpu().numpy().astype(np.float32)
            pred = detect_cells(recon, mode='3d')

            # match predicted -> target, radius as in cellmetrics
            radius = 0.765 * 13
            matched = np.zeros(len(tgt), dtype=bool)
            if len(pred) and len(tgt):
                d = np.linalg.norm(pred[:, None, :] - tgt[None, :, :], axis=-1)
                ri_, ci_ = linear_sum_assignment(d)
                for a, b in zip(ri_, ci_):
                    if d[a, b] <= radius:
                        matched[b] = True

            props = nucleus_properties(roi, tgt)
            if len(p0):
                d0 = np.linalg.norm(tgt[:, None, :] - p0[None, :, :], axis=-1).min(axis=1)
            else:
                d0 = np.full(len(tgt), np.nan)

            for i, pr in enumerate(props):
                rows.append({**pr, 'found': bool(matched[i]), 'region': ri,
                             'init': init_strategy, 'dist_to_init': float(d0[i])})
            print(f'  region {ri}: {len(tgt)} nuclei, seeded={(d0<3).mean():.2f}, '
                  f'recall={matched.mean():.2f}, F1={res["cell_f1"]:.3f}, '
                  f'PSNR={res["full_psnr"]:.2f}')

    import pandas as pd
    df_all = pd.DataFrame(rows)
    df_all.to_csv(OUT / 'nuclei_found_vs_missed.csv', index=False)

    if len(args.inits) > 1:
        print('\n=== INIT COMPARISON: does coverage convert into recall? ===')
        print(f'{"init":18s} {"seeded":>8s} {"recall":>8s} {"stage1 miss":>12s} {"stage3 miss":>12s}')
        for init_strategy, sub in df_all.groupby('init'):
            seeded = (sub.dist_to_init < 3)
            nm = (~sub.found).sum()
            s1 = ((~sub.found) & ~seeded).sum()
            s3 = ((~sub.found) & seeded).sum()
            print(f'{init_strategy:18s} {seeded.mean():8.2f} {sub.found.mean():8.2f} '
                  f'{s1:5d} ({s1/max(nm,1)*100:3.0f}%) {s3:5d} ({s3/max(nm,1)*100:3.0f}%)')

    # the per-property analysis below uses the LAST init strategy
    df = df_all[df_all.init == args.inits[-1]]

    found = df[df.found]; missed = df[~df.found]
    print(f'\nPooled: {len(df)} nuclei -> {len(found)} found, {len(missed)} missed '
          f'(recall {len(found)/len(df):.2f})\n')

    fields = [('contrast', 'local contrast (peak - bg)', 'MSE weighting'),
              ('peak', 'peak intensity', 'MSE weighting'),
              ('size', 'nucleus size (voxels)', 'below representable scale'),
              ('nn_dist', 'distance to nearest neighbour', 'crowding / assignment'),
              ('depth_z', 'depth in z', 'light-sheet attenuation'),
              ('dist_to_init', 'distance to nearest initial Gaussian', 'placement')]

    from scipy import stats
    print(f'{"property":36s} {"found":>9s} {"missed":>9s} {"diff":>8s} {"p":>8s}  hypothesis')
    print('-' * 96)
    summary = []
    for key, label, hyp in fields:
        a = found[key].dropna().values; b = missed[key].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        t, pv = stats.mannwhitneyu(a, b, alternative='two-sided')
        rel = (np.median(b) - np.median(a)) / (abs(np.median(a)) + 1e-9) * 100
        print(f'{label:36s} {np.median(a):9.2f} {np.median(b):9.2f} '
              f'{rel:+7.1f}% {pv:8.4f}  {hyp}')
        summary.append(dict(property=key, label=label, hypothesis=hyp,
                            found_median=float(np.median(a)),
                            missed_median=float(np.median(b)),
                            pct_diff=float(rel), p_value=float(pv)))

    json.dump(summary, open(OUT / 'why_missed_summary.json', 'w'), indent=2)

    # figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (key, label, hyp) in zip(axes.ravel(), fields):
        a = found[key].dropna().values; b = missed[key].dropna().values
        if len(a) < 3 or len(b) < 3:
            ax.axis('off'); continue
        parts = ax.boxplot([a, b], labels=['found', 'missed'], patch_artist=True,
                           widths=0.55, showfliers=False)
        for pc, col in zip(parts['boxes'], ['#2ca02c', '#d62728']):
            pc.set_facecolor(col); pc.set_alpha(0.55)
        for j, v in enumerate([a, b]):
            ax.scatter(np.random.normal(j + 1, 0.055, len(v)), v, s=9,
                       color='k', alpha=0.35, zorder=3)
        t, pv = stats.mannwhitneyu(a, b, alternative='two-sided')
        ax.set_title(f'{label}\np = {pv:.4f}   ({hyp})', fontsize=10,
                     fontweight='bold' if pv < 0.05 else 'normal')
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle(f'Why are nuclei missed?  {len(found)} found vs {len(missed)} missed, '
                 f'{len(regions)} regions, k={args.k}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'why_missed.png', dpi=150, bbox_inches='tight')
    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

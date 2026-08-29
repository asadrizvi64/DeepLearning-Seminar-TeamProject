"""How strong is the detect-then-link baseline really?

Inter-frame identity ambiguity is only 0.31%, so LINKING is easy. But a detect-then-link
tracker must first DETECT the nucleus in every frame, and the manual-label validation
showed per-frame detection recall is far from perfect. A tracker that requires a
detection in each frame therefore loses a track the first time detection fails.

This measures, per frame, against the human annotation:
    R_t          detection recall in frame t
    P(hit|hit)   probability a nucleus detected at t is detected again at t+1
    run length   how many consecutive frames a nucleus stays detected
    survival     fraction of tracks a gap-intolerant linker keeps over the sequence

That quantifies the bar Gaussian persistence has to clear. If detection is reliable
every frame, persistence buys little and the baseline is strong. If detection drops out
frequently, a representation that carries identity THROUGH a missed detection has a
concrete advantage that linking cannot recover.

Detection uses the best ranking found earlier (LoG blob-likeness, physically isotropic
suppression) rather than raw intensity.

Usage:
    python scripts/baseline_fragility.py --frames 20 --budget 2000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import gaussian_filter, maximum_filter, gaussian_laplace

from volsplat.ctc import percentile_normalize

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/baseline_fragility'
DRO = Path(r'D:/Download/train/Fluo-N3DL-DRO (1)/Fluo-N3DL-DRO')
V = np.array([2.03, 0.406, 0.406])          # voxel size (z, y, x) in microns
MATCH_UM = 3.0
NMS_UM = 2.0
LOG_UM = 1.6


def detect(roi, budget, ranking='intensity'):
    """Peaks with a physically isotropic footprint, top-`budget` by `ranking`.

    Ranking matters and is budget-dependent (see manual_label_validation): LoG
    blob-likeness wins at SMALL budgets (R@500 0.196 vs 0.079) but LOSES at large ones
    (R@5000 0.540 vs 0.868). Detection peaks are always found on the smoothed intensity;
    only the ordering changes.
    """
    sm = gaussian_filter(roi, sigma=tuple(0.8 / V))
    size = [max(1, int(2 * round(NMS_UM / s)) + 1) for s in V]
    is_peak = (maximum_filter(sm, size=size) == sm) & (sm > 0)
    coords = np.argwhere(is_peak)
    if len(coords) == 0:
        return coords.astype(np.float32)

    if ranking == 'intensity':
        vals = sm[tuple(coords.T)]
    elif ranking == 'log':
        resp = -gaussian_laplace(roi, sigma=tuple(LOG_UM / V)) * (LOG_UM ** 2)
        vals = resp[tuple(coords.T)]
    elif ranking == 'contrast':
        vals = (sm - gaussian_filter(roi, sigma=tuple(6.0 / V)))[tuple(coords.T)]
    else:
        raise ValueError(f'unknown ranking {ranking!r}')
    order = np.argsort(-vals)[:budget]
    return coords[order].astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=int, default=20)
    p.add_argument('--budget', type=int, default=2000)
    p.add_argument('--sequence', default='01')
    p.add_argument('--ranking', default='intensity', choices=['intensity','log','contrast'])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tra_dir = DRO / f'{args.sequence}_GT/TRA'
    img_dir = DRO / args.sequence
    n = args.frames

    # fixed crop around the annotated territory (from frame 0)
    lab0 = tifffile.imread(str(tra_dir / 'man_track000.tif'))
    ids0 = np.unique(lab0); ids0 = ids0[ids0 > 0]
    c0 = np.array(ndimage.center_of_mass(lab0 > 0, lab0, ids0))
    lo = np.maximum(c0.min(0).astype(int) - 30, 0)
    hi = np.minimum(c0.max(0).astype(int) + 31, np.array(lab0.shape))
    print(f'crop z{lo[0]}:{hi[0]} y{lo[1]}:{hi[1]} x{lo[2]}:{hi[2]}   '
          f'budget={args.budget} detections/frame\n')

    hits = {}            # track id -> list of bool over frames
    recalls = []
    for t in range(n):
        lab = tifffile.imread(str(tra_dir / f'man_track{t:03d}.tif'))
        ids = np.unique(lab); ids = ids[ids > 0]
        cents = np.array(ndimage.center_of_mass(lab > 0, lab, ids))

        vol = percentile_normalize(tifffile.imread(str(img_dir / f't{t:03d}.tif')).astype(np.float32))
        roi = np.ascontiguousarray(vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        det = detect(roi, args.budget, ranking=args.ranking)

        man = cents - lo
        keep = np.all((man >= 0) & (man < np.array(roi.shape)), axis=1)
        man, ids = man[keep], ids[keep]
        if len(det) and len(man):
            d = np.linalg.norm((man * V)[:, None, :] - (det * V)[None, :, :], axis=-1)
            hit = d.min(axis=1) <= MATCH_UM
        else:
            hit = np.zeros(len(man), dtype=bool)
        recalls.append(float(hit.mean()))
        for i, tid in enumerate(ids):
            hits.setdefault(int(tid), {})[t] = bool(hit[i])
        if t % 5 == 0:
            print(f'  frame {t:3d}: {len(man)} annotated in crop, recall {hit.mean():.3f}')

    recalls = np.array(recalls)
    print(f'\n=== per-frame detection recall ===')
    print(f'  mean {recalls.mean():.3f}   min {recalls.min():.3f}   max {recalls.max():.3f}')

    # ---- persistence of detection
    cont, drop, runs, gap_hist = 0, 0, [], {}
    for tid, fr in hits.items():
        seq = [fr[t] for t in sorted(fr)]
        run = 0
        for i in range(len(seq) - 1):
            if seq[i]:
                if seq[i + 1]:
                    cont += 1
                else:
                    drop += 1
        for v in seq:
            if v:
                run += 1
            else:
                if run:
                    runs.append(run)
                run = 0
        if run:
            runs.append(run)
        # gaps: runs of False between Trues
        g, inside = 0, False
        for v in seq:
            if v:
                if inside and g:
                    gap_hist[g] = gap_hist.get(g, 0) + 1
                g, inside = 0, True
            elif inside:
                g += 1
    p_hit_given_hit = cont / max(cont + drop, 1)
    runs = np.array(runs) if runs else np.array([0])

    print(f'\n=== detection persistence ===')
    print(f'  P(detected at t+1 | detected at t) = {p_hit_given_hit:.3f}')
    print(f'  consecutive-detection run length: median {np.median(runs):.0f}, '
          f'mean {runs.mean():.1f}, max {runs.max()}')
    print(f'  gap lengths (frames missed between detections): ' +
          ', '.join(f'{k}:{v}' for k, v in sorted(gap_hist.items())[:6]))

    # save the hit matrix so survival can be recomputed without re-detecting
    tids = sorted(hits)
    hit_mat = np.array([[hits[t].get(f, False) for f in range(n)] for t in tids])
    np.save(OUT / f'hit_matrix_{args.ranking}_{args.budget}.npy', hit_mat)

    def survives(seq, horizon, tol):
        """Track survives if no run of consecutive misses exceeds `tol`, and it is
        detected at least once before the horizon."""
        s = seq[:horizon]
        if not s.any():
            return False
        run = 0
        for v in s:
            if v:
                run = 0
            else:
                run += 1
                if run > tol:
                    return False
        return True

    print(f'\n=== what this costs a linker, by GAP TOLERANCE ===')
    print(f'  tolerance = consecutive missed frames the linker can bridge')
    print(f'  {"horizon":>8s} ' + ' '.join(f'{"tol="+str(t):>9s}' for t in [0, 1, 2, 3, 5]))
    surv_rows = []
    for horizon in [5, 10, 20, n]:
        if horizon > n:
            continue
        vals = []
        for tol in [0, 1, 2, 3, 5]:
            s = float(np.mean([survives(hit_mat[i], horizon, tol)
                               for i in range(len(hit_mat))]))
            vals.append(s)
            surv_rows.append(dict(horizon=horizon, tol=tol, survival=s))
        print(f'  {horizon:8d} ' + ' '.join(f'{v*100:8.1f}%' for v in vals))

    summary = dict(frames=n, budget=args.budget, ranking=args.ranking,
                   recall_mean=float(recalls.mean()),
                   recall_min=float(recalls.min()),
                   p_hit_given_hit=float(p_hit_given_hit),
                   run_median=float(np.median(runs)), run_mean=float(runs.mean()),
                   gap_hist={str(k): v for k, v in gap_hist.items()},
                   survival=surv_rows)
    json.dump(summary, open(OUT / f'baseline_fragility_{args.ranking}_{args.budget}.json', 'w'), indent=2)
    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

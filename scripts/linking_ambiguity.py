"""Can the budget-5000 detections actually be LINKED, or was 0.31% measured on a
problem nobody has to solve?

The 0.31% identity-ambiguity figure was computed between MANUAL ANNOTATION CENTROIDS --
i.e. assuming the tracker already knows exactly which points are nuclei. A real
detect-then-link tracker links one detection at t to one of ~5000 detections at t+1,
most of which are unannotated structure. That is a different and harder problem, and it
is the last load-bearing assumption behind "persistence has no demonstrated advantage".

Protocol, per consecutive frame pair:
  1. detect at budget B in both frames (intensity ranking, physically isotropic NMS)
  2. for each annotated track, find the detection matched to its GT position at t (d_t)
     and at t+1 (d_next)
  3. link d_t forward by nearest neighbour among ALL detections at t+1
  4. correct link  <=> that nearest detection IS d_next

Reported:
    correct-link rate     links that land on the right nucleus
    ID-switch rate        links that land on a DIFFERENT annotated nucleus
    false-link rate       links that land on unannotated structure
    missed-detection rate track had no detection at t or t+1 (a detection failure,
                          not a linking failure -- counted separately)
and the same with 1-3 frame gap bridging.

Usage:
    python scripts/linking_ambiguity.py --frames 10 --budget 5000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import gaussian_filter, maximum_filter

from volsplat.ctc import percentile_normalize

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/linking_ambiguity'
DRO = Path(r'D:/Download/train/Fluo-N3DL-DRO (1)/Fluo-N3DL-DRO')
V = np.array([2.03, 0.406, 0.406])
MATCH_UM = 3.0
NMS_UM = 2.0


def detect(roi, budget):
    sm = gaussian_filter(roi, sigma=tuple(0.8 / V))
    size = [max(1, int(2 * round(NMS_UM / s)) + 1) for s in V]
    is_peak = (maximum_filter(sm, size=size) == sm) & (sm > 0)
    coords = np.argwhere(is_peak)
    if len(coords) == 0:
        return coords.astype(np.float32)
    vals = sm[tuple(coords.T)]
    return coords[np.argsort(-vals)[:budget]].astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=int, default=10)
    p.add_argument('--budget', type=int, default=5000)
    p.add_argument('--sequence', default='01')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tra = DRO / f'{args.sequence}_GT/TRA'
    img = DRO / args.sequence
    n = args.frames

    lab0 = tifffile.imread(str(tra / 'man_track000.tif'))
    i0 = np.unique(lab0); i0 = i0[i0 > 0]
    c0 = np.array(ndimage.center_of_mass(lab0 > 0, lab0, i0))
    lo = np.maximum(c0.min(0).astype(int) - 30, 0)
    hi = np.minimum(c0.max(0).astype(int) + 31, np.array(lab0.shape))
    print(f'crop {tuple(hi-lo)}   budget={args.budget}/frame   '
          f'match={MATCH_UM} um\n')

    # per frame: detections, and the GT position of every annotated track
    dets, gts = {}, {}
    for t in range(n):
        lab = tifffile.imread(str(tra / f'man_track{t:03d}.tif'))
        ids = np.unique(lab); ids = ids[ids > 0]
        cen = np.array(ndimage.center_of_mass(lab > 0, lab, ids)) - lo
        keep = np.all((cen >= 0) & (cen < (hi - lo)), axis=1)
        gts[t] = {int(i): c for i, c in zip(ids[keep], cen[keep])}

        vol = percentile_normalize(tifffile.imread(str(img / f't{t:03d}.tif')).astype(np.float32))
        roi = np.ascontiguousarray(vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        dets[t] = detect(roi, args.budget)
        if t % 5 == 0:
            print(f'  frame {t:3d}: {len(dets[t])} detections, {len(gts[t])} annotated')

    def matched_detection(t, pos):
        """Index of the detection matched to GT position `pos` in frame t, or None."""
        d = dets[t]
        if len(d) == 0:
            return None
        dist = np.linalg.norm((d - pos) * V, axis=1)
        j = int(dist.argmin())
        return j if dist[j] <= MATCH_UM else None

    print(f'\n=== linking on REAL detections (not GT centroids) ===')
    print(f'{"gap":>4s} {"attempts":>9s} {"correct":>9s} {"ID-switch":>10s} '
          f'{"false-link":>11s} {"no-detection":>13s}')
    rows = []
    for gap in [1, 2, 3]:
        correct = switch = false = missed = 0
        for t in range(n - gap):
            tn = t + gap
            # which detection index corresponds to which annotated track, at tn
            det_of_track_next = {}
            for tid, pos in gts[tn].items():
                j = matched_detection(tn, pos)
                if j is not None:
                    det_of_track_next[j] = tid

            for tid, pos in gts[t].items():
                if tid not in gts[tn]:
                    continue
                i = matched_detection(t, pos)
                j_true = matched_detection(tn, gts[tn][tid])
                if i is None or j_true is None:
                    missed += 1
                    continue
                # link forward: nearest detection at tn to detection i at t
                dist = np.linalg.norm((dets[tn] - dets[t][i]) * V, axis=1)
                j_link = int(dist.argmin())
                if j_link == j_true:
                    correct += 1
                elif j_link in det_of_track_next:
                    switch += 1          # landed on a different ANNOTATED nucleus
                else:
                    false += 1           # landed on unannotated structure
        tot = correct + switch + false
        rows.append(dict(gap=gap, attempts=tot, correct=correct, switch=switch,
                         false=false, missed=missed,
                         correct_rate=correct / max(tot, 1),
                         switch_rate=switch / max(tot, 1),
                         false_rate=false / max(tot, 1)))
        print(f'{gap:4d} {tot:9d} {correct/max(tot,1)*100:8.1f}% '
              f'{switch/max(tot,1)*100:9.2f}% {false/max(tot,1)*100:10.2f}% '
              f'{missed:13d}')

    print(f'\n  For reference: identity ambiguity measured between GT CENTROIDS was 0.31%.')
    r1 = rows[0]
    print(f'  On real detections at gap=1 the link is wrong '
          f'{(1-r1["correct_rate"])*100:.1f}% of the time '
          f'({r1["switch_rate"]*100:.2f}% onto another nucleus, '
          f'{r1["false_rate"]*100:.2f}% onto unannotated structure).')

    json.dump(dict(budget=args.budget, frames=n, match_um=MATCH_UM, rows=rows),
              open(OUT / f'linking_{args.budget}.json', 'w'), indent=2)
    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

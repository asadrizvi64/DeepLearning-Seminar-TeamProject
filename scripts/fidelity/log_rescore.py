"""Rescore the LoG-ranked detector on saved reconstructions at several match radii.

rate_detectability.py scored the LoG detector at one radius (0.4 x nucleus diameter) and
did not keep its detections. Cellpose's centres turned out to need a looser radius
(0.6 D) to reach normal precision on the raw volume, so both detectors must be reported
at the same radii. This re-runs the LoG-ranked detector on every recon/*.npz (identical
code path: rate_detectability.ranked_candidates), CACHES the ranked candidates, and
scores at 0.4 / 0.6 / 0.8 x D with the dense budget N = number of true nuclei.

    python scripts/fidelity/log_rescore.py runs/fidelity/codecs_ce_t150 \
        runs/fidelity/luxar_render_ce_t150 --out runs/fidelity/log_ce_t150.csv
"""
import argparse
import os
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rd', HERE / 'rate_detectability.py')
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)

RADIUS_FACTORS = (0.4, 0.6, 0.8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    cache = out.parent / (out.stem + '_candidates')
    cache.mkdir(parents=True, exist_ok=True)

    meta0 = json.load(open(Path(args.dirs[0]) / 'meta.json'))
    ds = rd.DATASETS[meta0['dataset']]
    rd.configure(ds['voxel'], ds['nucleus_um'])
    shape = None if meta0['origin'] == [0, 0, 0] else meta0['shape']
    seq = meta0.get('seq', '01')
    crop, lo, hi = rd.load_crop(ds['root'], meta0['frame'], meta0['origin'], shape, seq)
    gt = rd.gt_from_tra(ds['root'], meta0['frame'], meta0['origin'], shape, seq)
    raw_norm = rd.normalise(crop, lo, hi)
    _, raw_order, L = rd.ranked_candidates(raw_norm)
    g = L[tuple(np.clip(np.round(gt).astype(int), 0, np.array(crop.shape) - 1).T)]
    faint = g <= np.median(g)
    n_budget = len(gt) if ds['dense'] else 200
    group = f"{meta0['dataset']} s{seq} t{meta0['frame']:03d}"
    print(f'{group}: {len(gt)} labelled, budget {n_budget}', flush=True)

    jobs = [('raw', int(crop.nbytes), 'raw', raw_order)]
    for d in args.dirs:
        for f in sorted((Path(d) / 'recon').glob('*.npz')):
            method, nbytes = f.stem.rsplit('_', 1)
            if method != 'raw':
                jobs.append((method, int(nbytes), f.name, f))
    rows = []
    for method, nbytes, fname, src in jobs:
        cfile = cache / f'{fname}.log_ranked.npy'
        if isinstance(src, np.ndarray):
            order = src
        elif cfile.exists():
            order = np.load(cfile)
        else:
            _, order, _ = rd.ranked_candidates(np.load(src)['rec'].astype(np.float32))
            # atomic: an interrupted run must never leave a half-written cache (2026-10-03 left
            # one all-zero centroid file that scored a Luxar volume as 0% kept)
            tmp = cfile.with_name(cfile.name + '.tmp.npy')
            np.save(tmp, order)
            os.replace(tmp, cfile)
        row = dict(group=group, file=fname, method=method, bytes=nbytes, ratio=crop.nbytes / nbytes,
                   n_cand=int(len(order)))
        top = order[:n_budget]
        for fac in RADIUS_FACTORS:
            hit = np.zeros(len(gt), bool)
            if len(top):
                _, ci, _ = rd.match_indices(top * rd.V, gt * rd.V, fac * ds['nucleus_um'])
                hit[ci] = True
            row[f'found@r{fac}'] = int(hit.sum())
            row[f'found_faint@r{fac}'] = int(hit[faint].sum())
            row[f'found_bright@r{fac}'] = int(hit[~faint].sum())
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  {method:13s} {row['ratio']:6.1f}x  " +
              '  '.join(f"r{fac}: {row[f'found@r{fac}']:3d}" for fac in RADIUS_FACTORS), flush=True)


if __name__ == '__main__':
    main()

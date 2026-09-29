"""Second, independent detector: Cellpose 3 'nuclei' on every reconstruction.

The main analysis uses a LoG-ranked peak detector -- which is itself matched to
Gaussian blobs, so it could flatter a Gaussian-splat reconstruction. To test that, every
saved reconstruction (recon/*.npz, from codec runs and from render_luxar.py) is segmented
with Cellpose's pretrained nuclei model -- 2D per slice, stitched across z
(stitch_threshold 0.25, Cellpose's standard 3D-from-2D mode), diameter = the dataset's
measured nucleus diameter. Nucleus centres are matched to the TRA labels with the same
max-cardinality matcher and faint/bright split as the main analysis.

MATCH RADIUS SENSITIVITY. Segmentation centroids can sit further from the TRA markers
than LoG peaks do, so every file is scored at 0.4, 0.6 and 0.8 x the nucleus diameter
(0.4 = the main analysis). Detected centres are CACHED per file, so re-matching at any
other radius is instant -- never re-run Cellpose to rescore.

Cellpose is a segmenter, not a ranker, so it is scored at its own operating point:
recall, precision (meaningful on densely labelled CE) and F1, and nuclei kept relative
to Cellpose on the raw volume. Resumable.

Needs the Cellpose environment (cellpose<4):
    C:/Users/HP/cpenv/Scripts/python.exe scripts/fidelity/cellpose_score.py \
        runs/fidelity/codecs_ce_t150 runs/fidelity/luxar_render_ce_t150 --out runs/fidelity/cellpose_ce_t150.csv
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rd', HERE / 'rate_detectability.py')
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)

RADIUS_FACTORS = (0.4, 0.6, 0.8)          # x nucleus diameter; 0.4 = main analysis


def centroids(masks):
    ids = np.unique(masks)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return np.zeros((0, 3))
    return np.array(center_of_mass(masks > 0, masks, ids))


def detect(model, vol, diameter_px):
    masks, *_ = model.eval(vol.astype(np.float32), channels=[0, 0], diameter=diameter_px,
                           do_3D=False, stitch_threshold=0.25)
    return centroids(masks)


def match(pred, gt, faint, radius_um):
    hit = np.zeros(len(gt), bool)
    if len(pred):
        _, ci, _ = rd.match_indices(pred * rd.V, gt * rd.V, radius_um)
        hit[ci] = True
    tp = int(hit.sum())
    prec = tp / len(pred) if len(pred) else 0.0
    rec = tp / len(gt) if len(gt) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(found=tp, found_faint=int(hit[faint].sum()), found_bright=int(hit[~faint].sum()),
                precision=prec, recall=rec, f1=f1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out = Path(args.out)
    cache = out.parent / (out.stem + '_centroids')
    cache.mkdir(parents=True, exist_ok=True)
    meta0 = json.load(open(Path(args.dirs[0]) / 'meta.json'))
    ds = rd.DATASETS[meta0['dataset']]
    rd.configure(ds['voxel'], ds['nucleus_um'])
    diameter_px = ds['nucleus_um'] / rd.V[1]
    shape = None if meta0['origin'] == [0, 0, 0] else meta0['shape']
    seq = meta0.get('seq', '01')
    crop, lo, hi = rd.load_crop(ds['root'], meta0['frame'], meta0['origin'], shape, seq)
    group = f"{meta0['dataset']} s{seq} t{meta0['frame']:03d}"
    gt = rd.gt_from_tra(ds['root'], meta0['frame'], meta0['origin'], shape, seq)
    raw_norm = rd.normalise(crop, lo, hi)
    _, _, L = rd.ranked_candidates(raw_norm)
    g = L[tuple(np.clip(np.round(gt).astype(int), 0, np.array(crop.shape) - 1).T)]
    faint = g <= np.median(g)
    print(f'{group}: {len(gt)} labelled nuclei, Cellpose diameter {diameter_px:.0f} px, '
          f'radii {[round(f * ds["nucleus_um"], 2) for f in RADIUS_FACTORS]} um', flush=True)

    jobs = [('raw', int(crop.nbytes), 'raw', raw_norm)]
    for d in args.dirs:
        for f in sorted((Path(d) / 'recon').glob('*.npz')):
            method, nbytes = f.stem.rsplit('_', 1)
            if method != 'raw':
                jobs.append((method, int(nbytes), f.name, f))

    model = None
    rows = []
    for method, nbytes, fname, src in jobs:
        cfile = cache / f'{fname}.centroids.npy'
        if cfile.exists():
            pred = np.load(cfile)
        else:
            if model is None:
                from cellpose import models
                model = models.Cellpose(gpu=False, model_type='nuclei')
            vol = src if isinstance(src, np.ndarray) else np.load(src)['rec'].astype(np.float32)
            pred = detect(model, vol, diameter_px)
            np.save(cfile, pred)
        row = dict(group=group, file=fname, method=method, bytes=nbytes,
                   ratio=crop.nbytes / nbytes, n_pred=int(len(pred)))
        for fac in RADIUS_FACTORS:
            m = match(pred, gt, faint, fac * ds['nucleus_um'])
            row.update({f'{k}@r{fac}': v for k, v in m.items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  {method:13s} {nbytes / 1024:8.1f} KiB {row['ratio']:6.1f}x  n_pred {len(pred):3d}  "
              + '  '.join(f"r{fac}: found {row[f'found@r{fac}']:3d} F1 {row[f'f1@r{fac}']:.3f}"
                          for fac in RADIUS_FACTORS), flush=True)


if __name__ == '__main__':
    main()

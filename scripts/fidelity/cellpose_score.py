"""Second, independent detector: Cellpose 3 'nuclei' on every reconstruction.

The main analysis uses a LoG-ranked peak detector. To show the result is not an
artefact of that choice, every saved reconstruction (recon/*.npz, from codec runs and
from render_luxar.py) is segmented with Cellpose's pretrained nuclei model -- 2D per
slice, stitched across z (stitch_threshold 0.25, Cellpose's standard 3D-from-2D mode),
diameter = the dataset's measured nucleus diameter. Nucleus centres are matched to the
TRA labels with the SAME matcher, radius and faint/bright split as the main analysis.

Cellpose is a segmenter, not a ranker, so it is scored at its own operating point:
recall, precision (meaningful on the densely labelled CE data) and F1, plus nuclei kept
relative to Cellpose on the raw volume. Resumable: finished files are skipped.

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


def centroids(masks):
    ids = np.unique(masks)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return np.zeros((0, 3))
    return np.array(center_of_mass(masks > 0, masks, ids))


def score(model, vol, diameter_px, gt, faint):
    masks, *_ = model.eval(vol.astype(np.float32), channels=[0, 0], diameter=diameter_px,
                           do_3D=False, stitch_threshold=0.25)
    pred = centroids(masks)
    hit = np.zeros(len(gt), bool)
    if len(pred):
        _, ci, _ = rd.match_indices(pred * rd.V, gt * rd.V, rd.MATCH_UM)
        hit[ci] = True
    tp = int(hit.sum())
    prec = tp / len(pred) if len(pred) else 0.0
    rec = tp / len(gt) if len(gt) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n_pred=int(len(pred)), found=tp, found_faint=int(hit[faint].sum()),
                found_bright=int(hit[~faint].sum()), precision=prec, recall=rec, f1=f1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    from cellpose import models
    model = models.Cellpose(gpu=False, model_type='nuclei')

    out = Path(args.out)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    rows = done.to_dict('records')
    seen = {(r['group'], r['file']) for r in rows}
    meta0 = json.load(open(Path(args.dirs[0]) / 'meta.json'))
    ds = rd.DATASETS[meta0['dataset']]
    rd.configure(ds['voxel'], ds['nucleus_um'])
    diameter_px = ds['nucleus_um'] / rd.V[1]
    shape = None if meta0['origin'] == [0, 0, 0] else meta0['shape']
    crop, lo, hi = rd.load_crop(ds['root'], meta0['frame'], meta0['origin'], shape,
                                meta0.get('seq', '01'))
    group = f"{meta0['dataset']} s{meta0.get('seq', '01')} t{meta0['frame']:03d}"
    gt = rd.gt_from_tra(ds['root'], meta0['frame'], meta0['origin'], shape, meta0.get('seq', '01'))
    raw_norm = rd.normalise(crop, lo, hi)
    _, _, L = rd.ranked_candidates(raw_norm)
    g = L[tuple(np.clip(np.round(gt).astype(int), 0, np.array(crop.shape) - 1).T)]
    faint = g <= np.median(g)
    print(f'{group}: {len(gt)} labelled nuclei, Cellpose diameter {diameter_px:.0f} px, '
          f'match {rd.MATCH_UM:.2f} um', flush=True)

    jobs = [('raw', int(crop.nbytes), 'raw', raw_norm)]
    for d in args.dirs:
        for f in sorted((Path(d) / 'recon').glob('*.npz')):
            method, nbytes = f.stem.rsplit('_', 1)
            if method == 'raw':
                continue
            jobs.append((method, int(nbytes), f.name, f))
    for method, nbytes, fname, src in jobs:
        if (group, fname) in seen:
            continue
        vol = src if isinstance(src, np.ndarray) else np.load(src)['rec'].astype(np.float32)
        r = score(model, vol, diameter_px, gt, faint)
        r.update(group=group, file=fname, method=method, bytes=nbytes, ratio=crop.nbytes / nbytes)
        rows.append(r)
        seen.add((group, fname))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  {method:13s} {nbytes / 1024:8.1f} KiB {r['ratio']:6.1f}x  found {r['found']:3d} "
              f"(faint {r['found_faint']:3d})  precision {r['precision']:.2f}  F1 {r['f1']:.3f}",
              flush=True)


if __name__ == '__main__':
    main()

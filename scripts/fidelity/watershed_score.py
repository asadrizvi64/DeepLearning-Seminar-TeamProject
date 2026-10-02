"""Third detector (stage A2): a classical, non-learned 3D segmentation -- background
subtraction, Otsu threshold, distance-transform watershed -- on every reconstruction.

Why this one. The LoG detector is itself a blob filter (it could flatter Gaussian splats),
and Cellpose is a learned 2D model stitched across z. This detector is volumetric,
untrained and shape-based: it finds nuclei as separable bright regions, not as blob-shaped
intensity peaks. It is the kind of pipeline (threshold + watershed) a microscopist would
run without any model.

Pipeline (all spatial parameters in microns, scaled by the nucleus diameter D like the
rest of the study):
  1. smooth with a Gaussian, sigma 0.15 D;
  2. subtract a broad background (Gaussian, sigma 1.0 D) -- removes the embryo's
     cytoplasm glow so the threshold separates nuclei, not embryo from dish;
  3. foreground = background-subtracted image > Otsu threshold of its positive part;
  4. Euclidean distance transform of the foreground (anisotropic voxel spacing);
  5. markers = distance maxima at least 0.1 D deep, one per 0.4 D box neighbourhood;
  6. watershed of -distance inside the foreground; regions < 0.1 nucleus volume dropped;
  7. detections = region centroids.
The threshold is recomputed on every volume, as it would be in practice.

Centres are cached per file (<out stem>_centroids/<file>.centroids.npy) and scored with
the same matcher, radii (0.4/0.6/0.8 D) and faint/bright split as cellpose_score.py.

    python scripts/fidelity/watershed_score.py runs/fidelity/codecs_ce_t150 \
        runs/fidelity/luxar_render_ce_t150 --out runs/fidelity/watershed_ce_t150.csv
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import (center_of_mass, distance_transform_edt, gaussian_filter, label,
                           maximum_filter)
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rd', HERE / 'rate_detectability.py')
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)
spec = importlib.util.spec_from_file_location('cps', HERE / 'cellpose_score.py')
cps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cps)
# cellpose_score loads its OWN copy of rate_detectability; its match() must use the copy
# configured below, or it matches in the default (DRO) voxel size. Bug of 2026-10-02:
# the first watershed CSVs were matched with 0.406 um xy voxels instead of 0.09 um.
cps.rd = rd

# Frozen 2026-10-02 from a 16-setting check on RAW volumes only (E1 t150, E2 t180; F1 vs
# the TRA labels at 0.6 D) -- never tuned on reconstructions. Background sigma 1.0 D beat
# 1.5 D in every setting; separation 0.3/0.4 D and depth 0.1/0.2 D were tied (F1 0.73/0.66).
SMOOTH_D, BG_D, MIN_DEPTH_D, SEP_D, MIN_VOL_FRAC = 0.15, 1.0, 0.1, 0.4, 0.1


def detect(vol, D):
    V = rd.V
    sm = gaussian_filter(vol.astype(np.float32), sigma=tuple(SMOOTH_D * D / V))
    hp = sm - gaussian_filter(sm, sigma=tuple(BG_D * D / V))
    pos = hp[hp > 0]
    if pos.size < 2 or pos.max() <= pos.min():
        return np.zeros((0, 3))
    fg = hp > threshold_otsu(pos)
    dist = distance_transform_edt(fg, sampling=V)
    size = tuple(int(2 * round(SEP_D * D / v) + 1) for v in V)
    peaks = (dist == maximum_filter(dist, size=size)) & (dist >= MIN_DEPTH_D * D)
    markers, n = label(peaks)
    if n == 0:
        return np.zeros((0, 3))
    lab = watershed(-dist, markers, mask=fg)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    min_vox = MIN_VOL_FRAC * (np.pi / 6) * D ** 3 / np.prod(V)
    ids = ids[counts >= min_vox]
    if len(ids) == 0:
        return np.zeros((0, 3))
    return np.array(center_of_mass(lab > 0, lab, ids))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--root', default=None)
    args = p.parse_args()

    out = Path(args.out)
    cache = out.parent / (out.stem + '_centroids')
    cache.mkdir(parents=True, exist_ok=True)
    meta0 = json.load(open(Path(args.dirs[0]) / 'meta.json'))
    ds = rd.DATASETS[meta0['dataset']]
    rd.configure(ds['voxel'], ds['nucleus_um'])
    D = ds['nucleus_um']
    root = args.root or ds['root']
    shape = None if meta0['origin'] == [0, 0, 0] else meta0['shape']
    seq = meta0.get('seq', '01')
    crop, lo, hi = rd.load_crop(root, meta0['frame'], meta0['origin'], shape, seq)
    group = f"{meta0['dataset']} s{seq} t{meta0['frame']:03d}"
    gt = rd.gt_from_tra(root, meta0['frame'], meta0['origin'], shape, seq)
    raw_norm = rd.normalise(crop, lo, hi)
    _, _, L = rd.ranked_candidates(raw_norm)
    g = L[tuple(np.clip(np.round(gt).astype(int), 0, np.array(crop.shape) - 1).T)]
    faint = g <= np.median(g)
    print(f'{group}: {len(gt)} labelled nuclei, watershed detector', flush=True)

    jobs = [('raw', int(crop.nbytes), 'raw', raw_norm)]
    for d in args.dirs:
        for f in sorted((Path(d) / 'recon').glob('*.npz')):
            method, nbytes = f.stem.rsplit('_', 1)
            if method != 'raw':
                jobs.append((method, int(nbytes), f.name, f))
    rows = []
    for method, nbytes, fname, src in jobs:
        cfile = cache / f'{fname}.centroids.npy'
        if cfile.exists():
            pred = np.load(cfile)
        else:
            vol = src if isinstance(src, np.ndarray) else np.load(src)['rec'].astype(np.float32)
            pred = detect(vol, D)
            np.save(cfile, pred)
        row = dict(group=group, file=fname, method=method, bytes=nbytes,
                   ratio=crop.nbytes / nbytes, n_pred=int(len(pred)))
        for fac in cps.RADIUS_FACTORS:
            m = cps.match(pred, gt, faint, fac * D)
            row.update({f'{k}@r{fac}': v for k, v in m.items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  {method:13s} {nbytes / 1024:8.1f} KiB {row['ratio']:6.1f}x  n_pred {len(pred):3d}  "
              + '  '.join(f"r{fac}: found {row[f'found@r{fac}']:3d} F1 {row[f'f1@r{fac}']:.3f}"
                          for fac in cps.RADIUS_FACTORS), flush=True)


if __name__ == '__main__':
    main()

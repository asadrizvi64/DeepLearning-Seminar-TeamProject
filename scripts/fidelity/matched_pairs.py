"""Luxar vs JPEG2000 at EXACTLY the same bytes, every fit, every detector, with 95% paired
bootstrap intervals over labelled nuclei.

After the JPEG2000 fix (2026-10-02) JPEG2000 is encoded at the byte size of every Luxar fit
(redo_jpeg2k.py), so each fit has a same-size codec twin. For each frame x detector x fit:
  kept_lux, kept_j2k  = nuclei found / nuclei found in the raw volume (same detector)
  diff = kept_lux - kept_j2k, CI from B = 2000 resamples of the labelled nuclei (paired:
  the same resampled nuclei for raw, Luxar and JPEG2000)
JPEG-XL is paired too where it reaches the size (within 10%).

Writes runs/fidelity/matched_pairs.csv. Uses only cached detections (no detector reruns),
except the LoG ranking of each raw volume (about a minute per frame).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
F = REPO / 'runs' / 'fidelity'
spec = importlib.util.spec_from_file_location('rd', REPO / 'scripts/fidelity/rate_detectability.py')
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)
spec = importlib.util.spec_from_file_location('rt', REPO / 'scripts/fidelity/results_table.py')
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
B = 2000
EXACT, NEAR = 0.02, 0.10                   # J2K must match within 2%, JPEG-XL within 10%


def hits(pred, gt, rad):
    h = np.zeros(len(gt), bool)
    if len(pred):
        _, ci, _ = rd.match_indices(pred * rd.V, gt * rd.V, rad)
        h[ci] = True
    return h


def frame_hits(tag, det, gt, raw_log, rad):
    """{file: hit vector} for every cached detection of this detector, plus 'raw'."""
    if det == 'log':
        cache, suffix = F / f'log_{tag}_candidates', '.log_ranked.npy'
        H = {'raw': hits(raw_log[:len(gt)], gt, rad)}
        load = lambda p: np.load(p)[:len(gt)]
    else:
        cache, suffix = F / f'{det}_{tag}_centroids', '.centroids.npy'
        if not (cache / 'raw.centroids.npy').exists():
            return None
        H = {'raw': hits(np.load(cache / 'raw.centroids.npy'), gt, rad)}
        load = np.load
    if not cache.exists():
        return None
    for p in cache.glob('*' + suffix):
        name = p.name[:-len(suffix)]
        if name != 'raw':
            H[name] = hits(load(p), gt, rad)
    return H


def main():
    ds = rd.DATASETS['CE']
    rd.configure(ds['voxel'], ds['nucleus_um'])
    rad = 0.6 * ds['nucleus_um']
    rng = np.random.default_rng(0)
    res = pd.read_csv(F / 'results_all.csv')
    rows = []
    for tag, (seq, t, role, _) in rt.FRAMES.items():
        sub = res[res.tag == tag]
        if sub.empty:
            continue
        gt = rd.gt_from_tra(ds['root'], t, [0, 0, 0], None, seq)
        crop, lo, hi = rd.load_crop(ds['root'], t, [0, 0, 0], None, seq)
        _, raw_log, _ = rd.ranked_candidates(rd.normalise(crop, lo, hi))
        idx = rng.integers(0, len(gt), (B, len(gt)))
        for det in sorted(sub.detector.unique()):
            H = frame_hits(tag, det, gt, raw_log, rad)
            if H is None:
                continue
            d = sub[sub.detector == det]
            r0 = H['raw'][idx].sum(1)
            lux = d[d.family == 'luxar']
            for _, L in lux.iterrows():
                for fam, tol in (('jpeg2k', EXACT), ('jpegxl', NEAR)):
                    c = d[d.family == fam]
                    if c.empty:
                        continue
                    C = c.loc[(c.bytes / L.bytes - 1).abs().idxmin()]
                    if abs(C.bytes / L.bytes - 1) > tol or L.file not in H or C.file not in H:
                        continue
                    hl, hc = H[L.file], H[C.file]
                    boot = (hl[idx].sum(1) - hc[idx].sum(1)) / r0
                    rows.append(dict(tag=tag, frame=L.frame, role=role, detector=det, codec=fam,
                                     method=L.method, splats=L.splats, splats_per_nucleus=L.splats_per_nucleus,
                                     bytes_lux=L.bytes, bytes_codec=C.bytes, ratio=L.ratio,
                                     psnr_lux=L.psnr, psnr_codec=C.psnr,
                                     kept_lux=hl.sum() / H['raw'].sum(), kept_codec=hc.sum() / H['raw'].sum(),
                                     diff=(hl.sum() - hc.sum()) / H['raw'].sum(),
                                     lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5)))
        print(f'{tag}: done', flush=True)
    df = pd.DataFrame(rows).sort_values(['detector', 'codec', 'frame', 'ratio'])
    df.to_csv(F / 'matched_pairs.csv', index=False, float_format='%.4g')
    with pd.option_context('display.width', 200):
        print(df[['frame', 'detector', 'codec', 'method', 'ratio', 'psnr_lux', 'psnr_codec',
                  'kept_lux', 'kept_codec', 'diff', 'lo', 'hi']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()

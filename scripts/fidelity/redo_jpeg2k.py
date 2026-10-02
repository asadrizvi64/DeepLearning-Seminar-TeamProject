"""Re-run JPEG2000 on existing frames after the axis bug fix (2026-10-02).

The old JPEG2000 wavelet never ran along x (see rate_detectability.jpeg2k). For each frame
tag this
  1. moves the old jpeg2k recon volumes to <codec dir>/recon_superseded_jpeg2k/ and the old
     cached detections for them (LoG candidates, Cellpose / watershed centroids) into a
     superseded_jpeg2k/ subfolder of each cache, so no stale detection can be reused;
  2. encodes the corrected JPEG2000 at 50-800 KiB AND at the byte size of every Luxar fit
     of that frame (exact size matching), saves recon/jpeg2k_<bytes>.npz;
  3. replaces the jpeg2k rows of <codec dir>/rate_detectability.csv (PSNR, LoG@0.4D, survival).
Then rerun log_rescore.py / cellpose_score.py / watershed_score.py on the frame: every
other file is cached, so only the new JPEG2000 volumes are processed.

    python scripts/fidelity/redo_jpeg2k.py ce_t150 ce_t194 ce_s02_t150 ce_s02_t180 ce_s01_t100
"""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rd', HERE / 'rate_detectability.py')
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)
F = HERE.parents[1] / 'runs' / 'fidelity'
KIB = (50, 100, 200, 400, 800)


def supersede(tag, cdir):
    old = sorted((cdir / 'recon').glob('jpeg2k_*.npz'))
    dst = cdir / 'recon_superseded_jpeg2k'
    dst.mkdir(exist_ok=True)
    for f in old:
        shutil.move(str(f), dst / f.name)
        for cache, suffix in ((F / f'log_{tag}_candidates', '.log_ranked.npy'),
                              (F / f'cellpose_{tag}_centroids', '.centroids.npy'),
                              (F / f'watershed_{tag}_centroids', '.centroids.npy'),
                              (F / f'cellpose3d_{tag}_centroids', '.centroids.npy')):
            c = cache / (f.name + suffix)
            if c.exists():
                (cache / 'superseded_jpeg2k').mkdir(exist_ok=True)
                shutil.move(str(c), cache / 'superseded_jpeg2k' / c.name)
    return len(old)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('tags', nargs='+')
    p.add_argument('--add-only', action='store_true',
                   help='frames encoded after the fix: only add JPEG2000 at new Luxar sizes')
    args = p.parse_args()
    for tag in args.tags:
        cdir = F / f'codecs_{tag}'
        meta = json.load(open(cdir / 'meta.json'))
        ds = rd.DATASETS[meta['dataset']]
        rd.configure(ds['voxel'], ds['nucleus_um'])
        shape = None if meta['origin'] == [0, 0, 0] else meta['shape']
        seq = meta.get('seq', '01')
        crop, lo, hi = rd.load_crop(ds['root'], meta['frame'], meta['origin'], shape, seq)
        manual = rd.gt_from_tra(ds['root'], meta['frame'], meta['origin'], shape, seq)
        budgets = [tuple(b) for b in meta['budgets']]
        raw_norm = rd.normalise(crop, lo, hi)
        _, raw_log_order, L_raw = rd.ranked_candidates(raw_norm)
        g = L_raw[tuple(np.clip(np.round(manual).astype(int), 0, np.array(crop.shape) - 1).T)]
        faint = g <= np.median(g)
        ref = raw_log_order[:rd.SURVIVAL_S]
        rs = L_raw[tuple(np.round(ref).astype(int).T)]
        ref_faint = rs <= np.median(rs)

        lux_bytes = sorted({int(f.stem.rsplit('_', 1)[1])
                            for d in F.glob(f'luxar_render_{tag}*') for f in (d / 'recon').glob('luxar_K*.npz')})
        targets = sorted({k * 1024 for k in KIB} | set(lux_bytes))
        csv = cdir / 'rate_detectability.csv'
        keep = pd.read_csv(csv)
        if args.add_only:
            # post-fix frames: keep the (correct) JPEG2000 already there, add only the sizes
            # that no existing JPEG2000 volume matches within 2%
            have = keep[keep.method == 'jpeg2k'].bytes.tolist()
            targets = [t for t in targets if not any(abs(h / t - 1) <= 0.02 for h in have)]
            print(f'{tag}: adding {len(targets)} JPEG2000 sizes ({len(lux_bytes)} Luxar fits)', flush=True)
        else:
            n_old = supersede(tag, cdir)
            keep = keep[keep.method != 'jpeg2k']
            print(f'{tag}: moved {n_old} old JPEG2000 volumes; {len(targets)} targets '
                  f'({len(lux_bytes)} at Luxar sizes)', flush=True)
        rows = []
        for tgt in targets:
            rec, nb, prm = rd.jpeg2k(crop, tgt)
            prm['target_bytes'] = int(tgt)
            prm['size_matched'] = bool(abs(nb - tgt) <= 0.10 * tgt)
            rn = rd.normalise(rec, lo, hi)
            np.savez_compressed(cdir / 'recon' / f'jpeg2k_{nb}.npz', rec=rn.astype(np.float16))
            r = rd.score(rn, raw_norm, manual, faint, ref, ref_faint, budgets)
            r.update(method='jpeg2k', bytes=int(nb), ratio=crop.nbytes / nb,
                     bits_per_voxel=8.0 * nb / crop.size, params=json.dumps(prm))
            rows.append(r)
            pd.concat([keep, pd.DataFrame(rows)], ignore_index=True).to_csv(csv, index=False)
            print(f"  jpeg2k {nb / 1024:7.1f} KiB {r['ratio']:6.1f}x  PSNR {r['psnr']:5.2f}  "
                  f"survival {r['survival']:.2f}", flush=True)


if __name__ == '__main__':
    main()

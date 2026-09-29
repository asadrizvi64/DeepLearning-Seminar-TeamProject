"""Re-render saved Luxar fits to reconstruction volumes (for a second detector).

The cluster results were copied back WITHOUT the large recon/ volumes, but with the
small .gsplats.zarr fits. This renders each fit exactly as rate_detectability.py does
(voxel-space fit, affine map to raw counts, same normalisation) and writes
recon/luxar_K<seeds>_<bytes>.npz into --out, the same layout the codec runs use.

Needs the Luxar environment (Python >= 3.12):
    luxenv/Scripts/python.exe scripts/fidelity/render_luxar.py \
        runs/fidelity_cluster/fidelity/pilot_ce_t150 --out runs/fidelity/luxar_render_ce_t150
"""
import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rd', HERE / 'rate_detectability.py')
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--out', required=True)
    p.add_argument('--root', default=None)
    args = p.parse_args()
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.rendering import render_to_volume

    run = Path(args.run_dir)
    meta = json.load(open(run / 'meta.json'))
    ds = rd.DATASETS[meta['dataset']]
    rd.configure(ds['voxel'], ds['nucleus_um'])
    root = args.root or ds['root']
    full = meta['shape'] if meta['origin'] == [0, 0, 0] else None
    shape = None if full else meta['shape']
    crop, lo, hi = rd.load_crop(root, meta['frame'], meta['origin'], shape, meta.get('seq', '01'))
    out = Path(args.out)
    (out / 'recon').mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(out / 'meta.json', 'w'), indent=2)
    raw32 = crop.astype(np.float32)
    for z in sorted((run / 'luxar').glob('K*.gsplats.zarr'), key=lambda q: int(q.name[1:].split('.')[0])):
        k = int(z.name[1:].split('.')[0])
        nbytes = sum(f.stat().st_size for f in z.rglob('*') if f.is_file())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = GSplatData.load(str(z))
            rec = render_to_volume(res, shape=crop.shape, device='cpu').astype(np.float32)
        r = float(np.corrcoef(rec.ravel(), raw32.ravel())[0, 1])
        a, b = np.polyfit(rec.ravel(), raw32.ravel(), 1)
        rn = rd.normalise(a * rec + b, lo, hi)
        np.savez_compressed(out / 'recon' / f'luxar_K{k}_{nbytes}.npz', rec=rn.astype(np.float16))
        print(f'K{k}: {res.centers.shape[0]} splats, {nbytes / 1024:.1f} KiB, render-vs-data r={r:.3f}',
              flush=True)


if __name__ == '__main__':
    main()

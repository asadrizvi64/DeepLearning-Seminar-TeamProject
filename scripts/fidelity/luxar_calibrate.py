"""Where does Luxar's OWN budget recommendation land on the nucleus cliff?

Luxar picks its splat budget K* by blind-spot cross-validation (Noise2Self): hide 5% of
voxels, fit at each K, and take the K that best predicts the hidden voxels (held-out
PSNR). That is a principled, label-free rule -- but it optimises intensity fidelity.
Our rate-detectability sweep measured, on the SAME frames and the SAME K grid, how many
labelled nuclei survive at each K. This job runs Luxar's calibration so K* can be placed
on that curve: if K* falls below the cliff, following Luxar's recommendation loses
nuclei, and only an object-level check would say so.

fit_kwargs match the sweep (voxel_size, 1000 iterations) and add output_space='voxel'
so any internal render stays on the voxel grid (see rate_detectability.luxar_fit).

    python scripts/fidelity/luxar_calibrate.py --dataset CE --seq 01 --frame 194 \
        --root $WS/Fluo-N3DH-CE --out $WS/runs/fidelity/calibrate_ce_s01_t194 --device cuda
"""
import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rd', HERE / 'rate_detectability.py')
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)

K_GRID = (1000, 2000, 4000, 8000, 16000, 32000, 64000)   # identical to the sweep


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='CE', choices=list(rd.DATASETS))
    p.add_argument('--root', default=None)
    p.add_argument('--seq', default='01')
    p.add_argument('--frame', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--k-grid', type=int, nargs='*', default=list(K_GRID))
    args = p.parse_args()
    from luxar.gsplats.calibration import calibrate

    ds = rd.DATASETS[args.dataset]
    rd.configure(ds['voxel'], ds['nucleus_um'])
    vol, _, _ = rd.load_crop(args.root or ds['root'], args.frame, [0, 0, 0], None, args.seq)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'{args.dataset} s{args.seq} t{args.frame:03d} {vol.shape}, K grid {args.k_grid}', flush=True)

    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = calibrate(
            vol.astype(np.float32), k_grid=list(args.k_grid),
            fit_kwargs=dict(device=args.device, verbose=False, n_iters=1000,
                            voxel_size=list(rd.V), output_space='voxel'),
            keep_fits=out / 'fits',
            progress_callback=lambda i, n, msg: print(f'  [{i}/{n}] {msg}', flush=True))
    res.to_json(out / 'calibration.json')
    peak = res.held_out_peak
    summary = dict(dataset=args.dataset, seq=args.seq, frame=args.frame,
                   k_grid=list(args.k_grid), k_star=int(peak.k_star),
                   curve_type=str(peak.type), k_knee=getattr(peak, 'k_knee', None),
                   held_out_psnr_db=[float(x) for x in res.held_out_psnr_db],
                   minutes=round((time.time() - t) / 60, 1))
    json.dump(summary, open(out / 'summary.json', 'w'), indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()

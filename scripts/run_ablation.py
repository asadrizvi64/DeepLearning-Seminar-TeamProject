"""Systematic ablation runner for volumetric 3D Gaussian Splatting.

Sweeps the full grid

    dataset x initialization x k (num_gaussians) x parameterization x seed

and scores every cell on the complete metric battery (PSNR, SSIM, MSE, MAE,
max-error, correlation, param-count, bytes, compression, fit-time, efficiency).

Results are appended to a single CSV (one row per run) and each run's PSNR-vs-iter
training curve is saved as JSON. The runner is RESUMABLE: rows already present in the
CSV (matched on the config key) are skipped, so you can stop and restart freely.

Examples
--------
Fast smoke grid (finishes in a few minutes on CPU):
    python scripts/run_ablation.py --preset smoke

Full grid (the real experiment):
    python scripts/run_ablation.py --preset full

Custom slice (fix k, sweep parameterization x init on real data):
    python scripts/run_ablation.py \
        --datasets real --inits random intensity_weighted local_maxima \
        --k 50 --params isotropic diagonal full --iters 1500
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from volsplat.train import train_static
from volsplat.metrics import compute_all

REPO = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ datasets
# name -> path to a normalized .npy volume in [0, 1].
DATASETS = {
    'phantom': REPO / 'runs/real_roi_64x128x128.npy',   # placeholder if no phantom npy
    'real':    REPO / 'runs/real_roi_64x128x128.npy',
}

# Prefer a genuine phantom volume if one exists on disk.
_PHANTOM_CANDIDATES = [
    REPO / 'runs/synthetic_64x128x128.npy',
    REPO / 'runs/phantom_64x128x128.npy',
    REPO / 'data/phantom/phantom.npy',
]
for _c in _PHANTOM_CANDIDATES:
    if _c.exists():
        DATASETS['phantom'] = _c
        break


# ------------------------------------------------------------------ presets
PRESETS = {
    'smoke': dict(
        datasets=['real'],
        inits=['intensity_weighted'],
        ks=[10, 50],
        params=['isotropic', 'diagonal', 'full'],
        seeds=[0],
        iters=300,
    ),
    'params': dict(  # the parameterization story at a few k
        datasets=['real', 'phantom'],
        inits=['intensity_weighted'],
        ks=[10, 50, 100, 250],
        params=['isotropic', 'diagonal', 'full'],
        seeds=[0],
        iters=1500,
    ),
    'full': dict(
        datasets=['real', 'phantom'],
        inits=['random', 'intensity_weighted', 'local_maxima'],
        ks=[1, 10, 50, 100, 250, 500],
        params=['isotropic', 'diagonal', 'full'],
        seeds=[0],
        iters=1500,
    ),
}

CONFIG_KEYS = ['dataset', 'init', 'num_gaussians', 'parameterization', 'seed']


def load_dataset(name: str) -> np.ndarray:
    path = DATASETS[name]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset {name!r} expected at {path}, which does not exist. "
            f"Extract it first (see scripts/extract_real_roi.py) or edit DATASETS."
        )
    return np.load(path).astype(np.float32)


def already_done(results_csv: Path, cfg: dict) -> bool:
    if not results_csv.exists():
        return False
    df = pd.read_csv(results_csv)
    if df.empty:
        return False
    mask = np.ones(len(df), dtype=bool)
    for key in CONFIG_KEYS:
        if key not in df.columns:
            return False
        mask &= (df[key] == cfg[key])
    return bool(mask.any())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', choices=sorted(PRESETS), default=None,
                   help='Use a predefined grid. Overrides the individual axis flags.')
    p.add_argument('--datasets', nargs='+', default=['real'])
    p.add_argument('--inits', nargs='+', default=['intensity_weighted'],
                   help='random / intensity_weighted / local_maxima')
    p.add_argument('--k', dest='ks', nargs='+', type=int, default=[10, 50, 100])
    p.add_argument('--params', nargs='+', default=['isotropic', 'diagonal', 'full'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--iters', type=int, default=1500)
    p.add_argument('--out-dir', default='runs/ablation')
    args = p.parse_args()

    if args.preset:
        cfg = PRESETS[args.preset]
        datasets, inits, ks = cfg['datasets'], cfg['inits'], cfg['ks']
        params, seeds, iters = cfg['params'], cfg['seeds'], cfg['iters']
    else:
        datasets, inits, ks = args.datasets, args.inits, args.ks
        params, seeds, iters = args.params, args.seeds, args.iters

    out_dir = Path(REPO / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = out_dir / 'curves'
    curves_dir.mkdir(exist_ok=True)
    results_csv = out_dir / 'results.csv'

    grid = [
        dict(dataset=d, init=i, num_gaussians=k, parameterization=pm, seed=s)
        for d in datasets for i in inits for k in ks for pm in params for s in seeds
    ]
    total = len(grid)
    print(f'Ablation grid: {total} runs')
    print(f'  datasets       : {datasets}')
    print(f'  inits          : {inits}')
    print(f'  k              : {ks}')
    print(f'  parameterizations: {params}')
    print(f'  seeds          : {seeds}   iters/run: {iters}')
    print(f'  -> {results_csv}\n')

    # cache loaded volumes
    vols = {}

    for n, cfg in enumerate(grid, 1):
        tag = (f"[{n}/{total}] {cfg['dataset']}/{cfg['init']}/"
               f"k={cfg['num_gaussians']}/{cfg['parameterization']}/seed{cfg['seed']}")
        if already_done(results_csv, cfg):
            print(f'{tag}  -- skip (done)')
            continue

        if cfg['dataset'] not in vols:
            vols[cfg['dataset']] = load_dataset(cfg['dataset'])
        volume = vols[cfg['dataset']]

        print(f'{tag}  -- fitting...')
        t0 = time.time()
        try:
            gs, history = train_static(
                volume=volume,
                num_gaussians=cfg['num_gaussians'],
                iterations=iters,
                seed=cfg['seed'],
                init_strategy=cfg['init'],
                parameterization=cfg['parameterization'],
            )
        except Exception as e:
            print(f'    FAILED: {type(e).__name__}: {e}')
            continue
        fit_seconds = time.time() - t0

        recon = gs.query_volume(volume.shape).cpu().numpy().astype(np.float32)
        row = compute_all(
            recon, volume,
            num_gaussians=cfg['num_gaussians'],
            parameterization=cfg['parameterization'],
            fit_seconds=fit_seconds,
        )
        row.update(cfg)
        row['iters'] = iters

        # persist training curve (iter -> psnr) for later curve plots
        curve = [(h['iter'], h['psnr']) for h in history if 'psnr' in h and 'iter' in h]
        run_id = (f"{cfg['dataset']}_{cfg['init']}_k{cfg['num_gaussians']}"
                  f"_{cfg['parameterization']}_s{cfg['seed']}")
        with open(curves_dir / f'{run_id}.json', 'w') as f:
            json.dump({'config': cfg, 'curve': curve}, f)

        # append the row (write header only if file is new)
        df_row = pd.DataFrame([row])
        df_row.to_csv(results_csv, mode='a', header=not results_csv.exists(), index=False)

        print(f"    PSNR={row['psnr']:.2f}  SSIM={row['ssim']:.3f}  "
              f"MAE={row['mae']:.4f}  params={row['num_params']}  "
              f"time={fit_seconds:.1f}s")

    print(f'\nDone. Results -> {results_csv}')
    print(f'Plot with: python scripts/plot_ablation.py --results {results_csv}')


if __name__ == '__main__':
    main()

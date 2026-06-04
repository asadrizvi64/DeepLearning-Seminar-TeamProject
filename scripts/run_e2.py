"""E2 - initialization-strategy ablation (P3).

Compares the three init strategies (random, intensity_weighted, local_maxima) on
the same phantom with matched training budget. Reports per-strategy:
  * PSNR at init (before any training - tests how good the init alone is)
  * Final density PSNR
  * Fit time
  * Iterations to reach a target PSNR threshold

Uses the voxel-query supervisor (P2 decision) for all comparisons.

Usage:
    python scripts/run_e2.py --shape 32 48 48 --num-blobs 15 --sigma 2.0 \
        --num-gaussians 500 --iterations 1500 --out-dir runs/e2_v1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from volsplat.data import generate_phantom
from volsplat.init import init_gaussians, INIT_STRATEGIES
from volsplat.train import train_static, evaluate_full
from volsplat.losses import psnr


STRATEGIES = ('random', 'intensity_weighted', 'local_maxima')


def init_psnr(gs, volume_t, subsample: int = 50_000) -> float:
    """Density-field PSNR before any training - measures pure init quality."""
    return evaluate_full(gs, volume_t, subsample=subsample)


def iters_to_target(history, target_psnr: float) -> int:
    """First iteration in `history` where logged PSNR >= target. -1 if never."""
    for h in history:
        if isinstance(h, dict) and 'psnr' in h and h['psnr'] >= target_psnr:
            return int(h['iter'])
    return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=str, default='runs/e2')
    p.add_argument('--shape', type=int, nargs=3, default=[32, 48, 48],
                   metavar=('D', 'H', 'W'))
    p.add_argument('--num-blobs', type=int, default=15)
    p.add_argument('--sigma', type=float, default=2.0)
    p.add_argument('--num-gaussians', type=int, default=500)
    p.add_argument('--iterations', type=int, default=1500)
    p.add_argument('--target-psnr', type=float, default=25.0,
                   help='Target PSNR for time-to-target metric')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- phantom
    vol = generate_phantom(
        shape=tuple(args.shape),
        num_blobs=args.num_blobs,
        blob_scale_range=(args.sigma, args.sigma + 1e-6),
        blob_amp_range=(0.8, 1.2),
        seed=args.seed,
    )
    print(f'Phantom: shape={vol.shape}, sigma~{args.sigma}, '
          f'#blobs={args.num_blobs}, range=[{vol.min():.3f},{vol.max():.3f}]')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    volume_t = torch.from_numpy(vol.astype(np.float32)).to(device)

    # ----------------------------------------------------------- train per strategy
    results = {}
    for strat in STRATEGIES:
        print(f'\n[{strat}] training {args.iterations} iters, '
              f'{args.num_gaussians} Gaussians')
        # Pure-init PSNR before training (no optimizer step).
        gs_init = init_gaussians(
            vol, num_gaussians=args.num_gaussians, strategy=strat,
            init_scale=2.0, seed=args.seed,
        ).to(device)
        psnr_init = init_psnr(gs_init, volume_t)
        print(f'  init PSNR (pre-training)  = {psnr_init:.2f} dB')

        t0 = time.time()
        gs, hist = train_static(
            vol, num_gaussians=args.num_gaussians, iterations=args.iterations,
            init_strategy=strat, init_scale=2.0, seed=args.seed,
            log_every=50, eval_every=0,
        )
        fit_time = time.time() - t0
        psnr_final = evaluate_full(gs, volume_t, subsample=100_000)
        iters_target = iters_to_target(hist, args.target_psnr)
        print(f'  final density PSNR        = {psnr_final:.2f} dB')
        print(f'  fit time                  = {fit_time:.1f} s')
        print(f'  iters to {args.target_psnr:.0f} dB (sample PSNR) = '
              f'{iters_target if iters_target >= 0 else "never"}')
        results[strat] = {
            'init_psnr': float(psnr_init),
            'final_psnr': float(psnr_final),
            'fit_time_s': float(fit_time),
            'iters_to_target': int(iters_target),
        }

    # ------------------------------------------------------------------- save
    report = {'config': vars(args), 'results': results}
    with open(out / 'e2_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print('\n--- E2 summary ---')
    print(f'{"strategy":<22} {"init":>8} {"final":>8} {"time(s)":>8} {"iters_to_tgt":>14}')
    for strat, r in results.items():
        print(f'{strat:<22} {r["init_psnr"]:>8.2f} {r["final_psnr"]:>8.2f} '
              f'{r["fit_time_s"]:>8.1f} {r["iters_to_target"]:>14}')
    print(f'\nE2 report -> {out / "e2_report.json"}')


if __name__ == '__main__':
    main()

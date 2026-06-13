"""E7 - temporal representation comparison (P5 exit criterion).

Fits the same 4D phantom with two representations and tabulates the trade-off:
  * shared-identity per-frame  - O(T*N) storage, high per-frame fidelity, tracking
  * native-4D                  - O(N) storage, continuous-t interpolation (E10)

Reports per-frame PSNR, mean PSNR, parameter count (storage), temporal smoothness,
and (native-4D) held-out interpolation PSNR.

Usage:
    python scripts/compare_temporal.py --frames 6 --num-gaussians 600 \
        --division-frame 4 --out-dir runs/e7_temporal
"""
import argparse
import json
from pathlib import Path

import numpy as np

from volsplat.temporal import (
    generate_phantom_4d, fit_timeseries_shared_identity, fit_native_4d,
    evaluate_4d_psnr, temporal_smoothness, frame_to_frame_consistency,
)


def _params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shape', type=int, nargs=3, default=[20, 40, 40])
    p.add_argument('--frames', type=int, default=6)
    p.add_argument('--num-blobs', type=int, default=10)
    p.add_argument('--num-gaussians', type=int, default=600)
    p.add_argument('--division-frame', type=int, default=None)
    p.add_argument('--out-dir', type=str, default='runs/e7_temporal')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    vols, tracks = generate_phantom_4d(
        shape=tuple(args.shape), num_blobs=args.num_blobs, num_frames=args.frames,
        division_frame=args.division_frame, seed=args.seed)
    T = len(vols)
    print(f'4D phantom: {T} frames {vols[0].shape}, blob counts {[t.shape[0] for t in tracks]}')

    # --- shared-identity ---
    sets, hist = fit_timeseries_shared_identity(
        vols, num_gaussians=args.num_gaussians, iters_first=1200, iters_warm=300, seed=args.seed)
    si_psnr = [h['psnr'] for h in hist]
    si_params = sum(_params(s) for s in sets)
    si_smooth = temporal_smoothness(sets)

    # --- native-4D ---
    gs4 = fit_native_4d(vols, num_gaussians=args.num_gaussians, iterations=2500, seed=args.seed)
    n4_psnr = [evaluate_4d_psnr(gs4, vols, t) for t in range(T)]
    n4_params = _params(gs4)
    # E10 interpolation at half-integer times (proxy GT = nearest frame)
    interp = [evaluate_4d_psnr(gs4, vols, t + 0.5) for t in range(T - 1)]

    table = {
        'shared_identity': {'per_frame_psnr': si_psnr, 'mean_psnr': float(np.mean(si_psnr)),
                            'params': si_params, 'temporal_smoothness': si_smooth},
        'native_4d': {'per_frame_psnr': n4_psnr, 'mean_psnr': float(np.mean(n4_psnr)),
                     'params': n4_params, 'interp_psnr': interp},
        'storage_ratio_si_over_4d': si_params / n4_params,
    }
    with open(out / 'e7_report.json', 'w') as f:
        json.dump({'config': vars(args), 'results': table}, f, indent=2)

    print('\n=== E7: temporal representation comparison ===')
    print(f'{"variant":<18}{"mean PSNR":>11}{"params":>10}{"storage":>10}')
    print(f'{"shared-identity":<18}{np.mean(si_psnr):>11.1f}{si_params:>10}{"O(T*N)":>10}')
    print(f'{"native-4D":<18}{np.mean(n4_psnr):>11.1f}{n4_params:>10}{"O(N)":>10}')
    print(f'\nstorage ratio (SI / 4D) = {si_params/n4_params:.1f}x')
    print(f'native-4D interpolation PSNR @half-frames = {[round(x,1) for x in interp]} dB')
    print(f'\nTrade-off: shared-identity wins per-frame fidelity + tracking;')
    print(f'native-4D wins storage ({si_params/n4_params:.1f}x smaller) + continuous-t rendering.')
    print(f'\nE7 report -> {out/"e7_report.json"}')


if __name__ == '__main__':
    main()

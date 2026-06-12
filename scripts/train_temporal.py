"""Fit a shared-identity temporal representation to a 4D phantom (P5 demo).

Generates a synthetic moving-blob time series (optionally with a division event),
fits the shared-identity per-frame representation, reports E7 temporal metrics,
and exports one .splat per frame so the result scrubs in viewer/index.html.

Usage:
    python scripts/train_temporal.py --frames 8 --num-gaussians 800 \
        --division-frame 5 --out-dir runs/temporal_demo
"""
import argparse
import json
from pathlib import Path

import torch

from volsplat.temporal import (
    generate_phantom_4d, fit_timeseries_shared_identity,
    temporal_smoothness, frame_to_frame_consistency,
)
from volsplat.export import to_splat


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shape', type=int, nargs=3, default=[24, 48, 48])
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--num-blobs', type=int, default=12)
    p.add_argument('--num-gaussians', type=int, default=800)
    p.add_argument('--iters-first', type=int, default=1500)
    p.add_argument('--iters-warm', type=int, default=400)
    p.add_argument('--division-frame', type=int, default=None)
    p.add_argument('--out-dir', type=str, default='runs/temporal_demo')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    vols, tracks = generate_phantom_4d(
        shape=tuple(args.shape), num_blobs=args.num_blobs, num_frames=args.frames,
        division_frame=args.division_frame, seed=args.seed)
    print(f'4D phantom: {len(vols)} frames {vols[0].shape}; '
          f'blob counts {[t.shape[0] for t in tracks]}')

    sets, hist = fit_timeseries_shared_identity(
        vols, num_gaussians=args.num_gaussians,
        iters_first=args.iters_first, iters_warm=args.iters_warm, seed=args.seed)

    psnrs = [h['psnr'] for h in hist]
    smooth = temporal_smoothness(sets)
    consist = frame_to_frame_consistency(sets, vols)
    print(f'per-frame PSNR : {[round(x,1) for x in psnrs]}')
    print(f'mean PSNR      : {sum(psnrs)/len(psnrs):.2f} dB')
    print(f'temporal smooth: {smooth:.4f} voxels (mean accel)')
    print(f'consistency    : {[round(x,1) for x in consist]} dB')

    # Export per-frame splats for the browser viewer.
    splat_dir = out / 'splats'; splat_dir.mkdir(exist_ok=True)
    for t, gs in enumerate(sets):
        ckpt = {'positions': gs.positions.detach().cpu(),
                'log_scales': gs.log_scales.detach().cpu(),
                'quaternions': gs.quaternions.detach().cpu(),
                'amp_logits': gs.amp_logits.detach().cpu(),
                'num_gaussians': gs.num_gaussians}
        to_splat(ckpt, splat_dir / f't{t:03d}.splat')

    with open(out / 'temporal_report.json', 'w') as f:
        json.dump({'config': vars(args), 'per_frame_psnr': psnrs,
                   'temporal_smoothness': smooth, 'consistency': consist}, f, indent=2)
    print(f'\nExported {len(sets)} frame splats -> {splat_dir}')
    print(f'View 3D+t: open viewer/index.html and load all t*.splat from {splat_dir}')


if __name__ == '__main__':
    main()

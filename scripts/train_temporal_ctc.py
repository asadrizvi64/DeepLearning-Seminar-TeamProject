"""Shared-identity temporal reconstruction on REAL Fluo-N3DL-DRO frames (P5 on real data).

Loads several consecutive CTC frames, crops them to the SAME ROI, fits the
shared-identity temporal representation, reports E7 metrics, and exports one
.splat per frame so the real embryo sequence scrubs in viewer/index.html.

Usage:
    python scripts/train_temporal_ctc.py --root "<DRO root>" --frames 5 \
        --target-shape 24 64 64 --num-gaussians 1500 --out-dir runs/ctc_temporal
"""
import argparse
import json
from pathlib import Path

import numpy as np

from volsplat.ctc import load_ctc_frame, find_intensity_bbox, center_roi, VOXEL_SIZE_DRO_UM
from volsplat.temporal import (
    fit_timeseries_shared_identity, temporal_smoothness, frame_to_frame_consistency,
)
from volsplat.export import to_splat


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--sequence', default='01')
    p.add_argument('--frames', type=int, default=5)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--target-shape', type=int, nargs=3, default=[24, 64, 64])
    p.add_argument('--num-gaussians', type=int, default=1500)
    p.add_argument('--iters-first', type=int, default=1500)
    p.add_argument('--iters-warm', type=int, default=400)
    p.add_argument('--out-dir', default='runs/ctc_temporal')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Fix one ROI from the first frame, apply it to all frames (consistent crop).
    f0 = load_ctc_frame(args.root, args.sequence, args.start, normalize=True)
    bbox = center_roi(find_intensity_bbox(f0), tuple(args.target_shape), f0.shape)
    vols = []
    for i in range(args.frames):
        v = load_ctc_frame(args.root, args.sequence, args.start + i, normalize=True)
        vols.append(v[bbox])
    print(f'Loaded {len(vols)} real frames, ROI {vols[0].shape}, '
          f'mean intensity {np.mean([v.mean() for v in vols]):.3f} '
          f'(dense nuclei: {np.mean([(v>0.01).mean() for v in vols]):.2f} nonzero)')

    sets, hist = fit_timeseries_shared_identity(
        vols, num_gaussians=args.num_gaussians,
        iters_first=args.iters_first, iters_warm=args.iters_warm, seed=args.seed)

    psnrs = [h['psnr'] for h in hist]
    smooth = temporal_smoothness(sets)
    consist = frame_to_frame_consistency(sets, vols)
    print(f'\nper-frame density PSNR : {[round(x,1) for x in psnrs]}')
    print(f'mean PSNR              : {np.mean(psnrs):.2f} dB')
    print(f'temporal smoothness    : {smooth:.3f} voxels (mean accel)')
    print(f'frame-to-frame consist : {[round(x,1) for x in consist]} dB')

    splat_dir = out / 'splats'; splat_dir.mkdir(exist_ok=True)
    for t, gs in enumerate(sets):
        ckpt = {'positions': gs.positions.detach().cpu(),
                'log_scales': gs.log_scales.detach().cpu(),
                'quaternions': gs.quaternions.detach().cpu(),
                'amp_logits': gs.amp_logits.detach().cpu(),
                'num_gaussians': gs.num_gaussians}
        to_splat(ckpt, splat_dir / f't{t:03d}.splat')

    with open(out / 'report.json', 'w') as f:
        json.dump({'config': vars(args), 'per_frame_psnr': psnrs,
                   'temporal_smoothness': smooth, 'consistency': consist,
                   'roi': [int(s.start) for s in bbox] + [int(s.stop) for s in bbox]}, f, indent=2)
    print(f'\nReal embryo 3D+t: {len(sets)} frame splats -> {splat_dir}')
    print(f'Scrub it: open viewer/index.html and load all t*.splat')


if __name__ == '__main__':
    main()

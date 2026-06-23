"""Fit a single frame of a real CTC volume (Fluo-N3DL-DRO targeted, but generic).

This is the bridge from synthetic phantom (P1 verified) to real microscopy.
Expect the PSNR ceiling to drop a lot - light-sheet volumes have shot noise,
background fluorescence, and non-Gaussian structure.

Usage:
    # 1. Look at the data first
    python scripts/inspect_ctc.py --root /path/to/Fluo-N3DL-DRO \
        --sequence 01 --frame 0 --crop --out-dir runs/inspect_t000

    # 2. Fit one frame (crop to intensity bbox, recenter to a fixed ROI)
    python scripts/train_ctc.py --root /path/to/Fluo-N3DL-DRO \
        --sequence 01 --frame 0 --crop --target-shape 128 256 256 \
        --num-gaussians 20000 --iterations 5000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from volsplat.ctc import (
    VOXEL_SIZE_DRO_UM,
    center_roi,
    find_intensity_bbox,
    load_ctc_frame,
)
from volsplat.train import evaluate_full, train_static


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, required=True, help='Path to Fluo-N3DL-DRO root')
    p.add_argument('--sequence', type=str, default='01')
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--crop', action='store_true',
                   help='Crop to intensity bounding box before fitting')
    p.add_argument('--target-shape', type=int, nargs=3, default=None,
                   metavar=('D', 'H', 'W'),
                   help='Recenter crop to this shape (requires --crop)')
    p.add_argument('--out-dir', type=str, default='runs/ctc_frame')
    p.add_argument('--num-gaussians', type=int, default=10000)
    p.add_argument('--iterations', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--init-scale-xy', type=float, default=2.0,
                   help='Initial Gaussian scale along x,y in voxel units')
    p.add_argument('--init-scale-z', type=float, default=0.4,
                   help='Initial Gaussian scale along z in voxel units '
                        '(~0.4 voxels = isotropic in microns for DRO)')
    p.add_argument('--densify-every', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    print(f'Loading {args.root}/{args.sequence}/t{args.frame:03d}.tif ...')
    vol = load_ctc_frame(args.root, args.sequence, args.frame, normalize=True)
    print(f'  shape (D,H,W) = {vol.shape},  range = [{vol.min():.3f}, {vol.max():.3f}]')

    # Crop — anchor z at the TOP surface of the bright bbox (blastoderm pole),
    # not the bbox centre (which is the empty interior for hollow-sphere embryos).
    if args.crop:
        bbox_full = find_intensity_bbox(vol)
        if args.target_shape is not None:
            D, H, W = vol.shape
            tz, ty, tx = args.target_shape
            z_s = max(0, bbox_full[0].start)
            z_e = min(D, z_s + tz); z_s = max(0, z_e - tz)
            y_c = (bbox_full[1].start + bbox_full[1].stop) // 2
            x_c = (bbox_full[2].start + bbox_full[2].stop) // 2
            y_s = max(0, min(H - ty, y_c - ty // 2))
            x_s = max(0, min(W - tx, x_c - tx // 2))
            bbox = (slice(z_s, z_s + tz), slice(y_s, y_s + ty), slice(x_s, x_s + tx))
        else:
            bbox = bbox_full
        vol = vol[bbox]
        print(f'  cropped to    = {vol.shape}')

    # Sanity log
    vx, vy, vz = VOXEL_SIZE_DRO_UM
    print(f'\nVoxel size (DRO): xy={vx} um, z={vz} um  (aspect z/xy = {vz/vx:.2f})')
    print(f'init scales (voxel): xy={args.init_scale_xy}, z={args.init_scale_z}')
    print(f'init scales (um)   : xy={args.init_scale_xy * vx:.3f}, '
          f'z={args.init_scale_z * vz:.3f}')

    # Fit
    gs, history = train_static(
        volume=vol,
        num_gaussians=args.num_gaussians,
        iterations=args.iterations,
        batch_size=args.batch_size,
        densify_every=args.densify_every,
        init_scale=(args.init_scale_xy, args.init_scale_xy, args.init_scale_z),
        seed=args.seed,
    )

    # Final eval
    volume_t = torch.from_numpy(vol.astype(np.float32)).to(gs.positions.device)
    final_psnr = evaluate_full(gs, volume_t, subsample=200_000)
    history.append({'final_psnr': float(final_psnr)})

    # Save the exact cropped ROI so the post-fit diagnostic is reproducible
    # (compare_fit.py --volume runs/.../roi.npy matches the fit's target exactly,
    # instead of re-deriving the crop and risking a mismatch).
    np.save(out / 'roi.npy', vol.astype(np.float32))
    print(f'  saved ROI -> {out / "roi.npy"}  (shape {vol.shape})')

    # Save
    torch.save({
        'positions':     gs.positions.detach().cpu(),
        'log_scales':    gs.log_scales.detach().cpu(),
        'quaternions':   gs.quaternions.detach().cpu(),
        'amp_logits':    gs.amp_logits.detach().cpu(),
        'num_gaussians': gs.num_gaussians,
        'volume_shape':  tuple(vol.shape),
        'voxel_size_um': VOXEL_SIZE_DRO_UM,
        'init_scale':    (args.init_scale_xy, args.init_scale_xy, args.init_scale_z),
        'sequence':      args.sequence,
        'frame':         args.frame,
    }, out / 'final.pt')

    with open(out / 'history.json', 'w') as f:
        json.dump(
            history, f, indent=2,
            default=lambda o: o.item() if hasattr(o, 'item') else str(o),
        )

    print(f'\nDone. Final N={gs.num_gaussians}, '
          f'final full-volume PSNR={final_psnr:.2f} dB')
    print(f'Outputs in: {out}')


if __name__ == '__main__':
    main()

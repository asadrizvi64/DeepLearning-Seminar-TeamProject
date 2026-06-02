"""Fit a single Lund Tribolium timepoint (Zenodo 5837363).

Sister of train_ctc.py for the simpler flat-TIFF Tribolium dataset. No GT loading,
no sequence/frame indexing - just point at one .tif and fit.

Usage:
    # Download a single timepoint first (~75 MB):
    # wget https://zenodo.org/records/5837363/files/lund_i000022_oi_000096.tif

    python scripts/train_tribolium.py \\
        --path lund_i000022_oi_000096.tif \\
        --crop --target-shape 96 256 256 \\
        --num-gaussians 30000 --iterations 5000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from volsplat.ctc import center_roi, find_intensity_bbox  # generic helpers
from volsplat.train import evaluate_full, train_static
from volsplat.tribolium import VOXEL_SIZE_LUND_UM, load_lund_volume


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, required=True, help='Path to one Lund .tif')
    p.add_argument('--crop', action='store_true',
                   help='Crop to intensity bounding box before fitting')
    p.add_argument('--target-shape', type=int, nargs=3, default=None,
                   metavar=('D', 'H', 'W'),
                   help='Recenter crop to this shape (requires --crop)')
    p.add_argument('--out-dir', type=str, default='runs/tribolium')
    p.add_argument('--num-gaussians', type=int, default=20000)
    p.add_argument('--iterations', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--init-scale-xy', type=float, default=2.0)
    p.add_argument('--init-scale-z', type=float, default=0.46,
                   help='~0.46 voxels = isotropic in microns for Lund (0.6934 / 3.0 * 2.0)')
    p.add_argument('--densify-every', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Loading {args.path}...')
    vol = load_lund_volume(args.path, normalize=True)
    print(f'  shape (D, H, W) = {vol.shape},  range = [{vol.min():.3f}, {vol.max():.3f}]')

    if args.crop:
        bbox = find_intensity_bbox(vol)
        if args.target_shape is not None:
            bbox = center_roi(bbox, tuple(args.target_shape), vol.shape)
        vol = vol[bbox]
        print(f'  cropped to    = {vol.shape}')

    vx, vy, vz = VOXEL_SIZE_LUND_UM
    print(f'\nVoxel size (Lund): xy={vx} um, z={vz} um (aspect z/xy = {vz/vx:.2f})')
    print(f'init scales (voxel): xy={args.init_scale_xy}, z={args.init_scale_z}')
    print(f'init scales (um)   : xy={args.init_scale_xy * vx:.3f}, '
          f'z={args.init_scale_z * vz:.3f}')

    gs, history = train_static(
        volume=vol,
        num_gaussians=args.num_gaussians,
        iterations=args.iterations,
        batch_size=args.batch_size,
        densify_every=args.densify_every,
        init_scale=(args.init_scale_xy, args.init_scale_xy, args.init_scale_z),
        seed=args.seed,
    )

    volume_t = torch.from_numpy(vol.astype(np.float32)).to(gs.positions.device)
    final_psnr = evaluate_full(gs, volume_t, subsample=200_000)
    history.append({'final_psnr': float(final_psnr)})

    torch.save({
        'positions':     gs.positions.detach().cpu(),
        'log_scales':    gs.log_scales.detach().cpu(),
        'quaternions':   gs.quaternions.detach().cpu(),
        'amp_logits':    gs.amp_logits.detach().cpu(),
        'num_gaussians': gs.num_gaussians,
        'volume_shape':  tuple(vol.shape),
        'voxel_size_um': VOXEL_SIZE_LUND_UM,
        'init_scale':    (args.init_scale_xy, args.init_scale_xy, args.init_scale_z),
        'source':        args.path,
    }, out / 'final.pt')

    with open(out / 'history.json', 'w') as f:
        json.dump(history, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))

    print(f'\nDone. Final N={gs.num_gaussians}, '
          f'final full-volume PSNR={final_psnr:.2f} dB')
    print(f'Outputs in: {out}')


if __name__ == '__main__':
    main()

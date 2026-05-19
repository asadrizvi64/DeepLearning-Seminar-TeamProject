"""Quick visual sanity check on a CTC volume.

Loads one frame, prints stats, dumps per-axis MIPs as PNGs (correcting the aspect
ratio for light-sheet voxel anisotropy), and summarizes the tracking GT.

Usage:
    python scripts/inspect_ctc.py --root /path/to/Fluo-N3DL-DRO \
        --sequence 01 --frame 0 --crop --out-dir runs/inspect_t000
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless: no display needed for PNG dumps
import matplotlib.pyplot as plt

from volsplat.ctc import (
    VOXEL_SIZE_DRO_UM,
    find_intensity_bbox,
    load_ctc_frame,
    load_tracking_labels,
    parse_man_track,
)


def _save_mip(volume, axis, label, path, voxel_aspect=None):
    img = volume.max(axis=axis)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='magma', aspect=voxel_aspect)
    ax.set_title(f'MIP along {label}  (image shape {img.shape})')
    ax.axis('off')
    fig.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, required=True, help='Path to Fluo-N3DL-DRO root')
    p.add_argument('--sequence', type=str, default='01')
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--out-dir', type=str, default='runs/inspect')
    p.add_argument('--no-normalize', action='store_true')
    p.add_argument('--crop', action='store_true', help='Crop to intensity bbox before MIP')
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Loading {args.root}/{args.sequence}/t{args.frame:03d}.tif ...')
    vol = load_ctc_frame(
        args.root, args.sequence, args.frame, normalize=not args.no_normalize
    )
    print(f'  shape (D,H,W) = {vol.shape}')
    print(f'  dtype         = {vol.dtype}')
    print(f'  range         = [{vol.min():.4f}, {vol.max():.4f}]')
    print(f'  mean          = {vol.mean():.4f}')
    print(f'  nonzero frac  = {(vol > 0.01).mean():.4f}  (intensity > 0.01)')

    if args.crop:
        bbox = find_intensity_bbox(vol)
        vol = vol[bbox]
        print(f'  cropped to    = {vol.shape}  (bbox {bbox})')

    vx, vy, vz = VOXEL_SIZE_DRO_UM
    # imshow aspect = (height_um / height_px) / (width_um / width_px).
    # XY (axis=0): (H, W) px <-> (H*vy, W*vx) um  =>  aspect = vy/vx = 1.
    # XZ (axis=1): (D, W) px <-> (D*vz, W*vx) um  =>  aspect = vz/vx ~ 5.
    # YZ (axis=2): (D, H) px <-> (D*vz, H*vy) um  =>  aspect = vz/vy ~ 5.
    _save_mip(vol, 0, 'Z (XY view)', out / 'mip_z.png', voxel_aspect=vy / vx)
    _save_mip(vol, 1, 'Y (XZ view)', out / 'mip_y.png', voxel_aspect=vz / vx)
    _save_mip(vol, 2, 'X (YZ view)', out / 'mip_x.png', voxel_aspect=vz / vy)
    print(f'\nMIPs -> {out}/')

    # Tracking GT summary
    tra = load_tracking_labels(args.root, args.sequence, args.frame)
    if tra is not None:
        uniq = np.unique(tra)
        n_tracks = int((uniq > 0).sum())
        print(f'\nman_track{args.frame:03d}.tif: shape {tra.shape}, {n_tracks} tracks in this frame')

    track_meta_path = Path(args.root) / f'{args.sequence}_GT' / 'TRA' / 'man_track.txt'
    if track_meta_path.exists():
        tracks = parse_man_track(track_meta_path)
        with_parents = sum(1 for t in tracks.values() if t.parent_id != 0)
        all_full = sum(
            1 for t in tracks.values()
            if t.begin_frame == 0 and t.end_frame == max(x.end_frame for x in tracks.values())
        )
        print(
            f'man_track.txt: {len(tracks)} tracks total; '
            f'{with_parents} post-division (have parent); '
            f'{all_full} span the full sequence'
        )


if __name__ == '__main__':
    main()

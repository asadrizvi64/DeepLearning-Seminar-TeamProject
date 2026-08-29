"""
Extract a fixed-size, pole-anchored ROI from a raw Fluo-N3DL-DRO frame,
so the real-data k-sweep uses the SAME crop size as the synthetic phantom.

Usage:
    python scripts/extract_real_roi.py \
        --root "D:/Download/train/Fluo-N3DL-DRO (1)/Fluo-N3DL-DRO" \
        --target-shape 64 128 128 \
        --out runs/real_roi_64x128x128.npy
"""
import argparse
from pathlib import Path

import numpy as np

from volsplat.ctc import load_ctc_frame, find_intensity_bbox


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Path to Fluo-N3DL-DRO root')
    p.add_argument('--sequence', default='01')
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--target-shape', type=int, nargs=3, default=[64, 128, 128],
                   metavar=('D', 'H', 'W'))
    p.add_argument('--out', default='runs/real_roi_64x128x128.npy')
    args = p.parse_args()

    print(f"Loading frame {args.frame} from {args.root} ...")
    vol = load_ctc_frame(args.root, args.sequence, args.frame, normalize=True)
    print(f"Full volume shape: {vol.shape}")

    # Pole-anchored crop (the fix): z at bbox TOP, y/x centred on embryo.
    bbox = find_intensity_bbox(vol)
    D, H, W = vol.shape
    tz, ty, tx = args.target_shape
    z_s = max(0, bbox[0].start)
    z_s = max(0, min(D - tz, z_s))
    y_c = (bbox[1].start + bbox[1].stop) // 2
    x_c = (bbox[2].start + bbox[2].stop) // 2
    y_s = max(0, min(H - ty, y_c - ty // 2))
    x_s = max(0, min(W - tx, x_c - tx // 2))

    roi = vol[z_s:z_s + tz, y_s:y_s + ty, x_s:x_s + tx]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, roi.astype(np.float32))

    print(f"\nPole-anchored ROI extracted:")
    print(f"  z = {z_s}..{z_s + tz}  (bbox top pole = surface nuclei)")
    print(f"  y = {y_s}..{y_s + ty}")
    print(f"  x = {x_s}..{x_s + tx}")
    print(f"  shape = {roi.shape}  ({roi.size} voxels)")
    print(f"  intensity range = [{roi.min():.3f}, {roi.max():.3f}], mean = {roi.mean():.3f}")
    print(f"  bright fraction (>0.5) = {(roi > 0.5).mean() * 100:.1f}%")
    print(f"\nSaved: {out}")
    print(f"\nNext: sweep k on this ROI with the same command as the phantom:")
    print(f"  python scripts/sweep_k_visual.py --volume {out} \\")
    print(f"      --out-dir runs/sweep_k_real --iters 1500 \\")
    print(f"      --k-values 1 10 50 100 150 200 250 300")


if __name__ == '__main__':
    main()

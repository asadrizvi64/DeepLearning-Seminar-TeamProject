"""Extract a fixed-size ROI from a real Tribolium (Lund) light-sheet frame.

The default 'nuclei' mode finds the window richest in DISTINCT nuclei (blob detection),
not merely the brightest tissue -- a saturated interior slab has high total intensity
but no structure to fit, so metrics there are not discriminative. We want a region with
clear nuclei against darker background. This is the real-data target for the greedy
ablation.

Modes:
    nuclei    (default) max count of local-maxima (distinct nuclei) in the window
    contrast  max intensity standard deviation (structured regions)
    density   max total intensity (brightest; legacy)

Usage:
    python scripts/extract_tribolium_roi.py \
        --path data/lund_i000022_oi_000096.tif \
        --target-shape 64 128 128 --mode nuclei \
        --out runs/tribolium_roi_64x128x128.npy
"""
import argparse
from pathlib import Path

import numpy as np

from volsplat.tribolium import load_lund_volume


def _best_window_center(score_map, target_shape):
    """Given a per-voxel local score (box-summed), return the clamped start indices of
    the highest-scoring window."""
    D, H, W = score_map.shape
    tz, ty, tx = target_shape
    cz, cy, cx = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
    z_s = int(np.clip(cz - tz // 2, 0, max(D - tz, 0)))
    y_s = int(np.clip(cy - ty // 2, 0, max(H - ty, 0)))
    x_s = int(np.clip(cx - tx // 2, 0, max(W - tx, 0)))
    return z_s, y_s, x_s


def find_roi_start(vol, target_shape, mode='nuclei'):
    from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter
    tz, ty, tx = target_shape
    if mode == 'density':
        # separable per-axis cumsum (legacy, fast)
        def densest(prof, ext, dim):
            if ext >= dim:
                return 0
            c = np.concatenate([[0.0], np.cumsum(prof)])
            return int(np.argmax(c[ext:] - c[:-ext]))
        return (densest(vol.sum((1, 2)), tz, vol.shape[0]),
                densest(vol.sum((0, 2)), ty, vol.shape[1]),
                densest(vol.sum((0, 1)), tx, vol.shape[2]))

    smoothed = gaussian_filter(vol.astype(np.float32), sigma=1.5)
    if mode == 'nuclei':
        peaks = ((maximum_filter(smoothed, size=3) == smoothed) & (smoothed > 0.3))
        signal = peaks.astype(np.float32)
    elif mode == 'contrast':
        local_mean = uniform_filter(smoothed, size=5)
        signal = (smoothed - local_mean) ** 2      # local variance proxy
    else:
        raise ValueError(f'unknown mode {mode!r}')
    # box-sum the signal over the target window -> per-voxel window score
    score = uniform_filter(signal, size=(tz, ty, tx), mode='constant')
    return _best_window_center(score, target_shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', default='data/lund_i000022_oi_000096.tif')
    p.add_argument('--target-shape', type=int, nargs=3, default=[64, 128, 128],
                   metavar=('D', 'H', 'W'))
    p.add_argument('--mode', default='nuclei', choices=['nuclei', 'contrast', 'density'])
    p.add_argument('--out', default='runs/tribolium_roi_64x128x128.npy')
    args = p.parse_args()

    print(f'Loading Tribolium frame {args.path} ...')
    vol = load_lund_volume(args.path, normalize=True)
    D, H, W = vol.shape
    tz, ty, tx = args.target_shape
    print(f'Full volume: {vol.shape}  bright%={float((vol>0.5).mean()*100):.1f}')

    z_start, y_start, x_start = find_roi_start(vol, (tz, ty, tx), mode=args.mode)
    print(f'ROI selection mode: {args.mode}')

    roi = vol[z_start:z_start + tz, y_start:y_start + ty, x_start:x_start + tx]
    roi = np.ascontiguousarray(roi.astype(np.float32))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, roi)

    print('\nTribolium ROI (densest window):')
    print(f'  z = {z_start}..{z_start + tz}')
    print(f'  y = {y_start}..{y_start + ty}')
    print(f'  x = {x_start}..{x_start + tx}')
    print(f'  shape = {roi.shape}  ({roi.size} voxels)')
    print(f'  range = [{roi.min():.3f}, {roi.max():.3f}]  mean = {roi.mean():.3f}')
    print(f'  bright%(>0.5) = {(roi > 0.5).mean() * 100:.1f}%')
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()

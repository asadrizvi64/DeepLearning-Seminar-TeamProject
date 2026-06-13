"""Visual + quantitative comparison of a fitted GaussianSet against its target volume.

Renders, for each axis:
  [ground truth MIP] [reconstruction MIP] [absolute-error MIP]
and a matched-slice version (slice chosen at the max-signal plane per axis).

Also reports residual statistics that full-volume PSNR hides:
  * PSNR over the whole volume   (inflated by empty-space agreement)
  * PSNR over non-empty voxels   (the honest structure-fidelity number)
  * mean |error| in empty vs structured regions

Usage:
    python scripts/compare_fit.py --volume data/phantom/phantom.tif \
        --checkpoint runs/p1_phantom/final.pt --out-dir runs/p1_phantom/compare
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from volsplat.data import load_volume
from volsplat.gaussians import GaussianSet


def load_gaussian_set(checkpoint_path) -> GaussianSet:
    """Rebuild a GaussianSet from a saved checkpoint, overwriting raw parameters
    directly so no log/softplus round-trip is applied."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    n = int(ckpt['num_gaussians'])
    gs = GaussianSet(torch.zeros(n, 3), torch.ones(n, 3))  # dummy, immediately replaced
    gs.positions = nn.Parameter(ckpt['positions'])
    gs.log_scales = nn.Parameter(ckpt['log_scales'])
    gs.quaternions = nn.Parameter(ckpt['quaternions'])
    gs.amp_logits = nn.Parameter(ckpt['amp_logits'])
    return gs, ckpt


def _psnr(pred, target, max_val=1.0):
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 0:
        return float('inf')
    return 20.0 * math.log10(max_val / math.sqrt(mse))


AXES = [(0, 'Z (XY view)'), (1, 'Y (XZ view)'), (2, 'X (YZ view)')]


def _grid(gt, recon, err, out_path, title, slicer=None):
    """3 rows (axes) x 3 cols (GT, recon, |error|). If slicer is None, use MIPs."""
    vmax = max(float(gt.max()), float(recon.max()))
    emax = float(err.max()) if err.max() > 0 else 1.0
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for r, (axis, name) in enumerate(AXES):
        if slicer is None:
            g = gt.max(axis=axis); p = recon.max(axis=axis); e = err.max(axis=axis)
        else:
            idx = slicer[axis]
            g = np.take(gt, idx, axis=axis)
            p = np.take(recon, idx, axis=axis)
            e = np.take(err, idx, axis=axis)
        im0 = axes[r, 0].imshow(g, cmap='magma', vmin=0, vmax=vmax)
        im1 = axes[r, 1].imshow(p, cmap='magma', vmin=0, vmax=vmax)
        im2 = axes[r, 2].imshow(e, cmap='viridis', vmin=0, vmax=emax)
        axes[r, 0].set_ylabel(name, fontsize=11)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)
    axes[0, 0].set_title('Ground truth', fontsize=13)
    axes[0, 1].set_title('Reconstruction', fontsize=13)
    axes[0, 2].set_title('|Error|', fontsize=13)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches='tight', dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--volume', type=str, default='data/phantom/phantom.tif')
    p.add_argument('--checkpoint', type=str, default='runs/p1_phantom/final.pt')
    p.add_argument('--out-dir', type=str, default='runs/p1_phantom/compare')
    p.add_argument('--empty-threshold', type=float, default=0.01,
                   help='GT intensity below this counts as empty space')
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # .npy is assumed to be an already-normalized ROI (e.g. train_ctc.py's roi.npy):
    # load it RAW so we compare against the exact target the fit saw. A cropped ROI
    # is normalized on the full frame then cropped, so re-running min-max here would
    # distort it. .tif goes through load_volume (min-max to [0,1]) as before.
    if str(args.volume).endswith('.npy'):
        gt = np.load(args.volume).astype(np.float32)
        print(f'Loaded ROI raw (no renormalization) from {args.volume}')
    else:
        gt = load_volume(args.volume)                   # (D, H, W) in [0,1]
    gs, ckpt = load_gaussian_set(args.checkpoint)
    print(f'Volume {gt.shape}, {gs.num_gaussians} Gaussians')
    pos = gs.positions.detach().numpy()
    print(f'Gaussian x/y/z position ranges: '
          f'x[{pos[:,0].min():.1f},{pos[:,0].max():.1f}] '
          f'y[{pos[:,1].min():.1f},{pos[:,1].max():.1f}] '
          f'z[{pos[:,2].min():.1f},{pos[:,2].max():.1f}]  '
          f'(volume is W={gt.shape[2]} H={gt.shape[1]} D={gt.shape[0]})')

    recon = gs.query_volume(gt.shape).cpu().numpy()
    err = np.abs(recon - gt)

    # ---- residual statistics -------------------------------------------------
    empty = gt < args.empty_threshold
    struct = ~empty
    psnr_all = _psnr(recon, gt)
    psnr_struct = _psnr(recon[struct], gt[struct]) if struct.any() else float('nan')
    print('\n--- residual statistics ---')
    print(f'  full-volume PSNR        = {psnr_all:6.2f} dB')
    print(f'  non-empty-voxel PSNR    = {psnr_struct:6.2f} dB   '
          f'(the honest structure number)')
    print(f'  empty-space fraction    = {empty.mean():.3f}')
    empty_err = f'{err[empty].mean():.5f}' if empty.any() else 'n/a (ROI fully dense)'
    struct_err = f'{err[struct].mean():.5f}' if struct.any() else 'n/a'
    print(f'  mean |err| in empty     = {empty_err}')
    print(f'  mean |err| in structure = {struct_err}')
    print(f'  max |err|               = {err.max():.5f}  at GT={gt[np.unravel_index(err.argmax(), err.shape)]:.3f}')
    print(f'  recon intensity range   = [{recon.min():.3f}, {recon.max():.3f}]  (GT is [0,1])')

    # ---- max-signal slice per axis ------------------------------------------
    slicer = {
        0: int(gt.sum(axis=(1, 2)).argmax()),
        1: int(gt.sum(axis=(0, 2)).argmax()),
        2: int(gt.sum(axis=(0, 1)).argmax()),
    }

    _grid(gt, recon, err, out / 'compare_mip.png',
          f'MIP comparison  -  full PSNR {psnr_all:.2f} dB / non-empty PSNR {psnr_struct:.2f} dB')
    _grid(gt, recon, err, out / 'compare_slice.png',
          f'Max-signal slice comparison (z={slicer[0]}, y={slicer[1]}, x={slicer[2]})',
          slicer=slicer)
    print(f'\nWrote {out/"compare_mip.png"} and {out/"compare_slice.png"}')


if __name__ == '__main__':
    main()

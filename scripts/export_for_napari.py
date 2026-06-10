"""Export ground-truth, reconstruction, and |error| volumes as TIFFs for napari.

Produces three aligned volumes you can open as layers in napari and scrub through
interactively — a clean "it works in 3D" live demo:

    <out>/gt.tif      ground-truth volume (the fit target)
    <out>/recon.tif   GaussianSet reconstruction (query_volume at the same grid)
    <out>/error.tif   |recon - gt|

Usage (phantom):
    python scripts/export_for_napari.py \
        --volume data/phantom/phantom.tif \
        --checkpoint runs/p1_phantom/final.pt \
        --out runs/p1_phantom/napari

Then view (using your napari bundle python):
    napari runs/p1_phantom/napari/gt.tif runs/p1_phantom/napari/recon.tif \
           runs/p1_phantom/napari/error.tif
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tifffile

from volsplat.data import load_volume
from volsplat.gaussians import GaussianSet


def load_gaussian_set(checkpoint_path):
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    n = int(ck['num_gaussians'])
    gs = GaussianSet(torch.zeros(n, 3), torch.ones(n, 3))
    gs.positions = nn.Parameter(ck['positions'])
    gs.log_scales = nn.Parameter(ck['log_scales'])
    gs.quaternions = nn.Parameter(ck['quaternions'])
    gs.amp_logits = nn.Parameter(ck['amp_logits'])
    return gs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--volume', required=True, help='.tif or .npy target volume')
    p.add_argument('--checkpoint', required=True, help='final.pt from a fit')
    p.add_argument('--out', default='runs/napari_export')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # .npy is an already-normalized ROI; load raw. .tif goes through load_volume.
    if str(args.volume).endswith('.npy'):
        gt = np.load(args.volume).astype(np.float32)
    else:
        gt = load_volume(args.volume)
    print(f'Target {gt.shape}, range [{gt.min():.3f}, {gt.max():.3f}]')

    gs = load_gaussian_set(args.checkpoint)
    print(f'Reconstructing from {gs.num_gaussians} Gaussians (this streams z-slabs)...')
    recon = gs.query_volume(gt.shape).cpu().numpy().astype(np.float32)
    err = np.abs(recon - gt).astype(np.float32)

    tifffile.imwrite(str(out / 'gt.tif'), gt)
    tifffile.imwrite(str(out / 'recon.tif'), recon)
    tifffile.imwrite(str(out / 'error.tif'), err)
    print(f'Wrote gt.tif / recon.tif / error.tif -> {out}')
    print(f'\nView in napari:\n  napari "{out / "gt.tif"}" "{out / "recon.tif"}" "{out / "error.tif"}"')


if __name__ == '__main__':
    main()

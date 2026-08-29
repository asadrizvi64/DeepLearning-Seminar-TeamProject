"""
Fit k=250 Gaussians and save the checkpoint for visualization.

Usage:
    python scripts/fit_k_optimal.py \
        --volume runs/p1_phantom/napari/gt.tif \
        --k 250 \
        --out-dir runs/fit_k250
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import tifffile

from volsplat.train import train_static


def load_volume(path):
    """Load volume from .npy or .tif."""
    if str(path).endswith('.npy'):
        return np.load(path).astype(np.float32)
    else:
        return tifffile.imread(path).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--volume', required=True, help='Ground truth volume (.npy or .tif)')
    p.add_argument('--k', type=int, default=250, help='Number of Gaussians')
    p.add_argument('--out-dir', default='runs/fit_k250', help='Output directory')
    p.add_argument('--iters', type=int, default=1500, help='Training iterations')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading volume: {args.volume}")
    volume = load_volume(args.volume)
    print(f"Volume shape: {volume.shape}")

    print(f"\nFitting {args.k} Gaussians ({args.iters} iterations)...")
    gs, history = train_static(
        volume=volume,
        num_gaussians=args.k,
        iterations=args.iters,
        seed=0,
    )

    # Save checkpoint
    ckpt = {
        'positions': gs.positions.detach().cpu(),
        'log_scales': gs.log_scales.detach().cpu(),
        'quaternions': gs.quaternions.detach().cpu(),
        'amp_logits': gs.amp_logits.detach().cpu(),
        'num_gaussians': gs.num_gaussians,
    }
    ckpt_path = out_dir / 'final.pt'
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # Final PSNR
    final_psnr = history[-1].get('full_psnr', history[-1].get('psnr', 0))
    print(f"\nFinal PSNR: {final_psnr:.2f} dB")
    print(f"Checkpoint saved and ready for visualization!")


if __name__ == '__main__':
    main()

"""
Generate publication-quality visualizations comparing ground truth vs fitted Gaussians.
Shows 3D slices + error maps + intensity histograms.

Usage:
    python scripts/visualize_fit.py \
        --volume runs/p1_phantom/napari/gt.tif \
        --checkpoint runs/sweep_k_extended/final.pt \
        --k 250 \
        --out-dir runs/visualizations_k250
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import SymLogNorm, Normalize
import tifffile

from volsplat.gaussians import GaussianSet
from volsplat.losses import psnr as compute_psnr


def load_volume(path):
    """Load volume from .npy or .tif."""
    if str(path).endswith('.npy'):
        return np.load(path).astype(np.float32)
    else:
        return tifffile.imread(path).astype(np.float32)


def compute_reconstruction(checkpoint_path, volume_shape, device='cpu'):
    """Load checkpoint and reconstruct volume."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    positions = torch.from_numpy(ckpt['positions'].cpu().numpy()).to(device)
    log_scales = torch.from_numpy(ckpt['log_scales'].cpu().numpy()).to(device)
    quaternions = torch.from_numpy(ckpt['quaternions'].cpu().numpy()).to(device)
    amp_logits = torch.from_numpy(ckpt['amp_logits'].cpu().numpy()).to(device)

    gs = GaussianSet(positions, log_scales, quaternions, amp_logits)
    recon = gs.query_volume(volume_shape).cpu().numpy().astype(np.float32)
    return recon


def create_comparison_figure(gt, recon, k, out_dir):
    """Create side-by-side comparison with 3D slices and error maps."""
    D, H, W = gt.shape
    z_mid, y_mid, x_mid = D // 2, H // 2, W // 2

    # Compute metrics
    mse = np.mean((gt - recon) ** 2)
    psnr_val = compute_psnr(recon, gt)
    max_error = np.abs(gt - recon).max()
    mean_error = np.abs(gt - recon).mean()

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'k={k} Gaussians | PSNR={psnr_val:.2f} dB | MSE={mse:.4f}',
                 fontsize=16, fontweight='bold', y=0.98)

    slices = [
        (gt[z_mid, :, :], recon[z_mid, :, :], f'Z={z_mid} (XY plane)'),
        (gt[:, y_mid, :], recon[:, y_mid, :], f'Y={y_mid} (XZ plane)'),
        (gt[:, :, x_mid], recon[:, :, x_mid], f'X={x_mid} (YZ plane)'),
    ]

    for row, (gt_slice, recon_slice, label) in enumerate(slices):
        # Ground truth
        ax1 = plt.subplot(3, 4, row*4 + 1)
        im1 = ax1.imshow(gt_slice, cmap='viridis', interpolation='bilinear')
        ax1.set_title(f'Ground Truth: {label}', fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Reconstructed
        ax2 = plt.subplot(3, 4, row*4 + 2)
        im2 = ax2.imshow(recon_slice, cmap='viridis', interpolation='bilinear')
        ax2.set_title(f'Reconstructed (k={k})', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Absolute error
        ax3 = plt.subplot(3, 4, row*4 + 3)
        diff = np.abs(gt_slice - recon_slice)
        im3 = ax3.imshow(diff, cmap='hot', interpolation='bilinear')
        ax3.set_title(f'Absolute Error', fontweight='bold')
        ax3.axis('off')
        cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label(f'Max={max_error:.4f}', fontsize=9)

        # Relative error (%)
        ax4 = plt.subplot(3, 4, row*4 + 4)
        gt_max = gt_slice.max()
        if gt_max > 0:
            rel_error = (diff / (gt_max + 1e-6)) * 100
        else:
            rel_error = diff * 100
        im4 = ax4.imshow(rel_error, cmap='RdYlGn_r', interpolation='bilinear',
                         vmin=0, vmax=50)
        ax4.set_title(f'Relative Error (%)', fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Add text box with metrics
    textstr = f'Metrics:\nPSNR: {psnr_val:.2f} dB\nMSE: {mse:.6f}\nMax Error: {max_error:.4f}\nMean Error: {mean_error:.4f}'
    fig.text(0.02, 0.02, textstr, fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def create_histogram_figure(gt, recon, k, out_dir):
    """Create intensity and error histograms."""
    gt_flat = gt.flatten()
    recon_flat = recon.flatten()
    error_flat = np.abs(gt_flat - recon_flat)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'k={k} Gaussians: Intensity and Error Distributions',
                 fontsize=14, fontweight='bold')

    # Ground truth intensity
    axes[0].hist(gt_flat, bins=64, alpha=0.7, label='Ground Truth', color='blue', edgecolor='black')
    axes[0].hist(recon_flat, bins=64, alpha=0.7, label='Reconstructed', color='red', edgecolor='black')
    axes[0].set_xlabel('Intensity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Intensity Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    axes[1].hist(error_flat, bins=64, color='orange', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Absolute Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    axes[1].grid(True, alpha=0.3)

    # Scatter: GT vs Recon
    # Sample for clarity
    sample_idx = np.random.choice(len(gt_flat), size=min(10000, len(gt_flat)), replace=False)
    axes[2].scatter(gt_flat[sample_idx], recon_flat[sample_idx], alpha=0.3, s=1)
    axes[2].plot([gt_flat.min(), gt_flat.max()], [gt_flat.min(), gt_flat.max()],
                 'r--', label='Perfect fit')
    axes[2].set_xlabel('Ground Truth')
    axes[2].set_ylabel('Reconstructed')
    axes[2].set_title('GT vs Reconstructed (scatter)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--volume', required=True, help='Ground truth volume (.npy or .tif)')
    p.add_argument('--checkpoint', required=True, help='Fitted checkpoint (.pt)')
    p.add_argument('--k', type=int, required=True, help='Number of Gaussians fitted')
    p.add_argument('--out-dir', default='runs/visualizations', help='Output directory')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading volume: {args.volume}")
    gt = load_volume(args.volume)
    print(f"Volume shape: {gt.shape}")

    print(f"Loading checkpoint: {args.checkpoint}")
    recon = compute_reconstruction(args.checkpoint, gt.shape, device='cpu')

    psnr_val = compute_psnr(recon, gt)
    print(f"PSNR: {psnr_val:.2f} dB")

    # Generate figures
    print("Generating comparison figure...")
    fig1 = create_comparison_figure(gt, recon, args.k, out_dir)
    fig1_path = out_dir / f'comparison_k{args.k}.png'
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig1_path}")
    plt.close(fig1)

    print("Generating histogram figure...")
    fig2 = create_histogram_figure(gt, recon, args.k, out_dir)
    fig2_path = out_dir / f'histograms_k{args.k}.png'
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig2_path}")
    plt.close(fig2)

    print(f"\nAll visualizations saved to: {out_dir}")
    print("Ready to show your supervisor!")


if __name__ == '__main__':
    main()

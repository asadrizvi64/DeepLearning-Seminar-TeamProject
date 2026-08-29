"""Compare k=50 vs k=300 side-by-side with actual (shows if more Gaussians help)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile

REPO = Path(__file__).resolve().parent.parent


def main():
    # Load both fits
    gt = tifffile.imread(REPO / 'runs/napari_real_k50/gt.tif')
    recon_k50 = tifffile.imread(REPO / 'runs/napari_real_k50/recon.tif')
    recon_k300 = tifffile.imread(REPO / 'runs/napari_real_k300/recon.tif')

    D, H, W = gt.shape
    z_mid, y_mid, x_mid = D // 2, H // 2, W // 2

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Real Embryo: Actual vs k=50 vs k=300 Fit', fontsize=16, fontweight='bold')

    planes = [
        ('XY plane (z-mid)', gt[z_mid], recon_k50[z_mid], recon_k300[z_mid], 0),
        ('XZ plane (y-mid)', gt[:, y_mid, :], recon_k50[:, y_mid, :], recon_k300[:, y_mid, :], 1),
        ('YZ plane (x-mid)', gt[:, :, x_mid], recon_k50[:, :, x_mid], recon_k300[:, :, x_mid], 2),
    ]

    for title, gt_slice, k50_slice, k300_slice, row in planes:
        # Actual
        im0 = axes[row, 0].imshow(gt_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 0].set_title(f'Actual: {title}', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # k=50
        im1 = axes[row, 1].imshow(k50_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 1].set_title(f'k=50 (PSNR=21.04)', fontsize=11, fontweight='bold')
        axes[row, 1].axis('off')
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # k=300
        im2 = axes[row, 2].imshow(k300_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 2].set_title(f'k=300 (PSNR=22.17)', fontsize=11, fontweight='bold')
        axes[row, 2].axis('off')
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = REPO / 'runs/k50_vs_k300_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)

    # Metrics
    psnr_k50 = 10 * np.log10((gt.max() - gt.min())**2 / np.mean((gt - recon_k50)**2))
    psnr_k300 = 10 * np.log10((gt.max() - gt.min())**2 / np.mean((gt - recon_k300)**2))

    print(f'\nComparison:')
    print(f'  k=50:  PSNR = {psnr_k50:.2f} dB')
    print(f'  k=300: PSNR = {psnr_k300:.2f} dB')
    print(f'  Improvement: +{psnr_k300 - psnr_k50:.2f} dB (6x more Gaussians)')


if __name__ == '__main__':
    main()

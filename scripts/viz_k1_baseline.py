"""Baseline: k=1 shows what a single isotropic Gaussian captures."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile

REPO = Path(__file__).resolve().parent.parent


def main():
    gt = tifffile.imread(REPO / 'runs/napari_real_k1/gt.tif')
    recon = tifffile.imread(REPO / 'runs/napari_real_k1/recon.tif')
    error = tifffile.imread(REPO / 'runs/napari_real_k1/error.tif')

    D, H, W = gt.shape
    z_mid, y_mid, x_mid = D // 2, H // 2, W // 2

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Baseline: k=1 Single Isotropic Gaussian on Real Embryo',
                 fontsize=16, fontweight='bold')

    planes = [
        ('XY plane (z-mid)', gt[z_mid], recon[z_mid], error[z_mid], 0),
        ('XZ plane (y-mid)', gt[:, y_mid, :], recon[:, y_mid, :], error[:, y_mid, :], 1),
        ('YZ plane (x-mid)', gt[:, :, x_mid], recon[:, :, x_mid], error[:, :, x_mid], 2),
    ]

    for title, gt_slice, recon_slice, error_slice, row in planes:
        # Actual
        im0 = axes[row, 0].imshow(gt_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 0].set_title(f'Actual: {title}', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # k=1 fit
        im1 = axes[row, 1].imshow(recon_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 1].set_title(f'k=1 Fit: {title}', fontsize=11, fontweight='bold')
        axes[row, 1].axis('off')
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Residual (what k=1 is missing)
        im2 = axes[row, 2].imshow(error_slice, cmap='hot')
        axes[row, 2].set_title(f'Residual (k=1 missing): {title}', fontsize=11, fontweight='bold')
        axes[row, 2].axis('off')
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = REPO / 'runs/k1_baseline.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)

    # Metrics
    psnr = 10 * np.log10((gt.max() - gt.min())**2 / np.mean((gt - recon)**2))
    print(f'\nk=1 Baseline:')
    print(f'  PSNR: {psnr:.2f} dB')
    print(f'  Mean intensity: {gt.mean():.3f}')
    print(f'  k=1 captures: {(1 - np.mean(error) / gt.mean()) * 100:.1f}% of signal (rough)')


if __name__ == '__main__':
    main()

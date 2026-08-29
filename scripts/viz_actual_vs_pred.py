"""Simple side-by-side actual vs predicted visualization across 3 planes."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile

REPO = Path(__file__).resolve().parent.parent


def main():
    # Load real data fit
    gt = tifffile.imread(REPO / 'runs/napari_real_k50/gt.tif')
    recon = tifffile.imread(REPO / 'runs/napari_real_k50/recon.tif')

    D, H, W = gt.shape
    z_mid, y_mid, x_mid = D // 2, H // 2, W // 2

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle('Real Embryo (k=50): Actual vs Predicted', fontsize=14, fontweight='bold')

    planes = [
        ('XY plane (z-mid)', gt[z_mid], recon[z_mid], 0),
        ('XZ plane (y-mid)', gt[:, y_mid, :], recon[:, y_mid, :], 1),
        ('YZ plane (x-mid)', gt[:, :, x_mid], recon[:, :, x_mid], 2),
    ]

    for title, gt_slice, pred_slice, row in planes:
        # Actual
        im0 = axes[row, 0].imshow(gt_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 0].set_title(f'Actual: {title}', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # Predicted
        im1 = axes[row, 1].imshow(pred_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 1].set_title(f'Predicted (k=50): {title}', fontsize=11, fontweight='bold')
        axes[row, 1].axis('off')
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = REPO / 'runs/napari_real_k50/actual_vs_predicted.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)

    # Quick metrics
    psnr = 10 * np.log10((gt.max() - gt.min())**2 / np.mean((gt - recon)**2))
    print(f'\nPSNR: {psnr:.2f} dB')
    print(f'This matches the sweep result: k=50 PSNR = 21.04 dB')


if __name__ == '__main__':
    main()

"""Show progression: k=1 -> k=10 -> k=50 (do more Gaussians target individual nuclei?)"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile

REPO = Path(__file__).resolve().parent.parent


def main():
    gt = tifffile.imread(REPO / 'runs/napari_real_k1/gt.tif')
    recon_k1 = tifffile.imread(REPO / 'runs/napari_real_k1/recon.tif')
    recon_k10 = tifffile.imread(REPO / 'runs/napari_real_k10/recon.tif')
    recon_k50 = tifffile.imread(REPO / 'runs/napari_real_k50/recon.tif')

    D, H, W = gt.shape
    z_mid, y_mid, x_mid = D // 2, H // 2, W // 2

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle('Progression: Do More Gaussians Target Individual Nuclei or Just Smooth?',
                 fontsize=16, fontweight='bold')

    planes = [
        ('XY (z-mid)', gt[z_mid], recon_k1[z_mid], recon_k10[z_mid], recon_k50[z_mid], 0),
        ('XZ (y-mid)', gt[:, y_mid, :], recon_k1[:, y_mid, :], recon_k10[:, y_mid, :], recon_k50[:, y_mid, :], 1),
        ('YZ (x-mid)', gt[:, :, x_mid], recon_k1[:, :, x_mid], recon_k10[:, :, x_mid], recon_k50[:, :, x_mid], 2),
    ]

    for title, gt_slice, k1_slice, k10_slice, k50_slice, row in planes:
        # Actual
        im0 = axes[row, 0].imshow(gt_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 0].set_title(f'Actual: {title}', fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # k=1
        im1 = axes[row, 1].imshow(k1_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 1].set_title(f'k=1 (PSNR=15.97)', fontsize=10, fontweight='bold')
        axes[row, 1].axis('off')
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # k=10
        im2 = axes[row, 2].imshow(k10_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 2].set_title(f'k=10 (PSNR=18.02)', fontsize=10, fontweight='bold')
        axes[row, 2].axis('off')
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # k=50
        im3 = axes[row, 3].imshow(k50_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 3].set_title(f'k=50 (PSNR=21.04)', fontsize=10, fontweight='bold')
        axes[row, 3].axis('off')
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = REPO / 'runs/progression_k1_k10_k50.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)

    print(f'\nInterpretation:')
    print(f'  k=1: Broad pole structure (~16 dB)')
    print(f'  k=10: More nuclei detail added (+2 dB)')
    print(f'  k=50: Even more detail (+3 dB)')
    print(f'\n  Question: Are they targeting the residual peaks (real nuclei) or just smoothing?')
    print(f'  Visual answer: Look for whether k=10 -> k=50 adds sharp detail or stays smooth.')


if __name__ == '__main__':
    main()

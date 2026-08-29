"""Generate publication-quality k=50 real-data fit visualization."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    # Load the TIFFs we just exported
    import tifffile
    gt = tifffile.imread(REPO / 'runs/napari_real_k50/gt.tif')
    recon = tifffile.imread(REPO / 'runs/napari_real_k50/recon.tif')
    error = tifffile.imread(REPO / 'runs/napari_real_k50/error.tif')

    D, H, W = gt.shape
    z_mid, y_mid, x_mid = D // 2, H // 2, W // 2

    # 3 planes × 4 columns (GT, recon, abs error, rel error %)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Real Embryo (Fluo-N3DL-DRO) — k=50 Gaussian Fit',
                 fontsize=16, fontweight='bold', y=0.995)

    planes = [
        ('XY (z-slice)', gt[z_mid], recon[z_mid], error[z_mid], 0),
        ('XZ (y-slice)', gt[:, y_mid, :], recon[:, y_mid, :], error[:, y_mid, :], 1),
        ('YZ (x-slice)', gt[:, :, x_mid], recon[:, :, x_mid], error[:, :, x_mid], 2),
    ]

    for plane_name, gt_slice, recon_slice, error_slice, row in planes:
        # Ground truth
        im0 = axes[row, 0].imshow(gt_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 0].set_title(f'GT: {plane_name}', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # Reconstruction
        im1 = axes[row, 1].imshow(recon_slice, cmap='viridis', vmin=0, vmax=1)
        axes[row, 1].set_title(f'k=50 Fit: {plane_name}', fontsize=11, fontweight='bold')
        axes[row, 1].axis('off')
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Absolute error
        im2 = axes[row, 2].imshow(error_slice, cmap='hot')
        axes[row, 2].set_title(f'|Error|: {plane_name}', fontsize=11, fontweight='bold')
        axes[row, 2].axis('off')
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # Relative error (%)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = (error_slice / (np.abs(gt_slice) + 1e-6)) * 100
            rel_error = np.clip(rel_error, 0, 200)  # cap at 200% for visibility
        im3 = axes[row, 3].imshow(rel_error, cmap='RdYlGn_r', vmin=0, vmax=100)
        axes[row, 3].set_title(f'Error %: {plane_name}', fontsize=11, fontweight='bold')
        axes[row, 3].axis('off')
        cbar = plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)
        cbar.set_label('%', fontsize=9)

    plt.tight_layout()
    out = REPO / 'runs/napari_real_k50/fit_visualization.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)

    # Metrics summary
    psnr = 10 * np.log10((gt.max() - gt.min())**2 / np.mean((gt - recon)**2))
    mse = np.mean((gt - recon)**2)
    mae = np.mean(np.abs(gt - recon))
    max_err = np.max(np.abs(gt - recon))

    print(f'\nMetrics (k=50 vs GT):')
    print(f'  PSNR: {psnr:.2f} dB')
    print(f'  MSE:  {mse:.6f}')
    print(f'  MAE:  {mae:.6f}')
    print(f'  Max:  {max_err:.6f}')
    print(f'\nGT range: [{gt.min():.3f}, {gt.max():.3f}]')
    print(f'Recon range: [{recon.min():.3f}, {recon.max():.3f}]')


if __name__ == '__main__':
    main()

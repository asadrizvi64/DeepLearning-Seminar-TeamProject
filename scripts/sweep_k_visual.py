"""
Sweep number of Gaussians (k) from 1 to 100 on a fixed crop.
For each k: fit, measure PSNR, generate visualization slices and intensity difference maps.

Usage:
    python scripts/sweep_k_visual.py --volume runs/synthetic_64x64x64.npy --out-dir runs/sweep_k_results
    python scripts/sweep_k_visual.py --volume path/to/real_roi.npy --out-dir runs/sweep_k_real
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from volsplat.train import train_static, evaluate_full
from volsplat.losses import psnr as compute_psnr


def visualize_slices(ground_truth, reconstructed, title_prefix, out_dir):
    """Generate side-by-side slice comparisons (z, y, x midplanes)."""
    D, H, W = ground_truth.shape
    z_mid, y_mid, x_mid = D // 2, H // 2, W // 2

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(f"{title_prefix} - Ground Truth vs Reconstructed", fontsize=14, fontweight='bold')

    slices = [
        (ground_truth[z_mid, :, :], reconstructed[z_mid, :, :], f'Z={z_mid} (XY plane)', 0),
        (ground_truth[:, y_mid, :], reconstructed[:, y_mid, :], f'Y={y_mid} (XZ plane)', 1),
        (ground_truth[:, :, x_mid], reconstructed[:, :, x_mid], f'X={x_mid} (YZ plane)', 2),
    ]

    for gt_slice, recon_slice, label, row in slices:
        # Ground truth
        axes[row, 0].imshow(gt_slice, cmap='viridis')
        axes[row, 0].set_title(f'GT: {label}')
        axes[row, 0].axis('off')

        # Reconstructed
        axes[row, 1].imshow(recon_slice, cmap='viridis')
        axes[row, 1].set_title(f'Recon: {label}')
        axes[row, 1].axis('off')

        # Difference (error map)
        diff = np.abs(gt_slice - recon_slice)
        im = axes[row, 2].imshow(diff, cmap='hot')
        axes[row, 2].set_title(f'|Error|: {label}')
        axes[row, 2].axis('off')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def sweep_k(volume_path, out_dir, k_values=None, iterations=1500):
    """
    Sweep k (number of Gaussians) and generate results.

    Args:
        volume_path: Path to .npy or .tif volume, or 'generate' for synthetic phantom
        out_dir: Output directory for results
        k_values: List of k to try. Default: [1, 2, 3, 5, 10, 20, 50, 100]
        iterations: Training iterations per fit
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if k_values is None:
        k_values = [1, 2, 3, 5, 10, 20, 50, 100]

    # Load or generate volume
    print(f"Loading {volume_path}...")
    if volume_path.lower() == 'generate':
        print("Generating synthetic phantom...")
        from volsplat.temporal import generate_phantom_4d
        volumes, _ = generate_phantom_4d(shape=(48, 48, 48), num_blobs=15, num_frames=1, seed=0)
        volume = volumes[0].astype(np.float32)
    elif str(volume_path).endswith('.npy'):
        volume = np.load(volume_path).astype(np.float32)
    elif str(volume_path).endswith('.tif'):
        import tifffile
        volume = tifffile.imread(volume_path).astype(np.float32)
    else:
        raise ValueError(f"Unknown volume format: {volume_path}")
    print(f"Volume shape: {volume.shape}")

    volume_torch = torch.from_numpy(volume)
    if torch.cuda.is_available():
        volume_torch = volume_torch.cuda()

    results = []
    viz_dir = out_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    print("\nSweeping k...")
    for i, k in enumerate(k_values):
        print(f"\n[{i+1}/{len(k_values)}] Fitting k={k} Gaussians...")

        # Fit
        gs, history = train_static(
            volume=volume,
            num_gaussians=k,
            iterations=iterations,
            seed=0,
        )

        # Reconstruct and measure PSNR
        recon = gs.query_volume(volume.shape).cpu().numpy().astype(np.float32)
        final_psnr = float(compute_psnr(recon, volume))

        # Also compute full-volume PSNR from training history
        hist_psnr = history[-1].get('full_psnr', final_psnr)

        result = {'k': k, 'psnr': final_psnr, 'hist_psnr': hist_psnr}
        results.append(result)
        print(f"  k={k:3d}: PSNR={final_psnr:.2f} dB")

        # Generate visualizations for selected k values (skip tiny ones to save time)
        if k in [1, 3, 10, 20, 50, 100] or i == len(k_values) - 1:
            print(f"  Generating visualizations...")
            fig = visualize_slices(volume, recon, f'k={k}', viz_dir)
            viz_path = viz_dir / f'slices_k{k:03d}.png'
            fig.savefig(viz_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {viz_path}")

    # Save results
    results_path = out_dir / 'sweep_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Plot PSNR vs k (the main curve)
    ks = [r['k'] for r in results]
    psnrs = [r['psnr'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, psnrs, 'o-', linewidth=2, markersize=8, label='PSNR')
    ax.set_xlabel('Number of Gaussians (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('PSNR vs Number of Gaussians', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xscale('log')

    # Annotate knee point (approximate: largest second derivative)
    if len(psnrs) > 2:
        first_deriv = np.diff(psnrs)
        second_deriv = np.diff(first_deriv)
        knee_idx = np.argmax(np.abs(second_deriv)) + 1
        knee_k = ks[knee_idx]
        knee_psnr = psnrs[knee_idx]
        ax.plot(knee_k, knee_psnr, 'r*', markersize=20, label=f'Knee: k={knee_k}')
        ax.legend()

    curve_path = out_dir / 'psnr_vs_k.png'
    fig.savefig(curve_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Curve saved: {curve_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"k={r['k']:3d}: PSNR={r['psnr']:6.2f} dB")

    improvement = psnrs[-1] - psnrs[0]
    print(f"\nTotal improvement (k=1 -> k={ks[-1]}): {improvement:.2f} dB")
    print(f"\nTo show your supervisor:")
    print(f"  1. Open: {curve_path}")
    print(f"  2. Open: {viz_dir}/slices_k*.png")
    print(f"  3. Say: 'I swept k from 1 to {ks[-1]}. PSNR saturates around k={ks[results.index(max(results, key=lambda r: r['psnr']))]}.'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--volume', required=True, help='Path to .npy/.tif volume or "generate" for synthetic')
    p.add_argument('--out-dir', default='runs/sweep_k_results', help='Output directory')
    p.add_argument('--iters', type=int, default=1500, help='Training iterations per k')
    p.add_argument('--k-values', type=int, nargs='+', default=None,
                   help='k values to sweep (default: 1 2 3 5 10 20 50 100)')
    args = p.parse_args()

    if args.k_values is None:
        k_values = [1, 2, 3, 5, 10, 20, 50, 100]
    else:
        k_values = args.k_values

    sweep_k(args.volume, args.out_dir, k_values=k_values, iterations=args.iters)


if __name__ == '__main__':
    main()

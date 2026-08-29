"""Capacity sweep: how far does held-out quality climb as we add Gaussians?

Uses the greedy-selected champion config (full parameterization, intensity_weighted
init, init_scale=4.0) and pushes the Gaussian count up. Answers the open question from
the champion-vs-baseline visual: is the smooth residual a CAPACITY limit (more Gaussians
would capture the fine texture) or a representation limit (it saturates)?

Validation-only during the sweep (skips the expensive dense reconstruction); the largest
k is reconstructed once at the end for a visual texture check.

Usage:
    python scripts/capacity_sweep.py --ks 100 250 500 1000 2000 --iters 1000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from volsplat.ablation import fit_with_validation

REPO = Path(__file__).resolve().parent.parent

CHAMPION = dict(init_strategy='intensity_weighted', parameterization='full', init_scale=4.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--volume', default='runs/tribolium_roi_64x128x128.npy')
    p.add_argument('--out-dir', default='runs/capacity_tribolium')
    p.add_argument('--ks', type=int, nargs='+', default=[100, 250, 500, 1000, 2000])
    p.add_argument('--iters', type=int, default=1000)
    p.add_argument('--val-fraction', type=float, default=0.1)
    args = p.parse_args()

    out_dir = Path(REPO / args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    volume = np.load(REPO / args.volume).astype(np.float32)
    print(f'Volume {volume.shape}, champion config {CHAMPION}, iters {args.iters}\n')

    rows = []
    last_gs = None
    for k in args.ks:
        is_last = (k == args.ks[-1])
        gs, _, res = fit_with_validation(
            volume, num_gaussians=k, iterations=args.iters,
            val_fraction=args.val_fraction, full_recon=is_last, **CHAMPION)
        row = {'k': k, 'val_psnr': res['val_psnr'], 'val_mae': res['val_mae'],
               'val_corr': res['val_corr'], 'num_params': res['num_params'],
               'compression': res['compression'], 'fit_seconds': res['fit_seconds']}
        if 'full_ssim' in res:
            row['full_ssim'] = res['full_ssim']
        rows.append(row)
        print(f"  k={k:5d}  val_psnr={row['val_psnr']:.2f}  params={row['num_params']:6d}  "
              f"compression={row['compression']:.0f}x  time={row['fit_seconds']:.0f}s")
        if is_last:
            last_gs = gs

    with open(out_dir / 'capacity_results.json', 'w') as f:
        json.dump(rows, f, indent=2)

    ks = [r['k'] for r in rows]
    valp = [r['val_psnr'] for r in rows]
    nparams = [r['num_params'] for r in rows]

    # val PSNR vs k, and vs #params (efficiency)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
    a1.plot(ks, valp, 'o-', lw=2.5, ms=9, color='#2ca02c')
    for k, v in zip(ks, valp):
        a1.annotate(f'{v:.2f}', (k, v), textcoords='offset points', xytext=(0, 9),
                    ha='center', fontsize=9, fontweight='bold')
    a1.set_xscale('log'); a1.set_xlabel('Number of Gaussians (k)', fontweight='bold')
    a1.set_ylabel('held-out validation PSNR (dB)', fontweight='bold')
    a1.set_title('Capacity: does quality keep climbing?', fontsize=13, fontweight='bold')
    a1.grid(True, alpha=0.3)

    a2.plot(nparams, valp, 's-', lw=2.5, ms=9, color='#1f77b4')
    a2.set_xscale('log'); a2.set_xlabel('# free parameters', fontweight='bold')
    a2.set_ylabel('held-out validation PSNR (dB)', fontweight='bold')
    a2.set_title('Efficiency frontier (PSNR vs parameters)', fontsize=13, fontweight='bold')
    a2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'capacity_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # texture check: reconstruct the largest k and show actual vs predicted + error
    if last_gs is not None:
        recon = last_gs.query_volume(volume.shape).cpu().numpy().astype(np.float32)
        D, H, W = volume.shape
        planes = [('XY', np.s_[D // 2]), ('XZ', np.s_[:, H // 2, :]), ('YZ', np.s_[:, :, W // 2])]
        fig, axes = plt.subplots(3, 3, figsize=(14, 11))
        fig.suptitle(f'Largest capacity (k={ks[-1]}, {nparams[-1]} params): '
                     f'actual vs predicted vs error', fontsize=14, fontweight='bold')
        for r, (pn, sl) in enumerate(planes):
            gt_s, rc_s = volume[sl], recon[sl]
            for c, (img, cmap, vmax, title) in enumerate([
                    (gt_s, 'magma', 1, 'Actual'), (rc_s, 'magma', 1, f'k={ks[-1]} fit'),
                    (np.abs(rc_s - gt_s), 'hot', None, '|error|')]):
                ax = axes[r, c]; im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
                if r == 0:
                    ax.set_title(title, fontsize=12, fontweight='bold')
                if c == 0:
                    ax.set_ylabel(pn, fontsize=11, fontweight='bold')
                ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f'texture_check_k{ks[-1]}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # verdict
    gain = valp[-1] - valp[0]
    print(f'\nval PSNR {valp[0]:.2f} (k={ks[0]}) -> {valp[-1]:.2f} (k={ks[-1]})  = {gain:+.2f} dB')
    print(f'Outputs -> {out_dir}')


if __name__ == '__main__':
    main()

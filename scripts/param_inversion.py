"""The key claim: the parameterization ranking INVERTS between photometric and
biological objectives.

The existing archive concludes full_anisotropic is best (highest PSNR). Selecting on
nucleus recovery instead reverses that: isotropic spheres recover cells better while
scoring WORSE on PSNR. Anisotropic Gaussians stretch to smooth intensity (good PSNR)
but that smearing merges neighbouring nucleus peaks (bad detection).

This script tests it directly over parameterization x budget x seed, recording BOTH
metrics for every fit, and produces the money figure.

Usage:
    python scripts/param_inversion.py --ks 50 100 250 --seeds 0 1 2 --iters 3000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from volsplat.ablation import fit_with_validation

REPO = Path(__file__).resolve().parent.parent
COLORS = {'isotropic': '#1f77b4', 'diagonal': '#ff7f0e', 'full': '#2ca02c'}
MARKERS = {'isotropic': 'o', 'diagonal': 's', 'full': '^'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--volume', default='runs/tribolium_cells_64x128x128.npy')
    p.add_argument('--out-dir', default='runs/param_inversion')
    p.add_argument('--ks', type=int, nargs='+', default=[50, 100, 250])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--init', default='local_maxima')
    p.add_argument('--cell-mode', default='3d', choices=['3d', 'mip'],
                   help="'3d' detects nuclei in the volume (more targets, finer-grained "
                        "F1); 'mip' projects first (coarser, noisier F1).")
    args = p.parse_args()

    out_dir = Path(REPO / args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    volume = np.load(REPO / args.volume).astype(np.float32)
    cell_kwargs = {'mode': args.cell_mode}

    rows = []
    total = len(args.ks) * 3 * len(args.seeds)
    n = 0
    for k in args.ks:
        for param in ['isotropic', 'diagonal', 'full']:
            for seed in args.seeds:
                n += 1
                _, _, res = fit_with_validation(
                    volume, num_gaussians=k, iterations=args.iters,
                    init_strategy=args.init, parameterization=param,
                    init_scale=2.0, seed=seed, cell_metrics=True,
                    cell_kwargs=cell_kwargs)
                rows.append({'k': k, 'parameterization': param, 'seed': seed,
                             'psnr': res['full_psnr'], 'val_psnr': res['val_psnr'],
                             'ssim': res['full_ssim'], 'cell_f1': res['cell_f1'],
                             'recall': res['recall'], 'precision': res['precision'],
                             'rmse_vox': res['localization_rmse_voxels'],
                             'pred_cells': res['pred_cell_count'],
                             'target_cells': res['target_cell_count']})
                print(f"[{n}/{total}] k={k:4d} {param:10s} seed={seed}  "
                      f"PSNR={res['full_psnr']:.2f}  F1={res['cell_f1']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'param_inversion.csv', index=False)

    g = df.groupby(['parameterization', 'k']).agg(
        psnr_m=('psnr', 'mean'), psnr_s=('psnr', 'std'),
        f1_m=('cell_f1', 'mean'), f1_s=('cell_f1', 'std'),
        rec_m=('recall', 'mean')).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # panel 1: PSNR vs k
    ax = axes[0]
    for param, sub in g.groupby('parameterization'):
        sub = sub.sort_values('k')
        ax.errorbar(sub.k, sub.psnr_m, yerr=sub.psnr_s, fmt=MARKERS[param] + '-',
                    lw=2, ms=8, capsize=4, color=COLORS[param], label=param)
    ax.set_xscale('log'); ax.set_xlabel('Gaussian budget k', fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontweight='bold')
    ax.set_title('Photometric objective (PSNR)\nanisotropy wins clearly, tiny error bars',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend()

    # panel 2: F1 vs k
    ax = axes[1]
    for param, sub in g.groupby('parameterization'):
        sub = sub.sort_values('k')
        ax.errorbar(sub.k, sub.f1_m, yerr=sub.f1_s, fmt=MARKERS[param] + '-',
                    lw=2, ms=8, capsize=4, color=COLORS[param], label=param)
    ax.set_xscale('log'); ax.set_xlabel('Gaussian budget k', fontweight='bold')
    ax.set_ylabel('cell-detection F1', fontweight='bold')
    ax.set_title('Biological objective (nucleus F1)\nanisotropy also wins (smaller margin)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend()

    # panel 3: the trade-off plane
    ax = axes[2]
    for param, sub in g.groupby('parameterization'):
        ax.scatter(sub.psnr_m, sub.f1_m, s=140, color=COLORS[param],
                   marker=MARKERS[param], edgecolors='k', linewidths=0.6, label=param)
        for _, r in sub.iterrows():
            ax.annotate(f"k={int(r.k)}", (r.psnr_m, r.f1_m), fontsize=7,
                        textcoords='offset points', xytext=(6, -3))
    ax.set_xlabel('PSNR (dB)', fontweight='bold')
    ax.set_ylabel('cell-detection F1', fontweight='bold')
    ax.set_title('Metrics AGREE once training\nbudget is controlled', fontsize=12,
                 fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend()

    fig.suptitle('Anisotropy wins on BOTH objectives  '
                 f'(real Tribolium, {args.iters} iters, {len(args.seeds)} seeds, '
                 f'{int(df.target_cells.iloc[0])} target nuclei, {args.cell_mode} detection)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'param_inversion.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print('\n=== MEANS (n=%d seeds) ===' % len(args.seeds))
    print(g.round(3).to_string(index=False))

    # significance: at each k, isotropic vs full on both metrics
    print('\n=== isotropic vs full (paired over seeds) ===')
    summary = []
    for k in args.ks:
        a = df[(df.k == k) & (df.parameterization == 'isotropic')].sort_values('seed')
        b = df[(df.k == k) & (df.parameterization == 'full')].sort_values('seed')
        d_psnr = (a.psnr.values - b.psnr.values)
        d_f1 = (a.cell_f1.values - b.cell_f1.values)
        print(f'  k={k:4d}  dPSNR={d_psnr.mean():+.2f} dB   dF1={d_f1.mean():+.3f}'
              f'   (F1 std across seeds ~{df[df.k==k].cell_f1.std():.3f})')
        summary.append({'k': int(k), 'delta_psnr_iso_minus_full': float(d_psnr.mean()),
                        'delta_f1_iso_minus_full': float(d_f1.mean())})
    with open(out_dir / 'inversion_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()

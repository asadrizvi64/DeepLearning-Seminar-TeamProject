"""Turn an ablation results.csv into the full set of comparison graphs.

Generates, into <ablation-dir>/plots/:
    psnr_vs_k_by_param.png       PSNR vs k, one line per parameterization (facet: dataset)
    ssim_vs_k_by_param.png       SSIM vs k, one line per parameterization
    psnr_vs_k_by_init.png        PSNR vs k, one line per initialization
    psnr_vs_params.png           efficiency frontier: PSNR vs #free-parameters
    psnr_vs_time.png             PSNR vs fit-time
    metric_bars_k{K}.png         grouped bars of every metric across parameterizations
    training_curves.png          PSNR-vs-iteration for each run (from curves/*.json)

Usage:
    python scripts/plot_ablation.py --results runs/ablation/results.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARAM_COLORS = {'isotropic': '#1f77b4', 'diagonal': '#ff7f0e', 'full': '#2ca02c'}
PARAM_MARKERS = {'isotropic': 'o', 'diagonal': 's', 'full': '^'}


def _lineplot_by(df, ax, group_col, x='num_gaussians', y='psnr', colors=None):
    for key, sub in df.groupby(group_col):
        sub = sub.sort_values(x)
        c = (colors or {}).get(key)
        m = PARAM_MARKERS.get(key, 'o')
        ax.plot(sub[x], sub[y], marker=m, lw=2, ms=7, label=str(key), color=c)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)


def plot_metric_vs_k(df, plots_dir, group_col, y, colors, fname, ylabel):
    datasets = sorted(df['dataset'].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5.5),
                             squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        _lineplot_by(df[df['dataset'] == ds], ax, group_col, y=y, colors=colors)
        ax.set_title(f'{ds}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Number of Gaussians (k)', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.legend(title=group_col.replace('_', ' '))
    fig.suptitle(f'{ylabel} vs k  (by {group_col})', fontsize=15, fontweight='bold')
    fig.tight_layout()
    out = plots_dir / fname
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out.name}')


def plot_efficiency(df, plots_dir, x, xlabel, fname):
    """PSNR vs a cost axis (params or time), colored by parameterization."""
    datasets = sorted(df['dataset'].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5.5),
                             squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        dsub = df[df['dataset'] == ds]
        for pm, sub in dsub.groupby('parameterization'):
            sub = sub.sort_values(x)
            ax.plot(sub[x], sub['psnr'], marker=PARAM_MARKERS.get(pm, 'o'),
                    lw=2, ms=7, label=pm, color=PARAM_COLORS.get(pm))
        ax.set_title(f'{ds}', fontsize=13, fontweight='bold')
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(title='parameterization')
    fig.suptitle(f'Efficiency: PSNR vs {xlabel}', fontsize=15, fontweight='bold')
    fig.tight_layout()
    out = plots_dir / fname
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out.name}')


def plot_metric_bars(df, plots_dir):
    """For the largest k available, grouped bars of every metric across params."""
    k = sorted(df['num_gaussians'].unique())[-1]
    sub = df[df['num_gaussians'] == k]
    # one panel per dataset
    metrics = [('psnr', 'PSNR (dB)', True), ('ssim', 'SSIM', True),
               ('mae', 'MAE', False), ('corr', 'Correlation', True)]
    datasets = sorted(sub['dataset'].unique())
    fig, axes = plt.subplots(len(datasets), len(metrics),
                             figsize=(4 * len(metrics), 3.5 * len(datasets)),
                             squeeze=False)
    for i, ds in enumerate(datasets):
        dsub = sub[sub['dataset'] == ds]
        for jm, (metric, label, higher_better) in enumerate(metrics):
            ax = axes[i][jm]
            g = dsub.groupby('parameterization')[metric].mean()
            order = ['isotropic', 'diagonal', 'full']
            g = g.reindex([o for o in order if o in g.index])
            bars = ax.bar(range(len(g)), g.values,
                          color=[PARAM_COLORS.get(p) for p in g.index])
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels(g.index, rotation=20)
            arrow = '↑' if higher_better else '↓'
            ax.set_title(f'{ds}: {label} {arrow}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            for b, v in zip(bars, g.values):
                ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=8)
    fig.suptitle(f'Metric comparison at k={k} (parameterization ablation)',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    out = plots_dir / f'metric_bars_k{k}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out.name}')


def plot_training_curves(curves_dir, plots_dir):
    files = sorted(Path(curves_dir).glob('*.json'))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for f in files:
        d = json.load(open(f))
        curve = np.array(d['curve'])
        if curve.size == 0:
            continue
        cfg = d['config']
        c = PARAM_COLORS.get(cfg['parameterization'])
        ax.plot(curve[:, 0], curve[:, 1], lw=1.3, alpha=0.8, color=c,
                label=f"{cfg['parameterization']} k={cfg['num_gaussians']}")
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontweight='bold')
    ax.set_title('Training curves (color = parameterization)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # de-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=8, ncol=2)
    fig.tight_layout()
    out = plots_dir / 'training_curves.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out.name}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='runs/ablation/results.csv')
    args = p.parse_args()

    results = Path(args.results)
    df = pd.read_csv(results)
    if df.empty:
        print('No results yet.')
        return
    plots_dir = results.parent / 'plots'
    plots_dir.mkdir(exist_ok=True)
    print(f'{len(df)} runs -> {plots_dir}/')

    # 1-2. metric vs k, by parameterization
    plot_metric_vs_k(df, plots_dir, 'parameterization', 'psnr',
                     PARAM_COLORS, 'psnr_vs_k_by_param.png', 'PSNR (dB)')
    plot_metric_vs_k(df, plots_dir, 'parameterization', 'ssim',
                     PARAM_COLORS, 'ssim_vs_k_by_param.png', 'SSIM')
    # 3. metric vs k, by init (only if >1 init present)
    if df['init'].nunique() > 1:
        plot_metric_vs_k(df, plots_dir, 'init', 'psnr',
                         None, 'psnr_vs_k_by_init.png', 'PSNR (dB)')
    # 4-5. efficiency frontiers
    plot_efficiency(df, plots_dir, 'num_params', '# free parameters', 'psnr_vs_params.png')
    if 'fit_seconds' in df.columns:
        plot_efficiency(df, plots_dir, 'fit_seconds', 'fit time (s)', 'psnr_vs_time.png')
    # 6. grouped metric bars at largest k
    plot_metric_bars(df, plots_dir)
    # 7. training curves
    plot_training_curves(results.parent / 'curves', plots_dir)

    print('Done.')


if __name__ == '__main__':
    main()

"""Motivating figure: in the colleague's own 19 cell-matching runs, PSNR and cell-F1
are nearly uncorrelated -- so optimizing PSNR does not optimize nucleus recovery.

Reads his run archive directly and produces:
    runs/psnr_vs_cellf1.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = REPO / 'runs 2/runs'

COLORS = {'full_anisotropic': '#2ca02c', 'axis_aligned_anisotropic': '#1f77b4',
          'learned_isotropic': '#ff7f0e', 'fixed_isotropic': '#d62728'}


def load():
    rows = []
    for f in ARCHIVE.rglob('cell_count_matching.json'):
        d = json.load(open(f))
        parts = f.relative_to(ARCHIVE).parts
        s = f.parent / 'summary.json'
        if not s.exists():
            continue
        sd = json.load(open(s))
        psnr = sd.get('metrics', {}).get('psnr_db')
        if psnr is None:
            continue
        rows.append(dict(variant=parts[1], N=int(parts[2].split('_')[1]),
                         psnr=psnr, f1=d['f1'], recall=d['recall'],
                         precision=d['precision'],
                         pred=d['pred_cell_count'], target=d['target_cell_count']))
    return pd.DataFrame(rows)


def main():
    df = load()
    valid = df[df.psnr > 10]          # exclude the degenerate fixed_isotropic fits
    r = np.corrcoef(valid.psnr, valid.f1)[0, 1]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # 1. the decorrelation
    ax = axes[0]
    for v, sub in df.groupby('variant'):
        ax.scatter(sub.psnr, sub.f1, s=90, label=v, color=COLORS.get(v),
                   edgecolors='k', linewidths=0.5, alpha=0.85)
    ax.set_xlabel('PSNR (dB)', fontweight='bold')
    ax.set_ylabel('cell-detection F1', fontweight='bold')
    ax.set_title(f'PSNR vs nucleus recovery\nPearson r = {r:.3f} (excl. degenerate fits)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # 2. budget drives them in OPPOSITE directions
    ax = axes[1]
    sub = valid.sort_values('N')
    ax.plot(sub.N, sub.psnr, 'o-', color='#d62728', label='PSNR (dB)')
    ax.set_xlabel('Gaussian budget N', fontweight='bold')
    ax.set_ylabel('PSNR (dB)', color='#d62728', fontweight='bold')
    ax2 = ax.twinx()
    ax2.plot(sub.N, sub.f1, 's--', color='#2ca02c', label='cell F1')
    ax2.set_ylabel('cell F1', color='#2ca02c', fontweight='bold')
    rn_p = np.corrcoef(valid.N, valid.psnr)[0, 1]
    rn_f = np.corrcoef(valid.N, valid.f1)[0, 1]
    ax.set_title(f'Budget pulls them apart\nr(N,PSNR)={rn_p:.2f}   r(N,F1)={rn_f:.2f}',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. variant ranking flips between the two metrics
    ax = axes[2]
    g = valid.groupby('variant')[['psnr', 'f1']].mean()
    order = [v for v in ['learned_isotropic', 'axis_aligned_anisotropic',
                         'full_anisotropic'] if v in g.index]
    g = g.loc[order]
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.psnr / g.psnr.max(), 0.4, label='PSNR (normalized)', color='#d62728')
    ax.bar(x + 0.2, g.f1 / g.f1.max(), 0.4, label='cell F1 (normalized)', color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace('_', '\n') for o in g.index], fontsize=9)
    ax.set_ylabel('normalized score', fontweight='bold')
    ax.set_title('Ranking FLIPS between metrics\n(full wins PSNR, axis-aligned wins F1)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Why optimize for nucleus recovery, not PSNR  '
                 '(source: 19 cell-matching runs from the existing archive)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = REPO / 'runs/psnr_vs_cellf1.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    print(f'\nr(PSNR, F1) = {r:.3f}   r(N,PSNR) = {rn_p:.3f}   r(N,F1) = {rn_f:.3f}')
    print(g.round(3).to_string())


if __name__ == '__main__':
    main()

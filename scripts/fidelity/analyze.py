"""Combine rate-detectability runs into the paper's main figure and tables.

Every run directory written by rate_detectability.py (meta.json + rate_detectability.csv)
is grouped by (dataset, frame). Within a group, detection is expressed RELATIVE TO THE
RAW VOLUME, so 1.0 means "compression lost nothing the detector could find":

    kept_all    = labelled nuclei found in the reconstruction / found in raw
    kept_faint  = same, for the faint half (split on RAW LoG response)
    kept_bright = same, for the bright half

The budget is the sparse-annotation 200 detections (DRO) or 1.0 x the number of true
nuclei (dense CE), where recall is also precision.

Outputs (in --out):
    rate_detectability.png   top row: faint nuclei kept vs compression; bottom row: PSNR
                             vs compression (separate axes -- never a dual axis)
    matched.csv              every Luxar fit next to the codecs matched to its bytes
    psnr_disagreement.csv    per group: how often PSNR orders two reconstructions the
                             opposite way from faint nuclei kept

Usage:
    python scripts/fidelity/analyze.py runs/fidelity/codecs_pilot runs/fidelity/codecs_ce_t150 \
        runs/fidelity/codecs_ce_t194 runs/fidelity_cluster/fidelity/pilot_* --out runs/fidelity/figures
"""
import argparse
import glob
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical slots in fixed order (validated: CVD dE >= 9.1 adjacent, normal >= 19.6).
# Three slots sit below 3:1 on white, so every line also carries a marker and a direct label.
METHODS = [  # (key, label, colour, marker)
    ('luxar', 'Luxar (Gaussian splats)', '#2a78d6', 'o'),
    ('jpegxl', 'JPEG-XL', '#eb6834', 's'),
    ('jpeg2k', 'JPEG2000', '#1baf7a', '^'),
    ('zfp', 'ZFP', '#eda100', 'D'),
    ('downsample', 'Downsampling', '#e87ba4', 'v'),
]
INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
DISAGREE_DB = 1.0            # PSNR gap that counts as "PSNR says A is better"


def family(method):
    return 'luxar' if method.startswith('luxar') else method


def load(dirs):
    frames = []
    for d in dirs:
        d = Path(d)
        if not (d / 'rate_detectability.csv').exists():
            continue
        meta = json.load(open(d / 'meta.json'))
        df = pd.read_csv(d / 'rate_detectability.csv')
        budget = '1.0x' if meta.get('dataset', 'DRO') == 'CE' else '200'
        if f'gt_log_rank@{budget}' not in df:
            continue
        raw = df[df.method == 'raw'].iloc[0]
        for part, col in (('all', ''), ('faint', '_faint'), ('bright', '_bright')):
            denom = max(int(raw[f'gt_log_rank@{budget}{col}']), 1)
            df[f'kept_{part}'] = df[f'gt_log_rank@{budget}{col}'] / denom
        df['family'] = df.method.map(family)
        df['matched'] = df.params.fillna('{}').apply(
            lambda p: json.loads(p).get('size_matched', True))
        df['group'] = f"{meta.get('dataset', 'DRO')} t{meta.get('frame', 0):03d}"
        df['n_labelled'] = meta['n_manual']
        df['source'] = d.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # the same codec at the same bytes can appear in two runs: keep one
    return out.drop_duplicates(subset=['group', 'method', 'bytes'])


def place_labels(ax, items, min_gap_frac=0.06):
    """Direct labels at line ends, nudged apart vertically so none overprint.
    items: list of (x, y, text)."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda t: t[1])
    placed = []
    for x, y, text in items:
        y_lab = y if not placed else max(y, placed[-1] + gap)
        placed.append(y_lab)
        ax.annotate(text, (x, y), xytext=(x * 1.12, y_lab), textcoords='data', fontsize=8,
                    color=INK_2, va='center',
                    arrowprops=dict(arrowstyle='-', color=GRID, lw=0.8) if abs(y_lab - y) > gap / 3 else None)


def figure(df, path):
    from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
    groups = sorted(df.group.unique())
    fig, axes = plt.subplots(2, len(groups), figsize=(4.6 * len(groups), 7.2),
                             sharex='col', squeeze=False, facecolor=SURFACE)
    for c, g in enumerate(groups):
        sub = df[(df.group == g) & (df.method != 'raw')]
        n = int(sub.n_labelled.iloc[0])
        for r, (ycol, ylab) in enumerate((('kept_faint', 'faint nuclei kept (vs raw)'),
                                           ('psnr', 'PSNR (dB)'))):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            ends = []
            for key, label, colour, marker in METHODS:
                s = sub[sub.family == key].sort_values('ratio')
                if s.empty:
                    continue
                ax.plot(s.ratio, s[ycol], color=colour, lw=2, marker=marker, ms=8,
                        mec=SURFACE, mew=1.5, label=label, zorder=3)
                last = s.iloc[-1]
                ends.append((last.ratio, last[ycol], label.split(' (')[0]))
            ax.set_xscale('log')
            ticks = [t for t in (5, 10, 20, 50, 100, 200, 500, 1000)
                     if sub.ratio.min() / 1.5 <= t <= sub.ratio.max() * 1.5]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}×'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(sub.ratio.min() / 1.3, sub.ratio.max() * 2.2)
            ax.grid(True, color=GRID, lw=0.8, zorder=0)
            for s_ in ('top', 'right'):
                ax.spines[s_].set_visible(False)
            for s_ in ('left', 'bottom'):
                ax.spines[s_].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=9)
            if r == 0:
                ax.axhline(1.0, color=INK_2, lw=1, ls=':', zorder=1)
                ax.set_ylim(0, max(1.15, float(sub[ycol].max()) + 0.05))
                ax.set_title(f'{g}  ({n} labelled nuclei)', fontsize=11, color=INK, loc='left')
            place_labels(ax, ends)
            if c == 0:
                ax.set_ylabel(ylab, color=INK, fontsize=10)
            if r == 1:
                ax.set_xlabel('compression ratio (raw bytes / stored bytes)', color=INK, fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes[0, 1:]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def matched_table(df):
    rows = []
    for g, sub in df.groupby('group'):
        for _, lx in sub[sub.family == 'luxar'].iterrows():
            same = sub[(sub.family != 'luxar') & (sub.method != 'raw') & sub.matched &
                       (np.abs(sub.bytes - lx.bytes) <= 0.10 * lx.bytes)]
            row = dict(group=g, luxar=lx.method, bytes=int(lx.bytes), ratio=round(lx.ratio, 1),
                       luxar_psnr=round(lx.psnr, 2), luxar_kept_faint=round(lx.kept_faint, 3),
                       luxar_kept_bright=round(lx.kept_bright, 3))
            for _, cd in same.iterrows():
                row[f'{cd.method}_psnr'] = round(cd.psnr, 2)
                row[f'{cd.method}_kept_faint'] = round(cd.kept_faint, 3)
            rows.append(row)
    return pd.DataFrame(rows)


def disagreement(df):
    """Pairs of reconstructions where PSNR is >= DISAGREE_DB higher for one, yet that one
    keeps FEWER faint nuclei by more than one nucleus."""
    rows = []
    for g, sub in df[df.method != 'raw'].groupby('group'):
        n_faint = max(1, int(round(sub.n_labelled.iloc[0] / 2)))
        one = 1.0 / n_faint
        pairs = inverted = 0
        for (_, a), (_, b) in itertools.combinations(sub.iterrows(), 2):
            if abs(a.psnr - b.psnr) < DISAGREE_DB:
                continue
            hi, lo = (a, b) if a.psnr > b.psnr else (b, a)
            pairs += 1
            if hi.kept_faint < lo.kept_faint - one:
                inverted += 1
        tau = sub[['psnr', 'kept_faint']].corr(method='kendall').iloc[0, 1]
        rows.append(dict(group=g, n_reconstructions=len(sub), psnr_separated_pairs=pairs,
                         psnr_wrong_about_faint=inverted,
                         share_wrong=round(inverted / pairs, 3) if pairs else np.nan,
                         kendall_tau_psnr_vs_faint=round(tau, 3)))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+')
    p.add_argument('--out', default='runs/fidelity/figures')
    args = p.parse_args()
    dirs = sorted({d for pat in args.dirs for d in glob.glob(pat)})
    df = load(dirs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figure(df, out / 'rate_detectability.png')
    m = matched_table(df)
    m.to_csv(out / 'matched.csv', index=False)
    d = disagreement(df)
    d.to_csv(out / 'psnr_disagreement.csv', index=False)
    pd.set_option('display.width', 200)
    print(f'{len(dirs)} runs, {len(df)} reconstructions, groups: {sorted(df.group.unique())}')
    if len(m):
        print('\n=== Luxar vs codecs at matched bytes ===')
        print(m.to_string(index=False))
    print('\n=== does PSNR order reconstructions the same way as faint nuclei kept? ===')
    print(d.to_string(index=False))
    print(f'\nOutputs -> {out}')


if __name__ == '__main__':
    main()

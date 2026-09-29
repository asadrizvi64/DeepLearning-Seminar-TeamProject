"""Generate every number, table and figure in paper/ from the result files.

Nothing in the paper is typed by hand: rerun this after new results arrive.

Inputs (C. elegans, frames listed in FRAMES):
    runs/fidelity/log_ce_t*.csv        LoG detector, radii 0.4/0.6/0.8 D (log_rescore.py)
    runs/fidelity/cellpose_ce_t*.csv   Cellpose detector, same radii (cellpose_score.py)
    runs/fidelity/codecs_ce_t*/rate_detectability.csv            codec PSNR
    runs/fidelity_cluster/fidelity/pilot_ce_t*/rate_detectability.csv   Luxar PSNR,
                                                                  survival, fit time
Outputs:
    paper/numbers.tex, paper/table_main.tex, paper/table_cliff.tex, paper/table_radius.tex
    paper/figures/fig_rate_detectability.pdf, paper/figures/fig_psnr_blind.pdf
"""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / 'runs'
PAPER = REPO / 'paper'
FRAMES = (150, 194)
RAD = 0.6                                  # main match radius, x nucleus diameter
METHODS = [('luxar', 'Luxar (Gaussian splats)', '#2a78d6', 'o'),
           ('jpegxl', 'JPEG-XL', '#eb6834', 's'),
           ('jpeg2k', 'JPEG2000', '#1baf7a', '^'),
           ('zfp', 'ZFP', '#eda100', 'D'),
           ('downsample', 'Downsampling', '#e87ba4', 'v')]
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'


def family(m):
    return 'luxar' if m.startswith('luxar') else m


def load_detector(kind):
    frames = []
    for t in FRAMES:
        d = pd.read_csv(RUNS / 'fidelity' / f'{kind}_ce_t{t}.csv')
        raw = d[d.method == 'raw'].iloc[0]
        for r in (0.4, 0.6, 0.8):
            d[f'kept@{r}'] = d[f'found@r{r}'] / raw[f'found@r{r}']
            d[f'kept_faint@{r}'] = d[f'found_faint@r{r}'] / raw[f'found_faint@r{r}']
        d['frame'] = t
        d['family'] = d.method.map(family)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_psnr():
    rows = []
    for t in FRAMES:
        for path in (RUNS / 'fidelity' / f'codecs_ce_t{t}' / 'rate_detectability.csv',
                     RUNS / 'fidelity_cluster' / 'fidelity' / f'pilot_ce_t{t}' / 'rate_detectability.csv'):
            d = pd.read_csv(path)
            d['frame'] = t
            rows.append(d)
    p = pd.concat(rows, ignore_index=True)
    return p.drop_duplicates(subset=['frame', 'method', 'bytes'])


def luxar_meta():
    rows = []
    for t in FRAMES:
        run = RUNS / 'fidelity_cluster' / 'fidelity' / f'pilot_ce_t{t}'
        m = json.load(open(run / 'meta.json'))
        d = pd.read_csv(run / 'rate_detectability.csv')
        for _, r in d[d.method.str.startswith('luxar')].iterrows():
            prm = json.loads(r.params)
            rows.append(dict(frame=t, method=r.method, bytes=int(r.bytes), n_true=m['n_manual'],
                             splats=prm['n_splats'], seeds=prm['seeds'], fit_min=prm['fit_s'] / 60,
                             survival=r.survival, psnr=r.psnr, ratio=r.ratio))
    return pd.DataFrame(rows)


def near(df, fam, frame, target, tol=1.25):
    s = df[(df.family == fam) & (df.frame == frame)]
    if s.empty:
        return None
    i = (np.log(s.ratio) - np.log(target)).abs().idxmin()
    r = s.loc[i]
    return r if max(r.ratio / target, target / r.ratio) <= tol else None


def pct(x):
    return f'{100 * x:.0f}\\%'


def span(vals, unit='\\%', scale=100, fmt='{:.0f}'):
    lo, hi = min(vals) * scale, max(vals) * scale
    a, b = fmt.format(lo), fmt.format(hi)
    return f'{a}{unit}' if a == b else f'{a}--{b}{unit}'


def main():
    (PAPER / 'figures').mkdir(parents=True, exist_ok=True)
    log, cp = load_detector('log'), load_detector('cellpose')
    psnr = load_psnr()
    lux = luxar_meta()
    key = f'kept@{RAD}'
    N = {}

    # ---- counts and data facts
    N['NlabA'], N['NlabB'] = str(int(lux[lux.frame == 150].n_true.iloc[0])), str(int(lux[lux.frame == 194].n_true.iloc[0]))
    N['RawKiB'] = '12.1\\,MiB'
    jxl = psnr[psnr.method == 'jpegxl']
    jmax = jxl.groupby('frame').ratio.max()
    N['JXLmax'] = f'{jmax.min():.0f}--{jmax.max():.0f}$\\times$'

    # ---- ~250x and 55-94x comparisons
    def at(df, fam, target):
        vals = [near(df, fam, t, target) for t in FRAMES]
        return [v[key] for v in vals if v is not None]
    N['LuxQuarterCP'] = pct(np.mean(at(cp, 'luxar', 250)))
    N['LuxQuarterLoG'] = pct(np.mean(at(log, 'luxar', 250)))
    N['JtkQuarterCP'] = pct(np.mean(at(cp, 'jpeg2k', 250)))
    N['JtkQuarterLoG'] = pct(np.mean(at(log, 'jpeg2k', 250)))
    mid_lux = [r[key] for t in FRAMES for r in [near(cp, 'luxar', t, 57), near(cp, 'luxar', t, 92)] if r is not None]
    mid_jxl = [r[key] for t in FRAMES for r in [near(cp, 'jpegxl', t, 62), near(cp, 'jpegxl', t, 88)] if r is not None]
    N['LuxMidCP'] = span(mid_lux)
    N['JXLMidCP'] = span([min(v, 1.0) for v in mid_jxl])
    N['LuxMidLossCP'] = span([1 - v for v in mid_lux])

    # ---- JPEG2000 faint-first loss at ~124x, t194 (LoG)
    raw194 = log[(log.frame == 194) & (log.method == 'raw')].iloc[0]
    j124 = near(log, 'jpeg2k', 194, 124)
    N['JtkFaintLoss'] = str(int(raw194[f'found_faint@r{RAD}'] - j124[f'found_faint@r{RAD}']))
    N['JtkBrightLoss'] = str(int(raw194[f'found_bright@r{RAD}'] - j124[f'found_bright@r{RAD}']))

    # ---- Luxar PSNR range, fit time
    N['LuxPSNRrange'] = f'{lux.psnr.min():.1f}--{lux.psnr.max():.1f}\\,dB'
    N['LuxFitMin'] = f'{lux.fit_min.min():.0f}--{lux.fit_min.max():.0f}\\,min'

    # ---- cliff: t194 at ~1 and ~7 splats per nucleus
    def lux_row(df, frame, seeds):
        return df[(df.frame == frame) & (df.method == f'luxar_K{seeds}')].iloc[0]
    N['CliffOneLoG'] = pct(lux_row(log, 194, 1000)[key])
    N['CliffOneCP'] = pct(lux_row(cp, 194, 1000)[key])
    N['CliffSevenLoG'] = pct(lux_row(log, 194, 4000)[key])
    N['CliffSevenCP'] = pct(lux_row(cp, 194, 4000)[key])

    # ---- survival guard
    m = lux.merge(log[['frame', 'method', key]], on=['frame', 'method'])
    rho, _ = spearmanr(m.survival, m[key])
    N['SurvRho'], N['SurvN'] = f'{rho:.2f}', str(len(m))
    guard = []
    for t in FRAMES:
        s = m[m.frame == t].sort_values('seeds')
        ok = s[s.survival >= 0.95]
        k = int(ok.seeds.iloc[0]) if len(ok) else int(s.seeds.iloc[-1])
        guard.append((lux_row(log, t, k)[key], lux_row(cp, t, k)[key]))
    N['GuardKept'] = (f'{span([g[0] for g in guard])} (LoG) and '
                      f'{span([g[1] for g in guard])} (Cellpose)')

    # ---- PSNR wrong-order share (faint kept, pairs >1 dB apart), both detectors
    def wrong_share(df):
        shares = []
        for t in FRAMES:
            s = df[(df.frame == t) & (df.method != 'raw')].merge(
                psnr[['frame', 'method', 'bytes', 'psnr']], on=['frame', 'method', 'bytes'])
            n_faint = max(1, int(s[f'found_faint@r{RAD}'].max()))
            pairs = wrong = 0
            for (_, a), (_, b) in itertools.combinations(s.iterrows(), 2):
                if abs(a.psnr - b.psnr) < 1.0:
                    continue
                hi, lo = (a, b) if a.psnr > b.psnr else (b, a)
                pairs += 1
                wrong += hi[f'kept_faint@{RAD}'] < lo[f'kept_faint@{RAD}'] - 1.0 / n_faint
            shares.append(wrong / pairs)
        return shares
    N['PSNRwrong'] = f'{span(wrong_share(log) + wrong_share(cp))}'

    with open(PAPER / 'numbers.tex', 'w') as fh:
        fh.write('% generated by scripts/fidelity/paper_numbers.py -- do not edit\n')
        for k, v in N.items():
            fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')

    # ---- Table: nuclei kept near matched ratios (mean of frames)
    regimes = [(30, '$\\approx$30$\\times$'), (60, '$\\approx$60$\\times$'), (90, '$\\approx$90$\\times$'),
               (135, '$\\approx$135$\\times$'), (250, '$\\approx$250$\\times$')]
    lines = ['\\begin{tabular}{lcccccc}', '\\toprule',
             ' & \\multicolumn{3}{c}{LoG} & \\multicolumn{3}{c}{Cellpose} \\\\',
             '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
             'Ratio & JXL & J2K & Luxar & JXL & J2K & Luxar \\\\', '\\midrule']
    for target, lab in regimes:
        cells = []
        for df in (log, cp):
            for fam in ('jpegxl', 'jpeg2k', 'luxar'):
                v = at(df, fam, target)
                cells.append(f'{np.mean(v):.2f}' if len(v) == len(FRAMES) else '--')
        lines.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (PAPER / 'table_main.tex').write_text('\n'.join(lines) + '\n')

    # ---- Table: cliff
    lines = ['\\begin{tabular}{ccrrcccc}', '\\toprule',
             'Frame & Splats & per nuc. & Ratio & PSNR & LoG & Cellpose & Surv. \\\\', '\\midrule']
    for t in FRAMES:
        for k in (1000, 2000, 4000, 16000):
            r = lux[(lux.frame == t) & (lux.seeds == k)].iloc[0]
            lines.append(f't{t} & {int(r.splats):,} & {r.splats / r.n_true:.1f} & {r.ratio:.0f}$\\times$ & '
                         f'{r.psnr:.1f} & {lux_row(log, t, k)[key]:.2f} & {lux_row(cp, t, k)[key]:.2f} & '
                         f'{r.survival:.2f} \\\\')
        if t != FRAMES[-1]:
            lines.append('\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}']
    (PAPER / 'table_cliff.tex').write_text('\n'.join(lines).replace(',', '{,}') + '\n')

    # ---- Table: radius sensitivity (t150)
    lines = ['\\begin{tabular}{lcccccc}', '\\toprule',
             ' & \\multicolumn{3}{c}{LoG} & \\multicolumn{3}{c}{Cellpose} \\\\',
             '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
             'Radius & raw & JXL & Luxar & raw & JXL & Luxar \\\\', '\\midrule']
    for r in (0.4, 0.6, 0.8):
        cells = []
        for df in (log, cp):
            raw = df[(df.frame == 150) & (df.method == 'raw')].iloc[0]
            jx = near(df, 'jpegxl', 150, 90)
            lx = near(df, 'luxar', 150, 90)
            cells += [str(int(raw[f'found@r{r}'])), f'{jx[f"kept@{r}"]:.2f}', f'{lx[f"kept@{r}"]:.2f}']
        lines.append(f'{r}$D$ & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (PAPER / 'table_radius.tex').write_text('\n'.join(lines) + '\n')

    # ---- Figure 1: rate-detectability, detector rows x frame columns
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7})
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.3), sharex=True, sharey=True)
    for r_i, (df, dname) in enumerate(((log, 'LoG'), (cp, 'Cellpose'))):
        for c_i, t in enumerate(FRAMES):
            ax = axes[r_i, c_i]
            s = df[(df.frame == t) & (df.method != 'raw')]
            for fam, lab, col, mk in METHODS:
                q = s[s.family == fam].sort_values('ratio')
                if len(q):
                    ax.plot(q.ratio, q[key], color=col, lw=1.3, marker=mk, ms=3.5, mec='white',
                            mew=0.6, label=lab.split(' (')[0])
            ax.axhline(1, color=INK_2, lw=0.6, ls=':')
            ax.set_xscale('log')
            ax.set_xticks([20, 50, 100, 200, 500])
            ax.set_xticklabels(['20', '50', '100', '200', '500'])
            ax.set_ylim(0, 1.12)
            ax.grid(True, color=GRID, lw=0.5)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if r_i == 0:
                n = int(lux[lux.frame == t].n_true.iloc[0])
                ax.set_title(f't{t} ({n} nuclei)')
            if c_i == 0:
                ax.set_ylabel(f'{dname}: nuclei kept')
            if r_i == 1:
                ax.set_xlabel('compression ratio ($\\times$)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=6, handlelength=1.2,
               columnspacing=0.8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.3)
    fig.savefig(PAPER / 'figures' / 'fig_rate_detectability.pdf')
    plt.close(fig)

    # ---- Figure 2: PSNR vs nuclei kept (Cellpose), all reconstructions
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    s = cp[cp.method != 'raw'].merge(psnr[['frame', 'method', 'bytes', 'psnr']],
                                     on=['frame', 'method', 'bytes'])
    for fam, lab, col, mk in METHODS:
        q = s[s.family == fam]
        ax.scatter(q.psnr, q[key], color=col, marker=mk, s=14, edgecolor='white', lw=0.4,
                   label=lab.split(' (')[0], zorder=3)
    ax.set_xlabel('PSNR (dB)')
    ax.set_ylabel('Cellpose: nuclei kept')
    ax.grid(True, color=GRID, lw=0.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6, ncol=1, loc='lower right', handletextpad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(PAPER / 'figures' / 'fig_psnr_blind.pdf')
    plt.close(fig)

    qualitative_figure(cp)

    for k, v in N.items():
        print(f'{k:16s} {v}')


def qualitative_figure(cp, frame=194):
    """Same region from raw, JPEG-XL and Luxar at ~85x. Titles give WHOLE-FRAME nuclei kept
    (Cellpose, 0.6 D); no per-crop counts, which would be a cherry-pickable anecdote. The
    figure shows appearance: textured nuclei vs smooth blobs with thin streak splats."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('rd', REPO / 'scripts/fidelity/rate_detectability.py')
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    ds = rd.DATASETS['CE']
    rd.configure(ds['voxel'], ds['nucleus_um'])
    crop, lo, hi = rd.load_crop(ds['root'], frame, [0, 0, 0], None, '01')
    gt = rd.gt_from_tra(ds['root'], frame, [0, 0, 0], None, '01')
    panels = [('raw volume', rd.normalise(crop, lo, hi))]
    for fam, label, target in (('jpegxl', 'JPEG-XL', 85), ('luxar', 'Luxar', 85)):
        r = near(cp[cp.frame == frame], fam, frame, target)
        rdir = 'codecs_ce_t' if fam != 'luxar' else 'luxar_render_ce_t'
        vol = np.load(RUNS / 'fidelity' / f'{rdir}{frame}' / 'recon' / r.file)['rec'].astype(np.float32)
        panels.append((f'{label} {r.ratio:.0f}$\\times$ ({100 * r[f"kept@{RAD}"]:.0f}% of nuclei kept)', vol))
    z = int(np.median(np.round(gt[:, 0])))
    y0, x0, h, w = 120, 180, 260, 340
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0))
    for ax, (title, vol) in zip(axes, panels):
        img = vol[max(0, z - 1):z + 2].max(axis=0)[y0:y0 + h, x0:x0 + w]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=7)
        ax.set_axis_off()
    fig.tight_layout(pad=0.4, w_pad=0.3)
    fig.savefig(PAPER / 'figures' / 'fig_qualitative.pdf', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


if __name__ == '__main__':
    main()

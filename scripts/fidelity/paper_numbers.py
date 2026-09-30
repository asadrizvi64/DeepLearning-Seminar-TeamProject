"""Generate every number, table and figure in paper/ from the result files.

Nothing in the paper is typed by hand: rerun this after new results arrive.
Claims follow the verdicts of scripts/fidelity/check_rules.py (paper/DECISION_RULES.md).

Frames: C. elegans (Fluo-N3DH-CE) embryo 1 (sequence 01) t150, t194 and embryo 2
(sequence 02) t150, t180; embryo 1 t100 appears only in the cliff figure.
Detectors are scored at match radius 0.6 x nucleus diameter.

Outputs: paper/numbers.tex, paper/table_main.tex, paper/table_radius.tex,
         paper/figures/fig_rate_detectability.pdf, fig_cliff.pdf, fig_psnr_blind.pdf,
         fig_qualitative.pdf
"""
import importlib.util
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
F = REPO / 'runs' / 'fidelity'
C1 = REPO / 'runs' / 'fidelity_cluster' / 'fidelity'
C2 = REPO / 'runs' / 'fidelity_cluster2' / 'fidelity'
PAPER = REPO / 'paper'
KEY = 'kept@0.6'
MAIN = [('s01', 150), ('s01', 194), ('s02', 150), ('s02', 180)]
CLIFF_ONLY = [('s01', 100)]
LABEL = {('s01', 150): 'E1 t150', ('s01', 194): 'E1 t194', ('s02', 150): 'E2 t150',
         ('s02', 180): 'E2 t180', ('s01', 100): 'E1 t100'}
DET = {  # frame -> (log csv, cellpose csv, codec dir(s) for PSNR, luxar cluster run dirs, gpu)
    ('s01', 150): ('log_ce_t150.csv', 'cellpose_ce_t150.csv', ['codecs_ce_t150'], [C1 / 'pilot_ce_t150'], 'A100'),
    ('s01', 194): ('log_ce_t194.csv', 'cellpose_ce_t194.csv', ['codecs_ce_t194'], [C1 / 'pilot_ce_t194'], 'A100'),
    ('s02', 150): ('log_ce_s02_t150.csv', 'cellpose_ce_s02_t150.csv', ['codecs_ce_s02_t150'],
                   [C2 / 'pilot_ce_s02_t150', C2 / 'pilot_ce_s02_t150_hi'], 'H100'),
    ('s02', 180): ('log_ce_s02_t180.csv', 'cellpose_ce_s02_t180.csv', ['codecs_ce_s02_t180'],
                   [C2 / 'pilot_ce_s02_t180', C2 / 'pilot_ce_s02_t180_hi'], 'H100'),
    ('s01', 100): ('log_ce_s01_t100.csv', 'cellpose_ce_s01_t100.csv', ['codecs_ce_s01_t100'],
                   [C2 / 'pilot_ce_s01_t100'], 'H100'),
}
METHODS = [('luxar', 'Luxar (Gaussian splats)', '#2a78d6', 'o'),
           ('jpegxl', 'JPEG-XL', '#eb6834', 's'),
           ('jpeg2k', 'JPEG2000', '#1baf7a', '^'),
           ('zfp', 'ZFP', '#eda100', 'D'),
           ('downsample', 'Downsampling', '#e87ba4', 'v')]
FRAME_COL = {('s01', 150): '#2a78d6', ('s01', 194): '#4a3aa7', ('s02', 150): '#eb6834',
             ('s02', 180): '#e34948', ('s01', 100): '#1baf7a'}
INK_2, GRID = '#52514e', '#e6e5e0'


def load_detector(i):
    out = []
    for fr in MAIN + CLIFF_ONLY:
        d = pd.read_csv(F / DET[fr][i])
        raw = d[d.method == 'raw'].iloc[0]
        for r in (0.4, 0.6, 0.8):
            d[f'kept@{r}'] = d[f'found@r{r}'] / raw[f'found@r{r}']
            d[f'kept_faint@{r}'] = d[f'found_faint@r{r}'] / raw[f'found_faint@r{r}']
        d['fr'] = [fr] * len(d)
        d['family'] = d.method.str.replace(r'_K\d+$', '', regex=True)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def load_luxar_and_psnr():
    lux, ps = [], []
    for fr in MAIN + CLIFF_ONLY:
        _, _, codec_dirs, runs, gpu = DET[fr]
        for cd in codec_dirs:
            d = pd.read_csv(F / cd / 'rate_detectability.csv')
            d['fr'] = [fr] * len(d)
            ps.append(d[['fr', 'method', 'bytes', 'psnr']])
        for run in runs:
            m = json.load(open(run / 'meta.json'))
            d = pd.read_csv(run / 'rate_detectability.csv')
            d['fr'] = [fr] * len(d)
            ps.append(d[['fr', 'method', 'bytes', 'psnr']])
            for _, r in d[d.method.str.startswith('luxar')].iterrows():
                p = json.loads(r.params)
                lux.append(dict(fr=fr, method=r.method, bytes=int(r.bytes), ratio=r.ratio, psnr=r.psnr,
                                n_true=m['n_manual'], splats=p['n_splats'], seeds=p['seeds'],
                                fit_min=p['fit_s'] / 60, survival=r.survival, gpu=gpu))
    psnr = pd.concat(ps, ignore_index=True).drop_duplicates(subset=['fr', 'method', 'bytes'])
    return pd.DataFrame(lux), psnr


def near(df, fam, fr, target, tol=1.25):
    s = df[(df.family == fam) & (df.fr == fr)]
    if s.empty:
        return None
    i = (np.log(s.ratio) - np.log(target)).abs().idxmin()
    r = s.loc[i]
    return r if max(r.ratio / target, target / r.ratio) <= tol else None


def pc(x):
    return f'{100 * x:.0f}'


def rng(vals, pct=True):
    vals = [100 * v for v in vals] if pct else list(vals)
    a, b = f'{min(vals):.0f}', f'{max(vals):.0f}'
    unit = '\\%' if pct else ''
    return f'{a}{unit}' if a == b else f'{a}--{b}{unit}'


def main():
    (PAPER / 'figures').mkdir(parents=True, exist_ok=True)
    log, cp = load_detector(0), load_detector(1)
    lux, psnr = load_luxar_and_psnr()
    luxm = lux[lux.fr.isin(MAIN)]
    N = {}

    # ---- data facts
    N['NlabEone'] = f"{int(luxm[luxm.fr == ('s01', 150)].n_true.iloc[0])} and {int(luxm[luxm.fr == ('s01', 194)].n_true.iloc[0])}"
    N['NlabEtwo'] = f"{int(luxm[luxm.fr == ('s02', 150)].n_true.iloc[0])} and {int(luxm[luxm.fr == ('s02', 180)].n_true.iloc[0])}"
    N['NlabTotal'] = str(int(sum(luxm.groupby('fr').n_true.first())))

    # ---- JPEG-XL: reach and worst case
    jx = pd.concat([cp[(cp.family == 'jpegxl') & cp.fr.isin(MAIN)], log[(log.family == 'jpegxl') & log.fr.isin(MAIN)]])
    jmax = jx.groupby('fr').ratio.max()
    N['JXLmax'] = f'{jmax.min():.0f}--{jmax.max():.0f}$\\times$'
    N['JXLworst'] = f'{100 * jx[KEY].min():.0f}\\%'

    # ---- JPEG2000 collapse at ~220-250x (both detectors, all frames)
    j2 = [near(df, 'jpeg2k', fr, 230) for df in (log, cp) for fr in MAIN]
    N['JtkHighRange'] = rng([r[KEY] for r in j2 if r is not None])

    # ---- Luxar's own budget K*
    ks, kl, kc, kr = [], [], [], []
    for fr in (150, 194):
        s = json.load(open(C2 / f'calibrate_ce_s01_t{fr}' / 'summary.json'))
        k = s['k_star']
        ks.append(k)
        kl.append(float(log[(log.fr == ('s01', fr)) & (log.method == f'luxar_K{k}')][KEY].iloc[0]))
        kc.append(float(cp[(cp.fr == ('s01', fr)) & (cp.method == f'luxar_K{k}')][KEY].iloc[0]))
        kr.append(float(lux[(lux.fr == ('s01', fr)) & (lux.method == f'luxar_K{k}')].ratio.iloc[0]))
    N['Kstar'] = f'{ks[0]:,}'.replace(',', '{,}')
    N['KstarRatio'] = f'{min(kr):.0f}--{max(kr):.0f}$\\times$'
    N['KstarLoG'] = rng(kl)
    N['KstarCP'] = rng(kc)

    # ---- hand-chosen high ratios: Luxar vs JPEG2000 at ~200x+ (all frames, both detectors)
    lx_hi = {d: [near(df, 'luxar', fr, 230) for fr in MAIN] for d, df in (('log', log), ('cp', cp))}
    N['LuxHighLoG'] = rng([r[KEY] for r in lx_hi['log'] if r is not None])
    N['LuxHighCP'] = rng([r[KEY] for r in lx_hi['cp'] if r is not None])
    j2h = {d: [near(df, 'jpeg2k', fr, 230) for fr in MAIN] for d, df in (('log', log), ('cp', cp))}
    N['JtkHighLoG'] = rng([r[KEY] for r in j2h['log'] if r is not None])
    N['JtkHighCP'] = rng([r[KEY] for r in j2h['cp'] if r is not None])

    # ---- the 110-135x reversal under Cellpose (embryo 2)
    rev_l = [near(cp, 'luxar', fr, 133) for fr in MAIN if fr[0] == 's02']
    rev_j = [near(cp, 'jpeg2k', fr, 110) for fr in MAIN if fr[0] == 's02']
    N['RevLuxCP'] = rng([r[KEY] for r in rev_l if r is not None])
    N['RevJtkCP'] = rng([r[KEY] for r in rev_j if r is not None])

    # ---- detector dependence: Cellpose gap to JPEG-XL at 45-94x, per frame
    gaps, lxcp, lxlog = [], [], []
    for fr in MAIN:
        a = cp[(cp.fr == fr) & (cp.family == 'jpegxl') & cp.ratio.between(45, 94)][KEY]
        b = cp[(cp.fr == fr) & (cp.family == 'luxar') & cp.ratio.between(45, 94)][KEY]
        c = log[(log.fr == fr) & (log.family == 'luxar') & log.ratio.between(45, 94)][KEY]
        if len(a) and len(b):
            gaps.append(a.clip(upper=1.0).mean() - b.mean())
            lxcp += list(b)
            lxlog += list(c)
    N['GapRange'] = f'{100 * min(gaps):.0f}--{100 * max(gaps):.0f}'
    N['GapFrames'] = str(len(gaps))
    N['LuxMidCP'] = rng(lxcp)
    N['LuxMidLoG'] = rng(lxlog)

    # Cellpose wobble between neighbouring codec sizes (max jump of a codec curve)
    wob = []
    for fr in MAIN:
        for fam in ('jpeg2k', 'jpegxl'):
            s = cp[(cp.fr == fr) & (cp.family == fam) & (cp.ratio < 150)].sort_values('ratio')[KEY].values
            if len(s) > 1:
                wob.append(np.max(np.abs(np.diff(s))))
    N['CPwobble'] = f'{100 * max(wob):.0f}'

    # ---- cliff
    lg = log.merge(lux[['fr', 'method', 'splats', 'n_true']], on=['fr', 'method'])
    lg['per_nuc'] = lg.splats / lg.n_true
    main_lg = lg[lg.fr.isin(MAIN)]
    N['CliffBelow'] = f'{main_lg[main_lg[KEY] < 0.90].per_nuc.max():.1f}'
    one = main_lg[main_lg.per_nuc < 1.5]
    N['CliffOne'] = rng(one[KEY])

    # ---- PSNR range and fit time
    N['LuxPSNRrange'] = f'{luxm.psnr.min():.1f}--{luxm.psnr.max():.1f}\\,dB'
    fa = luxm[luxm.gpu == 'A100'].fit_min
    fh = luxm[luxm.gpu == 'H100'].fit_min
    N['FitA'] = f'{fa.min():.0f}--{fa.max():.0f}\\,min'
    N['FitH'] = f'{fh.min():.0f}--{fh.max():.0f}\\,min'

    # ---- survival guard (all Luxar fits on the four main frames)
    m = luxm.merge(log[['fr', 'method', KEY]], on=['fr', 'method'])
    rho, _ = spearmanr(m.survival, m[KEY])
    N['SurvRho'], N['SurvN'] = f'{rho:.2f}', str(len(m))

    # ---- PSNR ordering errors (pairs > 1 dB apart, faint nuclei kept), both detectors
    def wrong(df):
        out = []
        for fr in MAIN:
            s = df[(df.fr == fr) & (df.method != 'raw')].merge(psnr, on=['fr', 'method', 'bytes'])
            nf = max(1, int(s['found_faint@r0.6'].max()))
            pairs = bad = 0
            for (_, a), (_, b) in itertools.combinations(s.iterrows(), 2):
                if abs(a.psnr - b.psnr) < 1.0:
                    continue
                hi, lo = (a, b) if a.psnr > b.psnr else (b, a)
                pairs += 1
                bad += hi['kept_faint@0.6'] < lo['kept_faint@0.6'] - 1.0 / nf
            out.append(bad / pairs)
        return out
    N['PSNRwrong'] = rng(wrong(log) + wrong(cp))

    with open(PAPER / 'numbers.tex', 'w') as fh_:
        fh_.write('% generated by scripts/fidelity/paper_numbers.py -- do not edit\n')
        for k, v in N.items():
            fh_.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')

    # ---- Table 1: nuclei kept near matched ratios, mean over frames with a match
    regimes = [(30, 30), (50, 50), (90, 90), (130, 130), (230, 230)]
    lines = ['\\begin{tabular}{lccc@{\\hspace{8pt}}ccc}', '\\toprule',
             ' & \\multicolumn{3}{c}{LoG} & \\multicolumn{3}{c}{Cellpose} \\\\',
             '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
             'Ratio & JXL & J2K & Luxar & JXL & J2K & Luxar \\\\', '\\midrule']
    for target, _ in regimes:
        cells = []
        for df in (log, cp):
            for fam in ('jpegxl', 'jpeg2k', 'luxar'):
                v = [near(df, fam, fr, target) for fr in MAIN]
                v = [r[KEY] for r in v if r is not None]
                cells.append('--' if not v else (f'{np.mean(v):.2f}' + ('' if len(v) == 4 else f'$^{{{len(v)}}}$')))
        lines.append(f'$\\approx${target}$\\times$ & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (PAPER / 'table_main.tex').write_text('\n'.join(lines) + '\n')

    # ---- Table 2: radius sensitivity (all four frames, raw found summed; kept at ~90x mean)
    lines = ['\\begin{tabular}{lcccccc}', '\\toprule',
             ' & \\multicolumn{3}{c}{LoG} & \\multicolumn{3}{c}{Cellpose} \\\\',
             '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
             'Radius & raw & JXL & Luxar & raw & JXL & Luxar \\\\', '\\midrule']
    for r in (0.4, 0.6, 0.8):
        cells = []
        for df in (log, cp):
            raw = sum(int(df[(df.fr == fr) & (df.method == 'raw')][f'found@r{r}'].iloc[0]) for fr in MAIN)
            jv = [near(df, 'jpegxl', fr, 90) for fr in MAIN]
            lv = [near(df, 'luxar', fr, 90) for fr in MAIN]
            cells += [str(raw), f'{np.mean([x[f"kept@{r}"] for x in jv if x is not None]):.2f}',
                      f'{np.mean([x[f"kept@{r}"] for x in lv if x is not None]):.2f}']
        lines.append(f'{r}$D$ & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (PAPER / 'table_radius.tex').write_text('\n'.join(lines) + '\n')

    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7})

    # ---- Figure: rate-detectability, detector rows x 4 frames (full width)
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.1), sharex=True, sharey=True)
    for ri, (df, dn) in enumerate(((log, 'LoG'), (cp, 'Cellpose'))):
        for ci, fr in enumerate(MAIN):
            ax = axes[ri, ci]
            s = df[(df.fr == fr) & (df.method != 'raw')]
            for fam, lab, col, mk in METHODS:
                q = s[s.family == fam].sort_values('ratio')
                if len(q):
                    ax.plot(q.ratio, q[KEY], color=col, lw=1.2, marker=mk, ms=3.2, mec='white', mew=0.5,
                            label=lab.split(' (')[0])
            ax.axhline(1, color=INK_2, lw=0.6, ls=':')
            ax.set_xscale('log')
            ax.set_xticks([20, 50, 100, 200, 500])
            ax.set_xticklabels(['20', '50', '100', '200', '500'])
            ax.set_ylim(0, 1.15)
            ax.grid(True, color=GRID, lw=0.5)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if ri == 0:
                n = int(luxm[luxm.fr == fr].n_true.iloc[0])
                ax.set_title(f'{LABEL[fr]} ({n} nuclei)')
            if ci == 0:
                ax.set_ylabel(f'{dn}: nuclei kept')
            if ri == 1:
                ax.set_xlabel('compression ratio ($\\times$)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95), pad=0.3)
    fig.savefig(PAPER / 'figures' / 'fig_rate_detectability.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    # ---- Figure: cliff -- nuclei kept vs splats per labelled nucleus
    cpm = cp.merge(lux[['fr', 'method', 'splats', 'n_true']], on=['fr', 'method'])
    cpm['per_nuc'] = cpm.splats / cpm.n_true
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.7), sharey=True)
    for ax, (df, dn) in zip(axes, ((lg, 'LoG'), (cpm, 'Cellpose'))):
        for fr in MAIN + CLIFF_ONLY:
            q = df[df.fr == fr].sort_values('per_nuc')
            ax.plot(q.per_nuc, q[KEY], color=FRAME_COL[fr], lw=1.1, marker='o', ms=2.8, mec='white', mew=0.4,
                    label=LABEL[fr], ls='--' if fr in CLIFF_ONLY else '-')
        ax.axhline(1, color=INK_2, lw=0.6, ls=':')
        ax.axhline(0.9, color=INK_2, lw=0.5, ls='-', alpha=0.4)
        ax.set_xscale('log')
        ax.set_xticks([1, 3, 10, 30, 100])
        ax.set_xticklabels(['1', '3', '10', '30', '100'])
        ax.set_xlabel('splats per labelled nucleus')
        ax.set_title(dn)
        ax.set_ylim(0.5, 1.15)
        ax.grid(True, color=GRID, lw=0.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('nuclei kept')
    axes[1].legend(frameon=False, fontsize=5.5, loc='upper left', handlelength=1.4, labelspacing=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(PAPER / 'figures' / 'fig_cliff.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    # ---- Figure: PSNR vs nuclei kept (Cellpose), four frames
    s = cp[(cp.method != 'raw') & cp.fr.isin(MAIN)].merge(psnr, on=['fr', 'method', 'bytes'])
    fig, ax = plt.subplots(figsize=(3.45, 1.8))
    for fam, lab, col, mk in METHODS:
        q = s[s.family == fam]
        ax.scatter(q.psnr, q[KEY], color=col, marker=mk, s=12, edgecolor='white', lw=0.4,
                   label=lab.split(' (')[0], zorder=3)
    ax.set_xlabel('PSNR (dB)')
    ax.set_ylabel('Cellpose: nuclei kept')
    ax.grid(True, color=GRID, lw=0.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=5.5, loc='lower right', handletextpad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(PAPER / 'figures' / 'fig_psnr_blind.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    qualitative_figure(cp)
    for k, v in N.items():
        print(f'{k:14s} {v}')


def qualitative_figure(cp, fr=('s01', 194)):
    """Same region from raw, JPEG-XL and Luxar at ~85x. Titles give WHOLE-FRAME nuclei kept
    (Cellpose, 0.6 D); per-crop counts would be a cherry-pickable anecdote."""
    spec = importlib.util.spec_from_file_location('rd', REPO / 'scripts/fidelity/rate_detectability.py')
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    ds = rd.DATASETS['CE']
    rd.configure(ds['voxel'], ds['nucleus_um'])
    crop, lo, hi = rd.load_crop(ds['root'], fr[1], [0, 0, 0], None, fr[0][1:])
    gt = rd.gt_from_tra(ds['root'], fr[1], [0, 0, 0], None, fr[0][1:])
    panels = [('raw volume', rd.normalise(crop, lo, hi))]
    for fam, label, target in (('jpegxl', 'JPEG-XL', 85), ('luxar', 'Luxar', 85)):
        r = near(cp, fam, fr, target)
        rdir = 'codecs_ce_t' if fam != 'luxar' else 'luxar_render_ce_t'
        vol = np.load(F / f'{rdir}{fr[1]}' / 'recon' / r.file)['rec'].astype(np.float32)
        panels.append((f'{label} {r.ratio:.0f}$\\times$ ({100 * r[KEY]:.0f}% of nuclei kept)', vol))
    z = int(np.median(np.round(gt[:, 0])))
    y0, x0, h, w = 120, 180, 260, 340
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0))
    for ax, (title, vol) in zip(axes, panels):
        ax.imshow(vol[max(0, z - 1):z + 2].max(axis=0)[y0:y0 + h, x0:x0 + w], cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=7)
        ax.set_axis_off()
    fig.tight_layout(pad=0.4, w_pad=0.3)
    fig.savefig(PAPER / 'figures' / 'fig_qualitative.pdf', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


if __name__ == '__main__':
    main()

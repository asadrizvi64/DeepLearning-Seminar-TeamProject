"""Generate every number, table and figure in paper/ from the result files.

Nothing in the paper is typed by hand: rerun this after new results arrive.
Claims follow the verdicts of scripts/fidelity/check_rules.py (paper/DECISION_RULES.md).

Frames: C. elegans (Fluo-N3DH-CE) embryo 1 (sequence 01) t150, t194 and embryo 2
(sequence 02) t150, t180; embryo 1 t100 appears only in the cliff figure.
Methods: Luxar, JPEG-XL and JPEG2000 (downsampling and ZFP were naive configurations and
are not compared). JPEG2000 is the corrected codec (axis fix of 2026-10-02), encoded at the
exact byte size of every Luxar fit. Detectors: LoG (blob), 3D watershed (classical
segmenter), Cellpose (learned segmenter, 2D stitched), all at match radius 0.6 D.

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
TAG = {('s01', 150): 'ce_t150', ('s01', 194): 'ce_t194', ('s02', 150): 'ce_s02_t150',
       ('s02', 180): 'ce_s02_t180', ('s01', 100): 'ce_s01_t100'}
RUNS = {  # frame -> (Luxar cluster run dirs, gpu)
    ('s01', 150): ([C1 / 'pilot_ce_t150'], 'A100'),
    ('s01', 194): ([C1 / 'pilot_ce_t194'], 'A100'),
    ('s02', 150): ([C2 / 'pilot_ce_s02_t150', C2 / 'pilot_ce_s02_t150_hi'], 'H100'),
    ('s02', 180): ([C2 / 'pilot_ce_s02_t180', C2 / 'pilot_ce_s02_t180_hi'], 'H100'),
    ('s01', 100): ([C2 / 'pilot_ce_s01_t100'], 'H100'),
}
DETS = [('log', 'LoG'), ('watershed', 'Watershed'), ('cellpose', 'Cellpose')]
METHODS = [('luxar', 'Luxar (Gaussian splats)', '#2a78d6', 'o'),
           ('jpegxl', 'JPEG-XL', '#eb6834', 's'),
           ('jpeg2k', 'JPEG2000', '#1baf7a', '^')]
FRAME_COL = {('s01', 150): '#2a78d6', ('s01', 194): '#4a3aa7', ('s02', 150): '#eb6834',
             ('s02', 180): '#e34948', ('s01', 100): '#1baf7a'}
FR_OF = {v: k for k, v in LABEL.items()}
INK_2, GRID = '#52514e', '#e6e5e0'
BINS = [(25, 35, 30), (40, 60, 50), (80, 100, 90), (120, 160, 140), (200, 270, 230),
        (300, 400, 350), (420, 560, 480)]


def load_detector(det):
    out = []
    for fr in MAIN + CLIFF_ONLY:
        d = pd.read_csv(F / f'{det}_{TAG[fr]}.csv')
        raw = d[d.method == 'raw'].iloc[0]
        for r in (0.4, 0.6, 0.8):
            d[f'kept@{r}'] = d[f'found@r{r}'] / raw[f'found@r{r}']
            d[f'kept_faint@{r}'] = d[f'found_faint@r{r}'] / raw[f'found_faint@r{r}']
        d['fr'] = [fr] * len(d)
        d['family'] = d.method.str.replace(r'_K\d+$', '', regex=True)
        out.append(d[d.family.isin(['raw'] + [m[0] for m in METHODS])])
    return pd.concat(out, ignore_index=True)


def load_luxar_and_psnr():
    lux, ps = [], []
    for fr in MAIN + CLIFF_ONLY:
        runs, gpu = RUNS[fr]
        d = pd.read_csv(F / f'codecs_{TAG[fr]}' / 'rate_detectability.csv')
        d['fr'] = [fr] * len(d)
        ps.append(d[['fr', 'method', 'bytes', 'psnr']])
        for run in runs:
            m = json.load(open(run / 'meta.json'))
            d = pd.read_csv(run / 'rate_detectability.csv')
            d = d[d.method.str.startswith('luxar')]     # codec rows there are pre-fix
            d['fr'] = [fr] * len(d)
            ps.append(d[['fr', 'method', 'bytes', 'psnr']])
            for _, r in d.iterrows():
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


def rng(vals, pct=True):
    vals = [100 * v for v in vals] if pct else list(vals)
    a, b = f'{min(vals):.0f}', f'{max(vals):.0f}'
    unit = '\\%' if pct else ''
    return f'{a}{unit}' if a == b else f'{a}--{b}{unit}'


def times(lo, hi):
    return f'{lo:.0f}$\\times$' if round(lo) == round(hi) else f'{lo:.0f}--{hi:.0f}$\\times$'


def main():
    (PAPER / 'figures').mkdir(parents=True, exist_ok=True)
    D = {k: load_detector(k) for k, _ in DETS}
    log, ws, cp = D['log'], D['watershed'], D['cellpose']
    lux, psnr = load_luxar_and_psnr()
    luxm = lux[lux.fr.isin(MAIN)]
    pairs = pd.read_csv(F / 'matched_pairs.csv')
    pairs['fr'] = pairs.frame.map(FR_OF)
    pj = pairs[(pairs.codec == 'jpeg2k') & pairs.fr.isin(MAIN)]
    N = {}

    # ---- data facts
    nl = luxm.groupby('fr').n_true.first()
    N['NlabEone'] = f"{nl[('s01', 150)]} and {nl[('s01', 194)]}"
    N['NlabEtwo'] = f"{nl[('s02', 150)]} and {nl[('s02', 180)]}"
    N['NlabTotal'] = f'{int(nl.sum()):,}'.replace(',', '{,}')

    # ---- raw-volume recall of each detector (sum over the four frames)
    gt_total = int(nl.sum())
    for det, name in DETS:
        found = sum(int(D[det][(D[det].fr == fr) & (D[det].method == 'raw')]['found@r0.6'].iloc[0]) for fr in MAIN)
        N[f'Recall{name}'] = f'{100 * found / gt_total:.0f}\\%'

    # ---- JPEG-XL: reach and worst case (all three detectors)
    jx = pd.concat([d[(d.family == 'jpegxl') & d.fr.isin(MAIN)] for d in D.values()])
    jmax = jx.groupby('fr').ratio.max()
    N['JXLmax'] = times(jmax.min(), jmax.max())
    N['JXLworst'] = f'{100 * jx[KEY].min():.0f}\\%'

    # ---- JPEG2000 far out (>= 300x): kept per detector, vs Luxar at the same bytes
    far = pj[pj.ratio >= 300]
    N['FarRatio'] = times(far.ratio.min(), far.ratio.max())
    for det, name in DETS:
        f = far[far.detector == det]
        N[f'FarJtk{name}'] = rng(f.kept_codec)
        N[f'FarLux{name}'] = rng(f.kept_lux)
    N['JtkMaxRatio'] = f"{pj.ratio.max():.0f}$\\times$"

    # ---- rule 5: Luxar minus JPEG2000 at the same bytes, CI over nuclei
    for det, name in DETS:
        g = pj[pj.detector == det]
        N[f'PairsN{name}'] = str(len(g))
        N[f'PairsJ{name}'] = str(int((g.hi < 0).sum()))
        N[f'PairsL{name}'] = str(int((g.lo > 0).sum()))
    adv = pj[(pj.detector == 'log') & (pj.lo > 0)]
    if len(adv):
        N['LuxAdvPts'] = f"{100 * adv['diff'].min():.0f}--{100 * adv['diff'].max():.0f}"
        N['LuxAdvRatio'] = times(adv.ratio.min(), adv.ratio.max())
        N['LuxAdvFrames'] = ', '.join(sorted(adv.frame.unique()))
    early = pairs[(pairs.codec == 'jpeg2k') & (pairs.fr == ('s01', 100)) & (pairs.detector == 'log') & (pairs.lo > 0)]
    if len(early):
        N['EarlyAdvPts'] = f"{100 * early['diff'].min():.0f}--{100 * early['diff'].max():.0f}"
        N['EarlyAdvRatio'] = times(early.ratio.min(), early.ratio.max())
    for det, name in (('watershed', 'Watershed'), ('cellpose', 'Cellpose')):
        e = pairs[(pairs.codec == 'jpeg2k') & (pairs.fr == ('s01', 100)) & (pairs.detector == det)]
        N[f'EarlyL{name}'] = str(int((e.lo > 0).sum()))
    N['PSNRjtkWins'] =f"{int((pj.drop_duplicates(['frame', 'method']).psnr_codec > pj.drop_duplicates(['frame', 'method']).psnr_lux).sum())} of {pj.drop_duplicates(['frame', 'method']).shape[0]}"

    # ---- Luxar's own budget K*
    ks, kr, kv = [], [], {n: [] for _, n in DETS}
    kj = []
    for fr in (150, 194):
        s = json.load(open(C2 / f'calibrate_ce_s01_t{fr}' / 'summary.json'))
        k = s['k_star']
        ks.append(k)
        kr.append(float(lux[(lux.fr == ('s01', fr)) & (lux.method == f'luxar_K{k}')].ratio.iloc[0]))
        for det, name in DETS:
            kv[name].append(float(D[det][(D[det].fr == ('s01', fr)) & (D[det].method == f'luxar_K{k}')][KEY].iloc[0]))
        kj += list(pj[(pj.fr == ('s01', fr)) & (pj.method == f'luxar_K{k}')].kept_codec)
    N['Kstar'] = f'{ks[0]:,}'.replace(',', '{,}')
    N['KstarRatio'] = times(min(kr), max(kr))
    for _, name in DETS:
        N[f'Kstar{name}'] = rng(kv[name])
    N['KstarJtk'] = rng(kj)

    # ---- detector dependence: JPEG-XL minus Luxar at 45-94x, per frame, per segmenter
    for det, name in (('cellpose', 'CP'), ('watershed', 'WS')):
        df = D[det]
        gaps, lx = [], []
        for fr in MAIN:
            a = df[(df.fr == fr) & (df.family == 'jpegxl') & df.ratio.between(45, 94)][KEY]
            b = df[(df.fr == fr) & (df.family == 'luxar') & df.ratio.between(45, 94)][KEY]
            if len(a) and len(b):
                gaps.append(a.clip(upper=1.0).mean() - b.mean())
                lx += list(b)
        N[f'Gap{name}'] = f'{100 * min(gaps):.0f}--{100 * max(gaps):.0f}'
        N[f'GapFrames{name}'] = str(len(gaps))
        N[f'LuxMid{name}'] = rng(lx)
    N['LuxMidLoG'] = rng(log[log.fr.isin(MAIN) & (log.family == 'luxar') & log.ratio.between(45, 94)][KEY])

    # Cellpose wobble between neighbouring codec sizes (max jump of a codec curve below 150x)
    wob = []
    for fr in MAIN:
        for fam in ('jpeg2k', 'jpegxl'):
            s = cp[(cp.fr == fr) & (cp.family == fam) & (cp.ratio < 150)].sort_values('ratio')[KEY].values
            if len(s) > 1:
                wob.append(np.max(np.abs(np.diff(s))))
    N['CPwobble'] = f'{100 * max(wob):.0f}'

    # ---- run-to-run spread: three identical Luxar fits of E1 t194 (original + 2 repeats)
    for det, name in DETS:
        base = D[det][(D[det].fr == ('s01', 194))]
        spread = []
        for k in (2000, 4000, 16000):
            v = [float(base[base.method == f'luxar_K{k}'][KEY].iloc[0])]
            for rep in ('rep1', 'rep2'):
                d = pd.read_csv(F / f'{det}_ce_t194_{rep}.csv')
                raw = d[d.method == 'raw'].iloc[0]['found@r0.6']
                v.append(float(d[d.method == f'luxar_K{k}'].iloc[0]['found@r0.6'] / raw))
            spread.append(max(v) - min(v))
        N[f'Rep{name}'] = f'{100 * max(spread):.1f}'

    # ---- cliff (LoG)
    def per_nuc(df):
        m = df.merge(lux[['fr', 'method', 'splats', 'n_true']], on=['fr', 'method'])
        m['per_nuc'] = m.splats / m.n_true
        return m
    lg = per_nuc(log)
    main_lg = lg[lg.fr.isin(MAIN)]
    N['CliffBelow'] = f'{main_lg[main_lg[KEY] < 0.90].per_nuc.max():.1f}'
    N['CliffOne'] = rng(main_lg[main_lg.per_nuc < 1.5][KEY])

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

    # ---- PSNR ordering errors (pairs > 1 dB apart, faint nuclei kept), all detectors
    def wrong(df):
        out = []
        for fr in MAIN:
            s = df[(df.fr == fr) & (df.method != 'raw')].merge(psnr, on=['fr', 'method', 'bytes'])
            nf = max(1, int(s['found_faint@r0.6'].max()))
            n_pairs = bad = 0
            for (_, a), (_, b) in itertools.combinations(s.iterrows(), 2):
                if abs(a.psnr - b.psnr) < 1.0:
                    continue
                hi, lo = (a, b) if a.psnr > b.psnr else (b, a)
                n_pairs += 1
                bad += hi['kept_faint@0.6'] < lo['kept_faint@0.6'] - 1.0 / nf
            out.append(bad / n_pairs)
        return out
    N['PSNRwrong'] = rng(sum((wrong(d) for d in D.values()), []))

    with open(PAPER / 'numbers.tex', 'w') as fh_:
        fh_.write('% generated by scripts/fidelity/paper_numbers.py -- do not edit\n')
        for k, v in N.items():
            fh_.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')

    # ---- Table 1: Luxar minus JPEG2000 at the same bytes, by ratio, per detector
    lines = ['\\begin{tabular}{rrrrrr}', '\\toprule',
             'Ratio & Splats/nuc. & $\\Delta$PSNR & LoG & Watershed & Cellpose \\\\', '\\midrule']
    for lo_r, hi_r, lab in BINS:
        b = pj[pj.ratio.between(lo_r, hi_r)]
        if b.empty:
            continue
        one = b.drop_duplicates(['frame', 'method'])
        spn = one.splats_per_nucleus
        cells = [f'$\\approx${lab}$\\times$', f'{spn.min():.0f}--{spn.max():.0f}' if spn.max() >= 1.5 else f'{spn.min():.1f}--{spn.max():.1f}',
                 f'{(one.psnr_lux - one.psnr_codec).mean():+.1f}'.replace('-', '$-$')]
        for det, _ in DETS:
            g = b[b.detector == det]
            if g.empty:
                cells.append('--')
                continue
            v = f'{100 * g["diff"].mean():+.1f}'.replace('-', '$-$')
            if (g.hi < 0).all() or (g.lo > 0).all():
                v = f'\\textbf{{{v}}}'
            nf = g.frame.nunique()
            cells.append(v + ('' if nf == 4 else f'$^{{{nf}}}$'))
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (PAPER / 'table_main.tex').write_text('\n'.join(lines) + '\n')

    # ---- Table 2: radius sensitivity (raw found summed over frames; kept at ~90x mean)
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

    def style(ax):
        ax.grid(True, color=GRID, lw=0.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    # ---- Figure: rate-detectability, detector rows x 4 frames (full width)
    fig, axes = plt.subplots(3, 4, figsize=(7.1, 4.3), sharex=True, sharey=True)
    for ri, (det, dn) in enumerate(DETS):
        df = D[det]
        for ci, fr in enumerate(MAIN):
            ax = axes[ri, ci]
            s = df[(df.fr == fr) & (df.method != 'raw')]
            for fam, lab, col, mk in METHODS:
                q = s[s.family == fam].sort_values('ratio')
                if len(q):
                    ax.plot(q.ratio, q[KEY], color=col, lw=1.2, marker=mk, ms=3.0, mec='white', mew=0.5,
                            label=lab.split(' (')[0])
            ax.axhline(1, color=INK_2, lw=0.6, ls=':')
            ax.set_xscale('log')
            ax.set_xticks([20, 50, 100, 200, 500])
            ax.set_xticklabels(['20', '50', '100', '200', '500'])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_ylim(0.4, 1.12)
            style(ax)
            if ri == 0:
                ax.set_title(f'{LABEL[fr]} ({int(nl[fr])} nuclei)')
            if ci == 0:
                ax.set_ylabel(f'{dn}: nuclei kept')
            if ri == len(DETS) - 1:
                ax.set_xlabel('compression ratio ($\\times$)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.965), pad=0.3)
    fig.savefig(PAPER / 'figures' / 'fig_rate_detectability.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    # ---- Figure: cliff -- nuclei kept vs splats per labelled nucleus, per detector
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.55), sharey=True)
    for ax, (det, dn) in zip(axes, DETS):
        df = per_nuc(D[det])
        for fr in MAIN + CLIFF_ONLY:
            q = df[df.fr == fr].sort_values('per_nuc')
            ax.plot(q.per_nuc, q[KEY], color=FRAME_COL[fr], lw=1.0, marker='o', ms=2.4, mec='white', mew=0.3,
                    label=LABEL[fr], ls='--' if fr in CLIFF_ONLY else '-')
        ax.axhline(1, color=INK_2, lw=0.6, ls=':')
        ax.axhline(0.9, color=INK_2, lw=0.5, ls='-', alpha=0.4)
        ax.set_xscale('log')
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(['1', '10', '100'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(dn)
        ax.set_ylim(0.5, 1.12)
        style(ax)
    axes[1].set_xlabel('splats per labelled nucleus')
    axes[0].set_ylabel('nuclei kept')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=5.5, handlelength=1.4,
               columnspacing=0.8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.3, w_pad=0.2, rect=(0, 0.08, 1, 1))
    fig.savefig(PAPER / 'figures' / 'fig_cliff.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    # ---- Figure: PSNR vs nuclei kept (Cellpose), four frames
    s = cp[(cp.method != 'raw') & cp.fr.isin(MAIN)].merge(psnr, on=['fr', 'method', 'bytes'])
    fig, ax = plt.subplots(figsize=(3.45, 1.7))
    for fam, lab, col, mk in METHODS:
        q = s[s.family == fam]
        ax.scatter(q.psnr, q[KEY], color=col, marker=mk, s=12, edgecolor='white', lw=0.4,
                   label=lab.split(' (')[0], zorder=3)
    ax.set_xlabel('PSNR (dB)')
    ax.set_ylabel('Cellpose: nuclei kept')
    style(ax)
    ax.legend(frameon=False, fontsize=5.5, loc='lower right', handletextpad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(PAPER / 'figures' / 'fig_psnr_blind.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    qualitative_figure(cp)
    for k, v in N.items():
        print(f'{k:16s} {v}')

    # ---- claims the TEXT makes in words; fail loudly if the data stop supporting them
    claims = {
        'abstract/discussion: segmenters never favour Luxar at identical bytes':
            N['PairsLWatershed'] == '0' and N['PairsLCellpose'] == '0',
        'results: segmenters never favour Luxar on E1 t100':
            N.get('EarlyLWatershed') == '0' and N.get('EarlyLCellpose') == '0',
        'abstract: JPEG2000 has the higher PSNR at every matched size':
            N['PSNRjtkWins'].split(' of ')[0] == N['PSNRjtkWins'].split(' of ')[1],
        'results: blob-detector advantage exists (LuxAdv*, EarlyAdv* defined)':
            'LuxAdvPts' in N and 'EarlyAdvPts' in N,
        'limitations: embryo-2 calibration recommends the same K* as embryo 1':
            all(json.load(open(REPO / 'runs' / 'fidelity_cluster3' / 'fidelity' / f'calibrate_ce_s02_t{t}'
                               / 'summary.json'))['k_star'] == ks[0] for t in (150, 180)),
    }
    bad = [c for c, ok in claims.items() if not ok]
    if bad:
        raise SystemExit('TEXT CLAIMS NO LONGER SUPPORTED -- rewrite paper/main.tex:\n  ' + '\n  '.join(bad))
    print(f'all {len(claims)} worded claims supported')


def qualitative_figure(cp, fr=('s01', 194)):
    """Same region from raw, JPEG-XL, JPEG2000 and Luxar at ~90x. Titles give WHOLE-FRAME
    nuclei kept (Cellpose, 0.6 D); per-crop counts would be a cherry-pickable anecdote.
    JPEG2000 is the encoding at exactly the Luxar fit's bytes."""
    spec = importlib.util.spec_from_file_location('rd', REPO / 'scripts/fidelity/rate_detectability.py')
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    ds = rd.DATASETS['CE']
    rd.configure(ds['voxel'], ds['nucleus_um'])
    crop, lo, hi = rd.load_crop(ds['root'], fr[1], [0, 0, 0], None, fr[0][1:])
    gt = rd.gt_from_tra(ds['root'], fr[1], [0, 0, 0], None, fr[0][1:])
    panels = [('raw volume', rd.normalise(crop, lo, hi))]
    lx = near(cp, 'luxar', fr, 90)
    j2 = cp[(cp.fr == fr) & (cp.family == 'jpeg2k')]
    j2 = j2.loc[(j2.bytes / lx.bytes - 1).abs().idxmin()]
    for fam, label, r in (('jpegxl', 'JPEG-XL', near(cp, 'jpegxl', fr, 90)), ('jpeg2k', 'JPEG2000', j2),
                          ('luxar', 'Luxar', lx)):
        rdir = 'codecs_' if fam != 'luxar' else 'luxar_render_'
        vol = np.load(F / f'{rdir}{TAG[fr]}' / 'recon' / r.file)['rec'].astype(np.float32)
        panels.append((f'{label} {r.ratio:.0f}$\\times$ ({100 * r[KEY]:.0f}% kept)', vol))
    z = int(np.median(np.round(gt[:, 0])))
    y0, x0, h, w = 120, 180, 260, 340
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.6))
    for ax, (title, vol) in zip(axes, panels):
        ax.imshow(vol[max(0, z - 1):z + 2].max(axis=0)[y0:y0 + h, x0:x0 + w], cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=6.5)
        ax.set_axis_off()
    fig.tight_layout(pad=0.3, w_pad=0.2)
    fig.savefig(PAPER / 'figures' / 'fig_qualitative.pdf', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


if __name__ == '__main__':
    main()

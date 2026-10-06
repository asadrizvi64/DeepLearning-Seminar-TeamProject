"""Generate every number, table and figure in paper/ from the result files.

Nothing in the paper is typed by hand: rerun this after new results arrive.
Claims follow the verdicts of scripts/fidelity/check_rules.py and a1_crossings.py
(paper/DECISION_RULES.md); worded claims are asserted at the end and stop the build if the
data no longer support them.

Frames: C. elegans (Fluo-N3DH-CE), 12 frames from two embryos -- the 4 paper frames
(E1 t150, t194; E2 t150, t180) and the 8 A1 robustness frames (E1/E2 t110-t185); E1 t100
(larger early nuclei) is reported separately. Methods: Luxar, JPEG-XL, JPEG2000 (corrected
codec, encoded at the exact byte size of every Luxar fit). Detectors, all at 0.6 D: LoG
(blob), 3D watershed (classical segmenter), Cellpose 2D-stitched and Cellpose 3D mode.

Outputs: paper/numbers.tex, paper/table_main.tex, paper/table_radius.tex,
         paper/figures/fig_matched.pdf, fig_cliff.pdf, fig_qualitative.pdf,
         fig_rate_detectability.pdf (supplementary grid, 4 paper frames)
"""
import importlib.util
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
F = REPO / 'runs' / 'fidelity'
C1 = REPO / 'runs' / 'fidelity_cluster' / 'fidelity'
C2 = REPO / 'runs' / 'fidelity_cluster2' / 'fidelity'
C3 = REPO / 'runs' / 'fidelity_cluster3' / 'fidelity'      # embryo-2 calibration, repeats
CA1 = REPO / 'runs' / 'fidelity_a1' / 'fidelity'          # A1 fits, embryo-2 K=64000 fits
PAPER = REPO / 'paper'
KEY = 'kept@0.6'
MAIN = [('s01', 150), ('s01', 194), ('s02', 150), ('s02', 180)]
A1 = [('s01', 110), ('s01', 130), ('s01', 170), ('s01', 185),
      ('s02', 110), ('s02', 130), ('s02', 165), ('s02', 185)]
ALL = MAIN + A1
EARLY = [('s01', 100)]
LABEL = {fr: f'E{int(fr[0][1:])} t{fr[1]}' for fr in ALL + EARLY}
TAG = {('s01', 150): 'ce_t150', ('s01', 194): 'ce_t194', ('s02', 150): 'ce_s02_t150',
       ('s02', 180): 'ce_s02_t180', ('s01', 100): 'ce_s01_t100'}
TAG.update({fr: f'ce_{fr[0]}_t{fr[1]}' for fr in A1})
RUNS = {  # frame -> (Luxar cluster run dirs, gpu)
    ('s01', 150): ([C1 / 'pilot_ce_t150'], 'A100'),
    ('s01', 194): ([C1 / 'pilot_ce_t194'], 'A100'),
    ('s02', 150): ([C2 / 'pilot_ce_s02_t150', C2 / 'pilot_ce_s02_t150_hi', CA1 / 'pilot_ce_s02_t150_k64'], 'H100'),
    ('s02', 180): ([C2 / 'pilot_ce_s02_t180', C2 / 'pilot_ce_s02_t180_hi', CA1 / 'pilot_ce_s02_t180_k64'], 'H100'),
    ('s01', 100): ([C2 / 'pilot_ce_s01_t100'], 'H100'),
}
RUNS.update({fr: ([CA1 / f'pilot_ce_{fr[0]}_t{fr[1]}'], 'H100') for fr in A1})
# (csv prefix, label in figures, macro key)
DETS = [('log', 'LoG', 'LoG'), ('watershed', 'Watershed', 'WS'),
        ('cellpose', 'Cellpose (2D)', 'CP'), ('cellpose3d', 'Cellpose (3D)', 'CPthree')]
METHODS = [('luxar', 'Luxar (Gaussian splats)', '#2a78d6', 'o'),
           ('jpegxl', 'JPEG-XL', '#eb6834', 's'),
           ('jpeg2k', 'JPEG2000', '#1baf7a', '^')]
EMB_COL = {'s01': '#2a78d6', 's02': '#e34948'}
FR_OF = {v: k for k, v in LABEL.items()}
INK_2, GRID = '#52514e', '#e6e5e0'
BINS = [(25, 35, 30), (40, 60, 50), (80, 100, 90), (120, 160, 140), (200, 270, 230),
        (300, 400, 350), (420, 560, 480)]
FAR = 300                                   # "beyond ~300x"


def load_detector(det):
    out = []
    for fr in ALL + EARLY:
        p = F / f'{det}_{TAG[fr]}.csv'
        if not p.exists():
            continue
        d = pd.read_csv(p)
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
    for fr in ALL + EARLY:
        runs, gpu = RUNS[fr]
        d = pd.read_csv(F / f'codecs_{TAG[fr]}' / 'rate_detectability.csv')
        d['fr'] = [fr] * len(d)
        ps.append(d[['fr', 'method', 'bytes', 'psnr']])
        for run in runs:
            if not (run / 'rate_detectability.csv').exists():
                continue
            m = json.load(open(run / 'meta.json'))
            d = pd.read_csv(run / 'rate_detectability.csv')
            d = d[d.method.str.startswith('luxar')]     # codec rows there are not used
            d['fr'] = [fr] * len(d)
            ps.append(d[['fr', 'method', 'bytes', 'psnr']])
            for _, r in d.iterrows():
                p = json.loads(r.params)
                lux.append(dict(fr=fr, method=r.method, bytes=int(r.bytes), ratio=r.ratio, psnr=r.psnr,
                                n_true=m['n_manual'], splats=p['n_splats'], seeds=p['seeds'],
                                fit_min=p['fit_s'] / 60, survival=r.survival, gpu=gpu))
    psnr = pd.concat(ps, ignore_index=True).drop_duplicates(subset=['fr', 'method', 'bytes'])
    return pd.DataFrame(lux).drop_duplicates(['fr', 'method']), psnr


def near(df, fam, fr, target, tol=1.25):
    s = df[(df.family == fam) & (df.fr == fr)]
    if s.empty:
        return None
    i = (np.log(s.ratio) - np.log(target)).abs().idxmin()
    r = s.loc[i]
    return r if max(r.ratio / target, target / r.ratio) <= tol else None


def rng(vals, pct=True, nd=0):
    vals = [100 * v for v in vals] if pct else list(vals)
    a, b = f'{min(vals):.{nd}f}', f'{max(vals):.{nd}f}'
    unit = '\\%' if pct else ''
    return f'{a}{unit}' if a == b else f'{a}--{b}{unit}'


def pts(vals):
    """signed range in points, e.g. '$+$2--$+$6' or '$-$4--$-$18'."""
    a, b = sorted([100 * min(vals), 100 * max(vals)])
    f = lambda x: f'{x:+.0f}'.replace('-', '$-$').replace('+', '$+$')
    return f(a) if round(a) == round(b) else f'{f(a)} to {f(b)}'


def times(lo, hi):
    return f'{lo:.0f}$\\times$' if round(lo) == round(hi) else f'{lo:.0f}--{hi:.0f}$\\times$'


def main():
    (PAPER / 'figures').mkdir(parents=True, exist_ok=True)
    D = {k: load_detector(k) for k, _, _ in DETS}
    log, cp = D['log'], D['cellpose']
    lux, psnr = load_luxar_and_psnr()
    luxa = lux[lux.fr.isin(ALL)]
    pairs = pd.read_csv(F / 'matched_pairs.csv')
    pairs['fr'] = pairs.frame.map(FR_OF)
    pj = pairs[(pairs.codec == 'jpeg2k') & pairs.fr.isin(ALL)]
    N = {}

    # ---- data facts
    nl = luxa.groupby('fr').n_true.first()
    N['NFrames'] = str(len(nl))
    N['NlabTotal'] = f'{int(nl.sum()):,}'.replace(',', '{,}')
    N['NlabMin'], N['NlabMax'] = str(int(nl.min())), str(int(nl.max()))
    N['NFits'] = str(len(luxa))
    N['NPairs'] = str(len(pj[pj.detector == 'log']))
    for det, _, key in DETS:
        frs = [fr for fr in ALL if ((D[det].fr == fr) & (D[det].method == 'raw')).any()]
        found = sum(int(D[det][(D[det].fr == fr) & (D[det].method == 'raw')]['found@r0.6'].iloc[0]) for fr in frs)
        N[f'Recall{key}'] = f'{100 * found / nl[frs].sum():.0f}\\%'
        N[f'Frames{key}'] = str(len(frs))

    # ---- JPEG-XL: reach and worst case (all frames, all detectors)
    jx = pd.concat([d[(d.family == 'jpegxl') & d.fr.isin(ALL)] for d in D.values()])
    jmax = jx.groupby('fr').ratio.max()
    N['JXLmax'] = times(jmax.min(), jmax.max())
    N['JXLworst'] = f'{100 * jx[KEY].min():.0f}\\%'

    # ---- Luxar minus JPEG2000 at identical bytes, per detector: moderate vs far ratios
    N['FarRatio'] = times(pj[pj.ratio >= FAR].ratio.min(), pj.ratio.max())
    N['JtkMaxRatio'] = f"{pj.ratio.max():.0f}$\\times$"
    for det, _, key in DETS:
        g = pj[pj.detector == det]
        mod, far = g[g.ratio < FAR], g[g.ratio >= FAR]
        N[f'PairsN{key}'] = str(len(g))
        N[f'PairsJ{key}'] = str(int((g.hi < 0).sum()))
        N[f'PairsL{key}'] = str(int((g.lo > 0).sum()))
        N[f'ModN{key}'], N[f'ModJ{key}'], N[f'ModL{key}'] = str(len(mod)), str(int((mod.hi < 0).sum())), str(int((mod.lo > 0).sum()))
        N[f'ModDiff{key}'] = f"{100 * mod['diff'].mean():+.1f}".replace('-', '$-$').replace('+', '$+$')
        N[f'FarDiff{key}'] = f"{100 * far['diff'].mean():+.1f}".replace('-', '$-$').replace('+', '$+$')
        N[f'FarJ{key}'] = f"{int((far.hi < 0).sum())} of {len(far)}"
        N[f'FarJtk{key}'] = rng(far.kept_codec)
        N[f'FarLux{key}'] = rng(far.kept_lux)
    one = pj.drop_duplicates(['frame', 'method'])
    N['PSNRjtkWins'] = f'{int((one.psnr_codec > one.psnr_lux).sum())} of {len(one)}'

    # ---- frame-level analysis: frames as independent units (the broad claims rest on this,
    # not on per-pair intervals). Per frame and regime: mean Luxar - JPEG2000 over its pairs;
    # cluster bootstrap over frames (B=10000) for the CI; two-sided Wilcoxon signed-rank over
    # frames, Holm-adjusted across the 8 detector x regime tests. PSNR row is descriptive.
    from scipy.stats import wilcoxon
    rng_ = np.random.default_rng(0)
    FL = []
    for det, label, key in [('psnr', 'PSNR (dB)', 'PSNR')] + DETS:
        for reg, sel in (('mod', pj.ratio < FAR), ('far', pj.ratio >= FAR)):
            g = pj[sel & (pj.detector == ('log' if det == 'psnr' else det))]
            if det == 'psnr':
                m = g.groupby('frame').apply(lambda x: (x.psnr_lux - x.psnr_codec).mean())
            else:
                m = 100 * g.groupby('frame')['diff'].mean()
            boot = [rng_.choice(m.values, len(m)).mean() for _ in range(10000)]
            FL.append(dict(det=det, label=label, key=key, reg=reg, n=len(m), mean=m.mean(),
                           lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                           pos=int((m > 0).sum()), neg=int((m < 0).sum()),
                           p=wilcoxon(m.values).pvalue if det != 'psnr' else np.nan,
                           e1=m[m.index.str.startswith('E1')].mean(), e2=m[m.index.str.startswith('E2')].mean()))
    FL = pd.DataFrame(FL)
    t = FL[FL.det != 'psnr'].sort_values('p')
    run = 0.0
    for i, (idx, r) in enumerate(t.iterrows()):                       # Holm step-down
        run = max(run, min(1.0, (len(t) - i) * r.p))
        FL.loc[idx, 'p_holm'] = run
    FL.to_csv(F / 'frame_level.csv', index=False)
    fmt = lambda x: f'{x:+.1f}'.replace('-', '$-$').replace('+', '$+$')
    for _, r in FL.iterrows():
        k = f"Fr{'Mod' if r.reg == 'mod' else 'Far'}{r.key}"
        N[k] = fmt(r['mean'])
        N[k + 'CI'] = f"{fmt(r.lo)} to {fmt(r.hi)}"
        N[k + 'Pos'], N[k + 'Neg'] = str(r.pos), str(r.neg)
        if r.det != 'psnr':
            N[k + 'P'] = f'{r.p_holm:.3f}' if r.p_holm >= 0.001 else '$<$0.001'
            N[k + 'EOne'], N[k + 'ETwo'] = fmt(r.e1), fmt(r.e2)
    N['FrN'] = str(int(FL.n.max()))
    farj = FL[(FL.reg == 'far') & (FL.det != 'psnr')].neg
    N['FrFarNeg'] = f'{farj.min()}--{farj.max()}' if farj.min() != farj.max() else str(farj.min())
    N['DPSNR'] = f"{(one.psnr_codec - one.psnr_lux).min():.1f}--{(one.psnr_codec - one.psnr_lux).max():.1f}\\,dB"

    # ---- E1 t100 (larger, early nuclei): reported separately
    early = pairs[(pairs.codec == 'jpeg2k') & (pairs.fr == ('s01', 100)) & (pairs.detector == 'log') & (pairs.lo > 0)]
    if len(early):
        N['EarlyAdvPts'] = f"{100 * early['diff'].min():.0f}--{100 * early['diff'].max():.0f}"
        N['EarlyAdvRatio'] = times(early.ratio.min(), early.ratio.max())

    # ---- Luxar's own budget K*: scored on every frame with a full-data fit at K*
    ks, kr, kv, kj, k_frames = [], [], {k: [] for *_, k in DETS}, [], []
    for fr in MAIN:
        cal = (C2 if fr[0] == 's01' else C3) / f'calibrate_ce_{fr[0]}_t{fr[1]}' / 'summary.json'
        if not cal.exists():
            continue
        k = json.load(open(cal))['k_star']
        ks.append(k)
        m = f'luxar_K{k}'
        if not ((lux.fr == fr) & (lux.method == m)).any():
            continue
        k_frames.append(fr)
        kr.append(float(lux[(lux.fr == fr) & (lux.method == m)].ratio.iloc[0]))
        for det, _, key in DETS:
            v = D[det][(D[det].fr == fr) & (D[det].method == m)][KEY]
            if len(v):
                kv[key].append(float(v.iloc[0]))
        kj += list(pj[(pj.fr == fr) & (pj.method == m)].kept_codec)
    N['Kstar'] = f'{ks[0]:,}'.replace(',', '{,}')
    N['KstarCalFrames'], N['KstarFrames'] = str(len(ks)), str(len(k_frames))
    N['KstarRatio'] = times(min(kr), max(kr))
    for *_, key in DETS:
        if kv[key]:
            N[f'Kstar{key}'] = rng(kv[key])
    N['KstarJtk'] = rng(kj)

    # ---- detector dependence vs JPEG-XL at 45-94x, per segmenter (frames where both exist)
    for det, _, key in DETS[1:]:
        df = D[det]
        gaps, lx = [], []
        for fr in ALL:
            a = df[(df.fr == fr) & (df.family == 'jpegxl') & df.ratio.between(45, 94)][KEY]
            b = df[(df.fr == fr) & (df.family == 'luxar') & df.ratio.between(45, 94)][KEY]
            if len(a) and len(b):
                gaps.append(a.clip(upper=1.0).mean() - b.mean())
                lx += list(b)
        if gaps:
            N[f'Gap{key}'] = pts(gaps)
            N[f'GapFrames{key}'] = str(len(gaps))
            N[f'LuxMid{key}'] = rng(lx)
    N['LuxMidLoG'] = rng(log[log.fr.isin(ALL) & (log.family == 'luxar') & log.ratio.between(45, 94)][KEY])

    # pre-registered section-4 tests, recomputed with the rule's own definitions (a1_crossings)
    spec = importlib.util.spec_from_file_location('a1c', REPO / 'scripts/fidelity/a1_crossings.py')
    a1c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a1c)
    res = pd.read_csv(F / 'results_all.csv')
    for det, key in (('watershed', 'WS'), ('cellpose3d', 'CPthree'), ('cellpose', 'CP')):
        g = [a1c.gap(x) for _, x in res[(res.detector == det) & res.role.isin(['main', 'a1'])].groupby('frame')]
        g = [x[0] for x in g if x is not None]
        N[f'RuleGapBig{key}'] = f'{sum(v >= 0.05 for v in g)} of {len(g)}'
        N[f'RuleGapSmall{key}'] = f'{sum(v < 0.05 for v in g)} of {len(g)}'
    ok = 0
    a1r = res[res.role == 'a1']
    ks_ = [f'luxar_K{k}' for k in (1000, 1500, 2000, 4000, 16000)]
    nfr = 0
    for _, g in a1r[(a1r.detector == 'log') & a1r.method.isin(ks_)].groupby('frame'):
        nfr += 1
        ok += (g.psnr.max() - g.psnr.min() < 3) and (g[KEY.replace('@', '@r')].max() - g[KEY.replace('@', '@r')].min() >= 0.15)
    N['RulePsnrOK'] = f'{ok} of {nfr}'

    # Cellpose wobble between neighbouring codec sizes (max jump of a codec curve below 150x)
    wob = []
    for fr in ALL:
        for fam in ('jpeg2k', 'jpegxl'):
            s = cp[(cp.fr == fr) & (cp.family == fam) & (cp.ratio < 150)].sort_values('ratio')[KEY].values
            if len(s) > 1:
                wob.append(np.max(np.abs(np.diff(s))))
    N['CPwobble'] = f'{100 * max(wob):.0f}'

    # ---- run-to-run spread: three identical Luxar fits of E1 t194 (original + 2 repeats)
    for det, _, key in DETS[:3]:
        base = D[det][(D[det].fr == ('s01', 194))]
        spread = []
        for k in (2000, 4000, 16000):
            v = [float(base[base.method == f'luxar_K{k}'][KEY].iloc[0])]
            for rep in ('rep1', 'rep2'):
                d = pd.read_csv(F / f'{det}_ce_t194_{rep}.csv')
                raw = d[d.method == 'raw'].iloc[0]['found@r0.6']
                v.append(float(d[d.method == f'luxar_K{k}'].iloc[0]['found@r0.6'] / raw))
            spread.append(max(v) - min(v))
        N[f'Rep{key}'] = f'{100 * max(spread):.1f}'

    # ---- cliff (LoG): splats per nucleus, all 12 frames
    def per_nuc(df):
        m = df.merge(lux[['fr', 'method', 'splats', 'n_true']], on=['fr', 'method'])
        m['per_nuc'] = m.splats / m.n_true
        return m
    lg = per_nuc(log)
    lga = lg[lg.fr.isin(ALL)]
    N['CliffBelow'] = f'{lga[lga[KEY] < 0.90].per_nuc.max():.1f}'
    N['CliffOne'] = rng(lga[lga.per_nuc.between(0.7, 1.4)][KEY])
    cr = pd.read_csv(F / 'a1_crossings.csv')
    c90 = cr[(cr.detector == 'log') & (cr.family == 'luxar') & cr.frame.isin([LABEL[f] for f in ALL])]['spn@0.90']
    vals = [float(str(v).lstrip('<>')) for v in c90]
    N['CrossRange'] = f'{min(vals):.1f}--{max(vals):.1f}'

    # ---- PSNR range and fit time (Luxar, K >= 1000)
    big = luxa[luxa.seeds >= 1000]
    N['LuxPSNRrange'] = f'{big.psnr.min():.1f}--{big.psnr.max():.1f}\\,dB'
    span = big.groupby('fr').psnr.agg(lambda s: s.max() - s.min())
    N['LuxPSNRspan'] = f'{span.max():.1f}\\,dB'
    fa = luxa[luxa.gpu == 'A100'].fit_min
    fh = luxa[luxa.gpu == 'H100'].fit_min
    N['FitA'] = f'{fa.min():.0f}--{fa.max():.0f}\\,min'
    N['FitH'] = f'{fh.min():.0f}--{fh.max():.0f}\\,min'

    # ---- survival guard (all Luxar fits on the 12 frames)
    m = luxa.merge(log[['fr', 'method', KEY]], on=['fr', 'method'])
    rho, _ = spearmanr(m.survival, m[KEY])
    N['SurvRho'], N['SurvN'] = f'{rho:.2f}', str(len(m))

    with open(PAPER / 'numbers.tex', 'w') as fh_:
        fh_.write('% generated by scripts/fidelity/paper_numbers.py -- do not edit\n')
        for k, v in N.items():
            fh_.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')

    # ---- Table 1: Luxar minus JPEG2000 at identical bytes, by ratio, per detector
    lines = ['\\begin{tabular}{rrrrrrr}', '\\toprule',
             'Ratio & Spl./nuc. & $\\Delta$PSNR & LoG & Wshed & CP 2D & CP 3D \\\\', '\\midrule']
    for lo_r, hi_r, lab in BINS:
        b = pj[pj.ratio.between(lo_r, hi_r)]
        if b.empty:
            continue
        one = b.drop_duplicates(['frame', 'method'])
        spn = one.splats_per_nucleus
        cells = [f'$\\approx${lab}$\\times$',
                 f'{spn.min():.0f}--{spn.max():.0f}' if spn.min() >= 1.5 else f'{spn.min():.1f}--{spn.max():.1f}',
                 f'{(one.psnr_lux - one.psnr_codec).mean():+.1f}'.replace('-', '$-$')]
        for det, _, _ in DETS:
            g = b[b.detector == det]
            if g.empty:
                cells.append('--')
                continue
            v = f'{100 * g["diff"].mean():+.1f}'.replace('-', '$-$')
            if (g.hi < 0).all() or (g.lo > 0).all():
                v = f'\\textbf{{{v}}}'
            cells.append(v + f'$^{{{g.frame.nunique()}}}$')
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (PAPER / 'table_bins.tex').write_text('\n'.join(lines) + '\n')   # supplementary

    # ---- Table 1 (main): frame-level verdicts, PSNR first -- "PSNR says JPEG2000; detectors: it depends"
    lines = ['\\begin{tabular}{lrcrc}', '\\toprule',
             ' & \\multicolumn{2}{c}{below \\SI{300}{\\times}} & \\multicolumn{2}{c}{\\SI{300}{\\times} and beyond} \\\\',
             '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
             'Luxar $-$ JPEG2000 & mean [95\\% CI] & L\\,:\\,J & mean [95\\% CI] & L\\,:\\,J \\\\', '\\midrule']
    f1 = lambda x: f'{x:+.1f}'.replace('-', '$-$')
    for det, label, key in [('psnr', 'PSNR (dB)', 'PSNR')] + DETS:
        cells = [label if det != 'psnr' else 'PSNR (dB)']
        for reg in ('mod', 'far'):
            r = FL[(FL.det == det) & (FL.reg == reg)].iloc[0]
            v = f'{f1(r["mean"])} [{f1(r.lo)}, {f1(r.hi)}]'
            if det != 'psnr' and r.p_holm < 0.05:
                v = f'\\textbf{{{f1(r["mean"])}}} [{f1(r.lo)}, {f1(r.hi)}]'
            cells += [v, f'{r.pos}\\,:\\,{r.neg}']
        lines.append(' & '.join(cells) + ' \\\\')
        if det == 'psnr':
            lines.append('\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}']
    (PAPER / 'table_main.tex').write_text('\n'.join(lines) + '\n')

    # ---- Table 2: radius sensitivity (paper frames; raw found summed; kept at ~90x mean)
    lines = ['\\begin{tabular}{lcccccc}', '\\toprule',
             ' & \\multicolumn{3}{c}{LoG} & \\multicolumn{3}{c}{Cellpose (2D)} \\\\',
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

    def logx(ax, ticks):
        ax.set_xscale('log')
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # ---- Figure 2: Luxar minus JPEG2000 at identical bytes, every pair, per detector
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.85), sharey=True)
    for ax, (det, dn, _) in zip(axes, DETS):
        g = pj[pj.detector == det]
        floor = -48
        for emb, col in EMB_COL.items():
            q = g[g.fr.map(lambda f: f[0]) == emb]
            inr, out = q[100 * q['diff'] >= floor], q[100 * q['diff'] < floor]
            ax.errorbar(inr.ratio, 100 * inr['diff'], yerr=[100 * (inr['diff'] - inr.lo), 100 * (inr.hi - inr['diff'])],
                        fmt='o', ms=2.2, color=col, ecolor=col, elinewidth=0.5, alpha=0.75, capsize=0,
                        label=f'embryo {int(emb[1:])}')
            ax.plot(out.ratio, [floor + 1.5] * len(out), 'v', ms=2.6, color=col, alpha=0.75)  # off-scale
        ax.axhline(0, color=INK_2, lw=0.7)
        ax.axvline(FAR, color=INK_2, lw=0.5, ls=':')
        logx(ax, [30, 100, 300])
        ax.set_title(f'{dn} ({g.frame.nunique()} frames)')
        ax.set_xlabel('compression ratio ($\\times$)')
        ax.set_ylim(floor, 20)
        style(ax)
    axes[0].set_ylabel('Luxar $-$ JPEG2000\n(points of nuclei kept)')
    axes[0].legend(frameon=False, fontsize=5.5, loc='lower left', handletextpad=0.2)
    fig.tight_layout(pad=0.3, w_pad=0.4)
    fig.savefig(PAPER / 'figures' / 'fig_matched.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    # ---- Figure 3: cliff -- LoG nuclei kept vs splats per labelled nucleus, all frames
    fig, ax = plt.subplots(figsize=(3.45, 1.55))
    for fr in ALL + EARLY:
        q = lg[lg.fr == fr].sort_values('per_nuc')
        ax.plot(q.per_nuc, q[KEY], color=EMB_COL[fr[0]], lw=0.8, marker='o', ms=1.8, alpha=0.8,
                ls='--' if fr in EARLY else '-')
    ax.axhline(1, color=INK_2, lw=0.6, ls=':')
    ax.axhline(0.9, color=INK_2, lw=0.5, alpha=0.5)
    logx(ax, [1, 3, 10, 30, 100])
    ax.set_xlabel('splats per labelled nucleus')
    ax.set_ylabel('LoG: nuclei kept')
    ax.set_ylim(0.55, 1.12)
    ax.plot([], [], color=EMB_COL['s01'], label='embryo 1')
    ax.plot([], [], color=EMB_COL['s02'], label='embryo 2')
    ax.plot([], [], color=EMB_COL['s01'], ls='--', label='E1 t100')
    ax.legend(frameon=False, fontsize=5.5, loc='lower right')
    style(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(PAPER / 'figures' / 'fig_cliff.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    # ---- supplementary grid: rate-detectability on the 4 paper frames, all detectors
    fig, axes = plt.subplots(len(DETS), 4, figsize=(7.1, 5.4), sharex=True, sharey=True)
    for ri, (det, dn, _) in enumerate(DETS):
        for ci, fr in enumerate(MAIN):
            ax = axes[ri, ci]
            s = D[det][(D[det].fr == fr) & (D[det].method != 'raw')]
            for fam, lab, col, mk in METHODS:
                q = s[s.family == fam].sort_values('ratio')
                if len(q):
                    ax.plot(q.ratio, q[KEY], color=col, lw=1.1, marker=mk, ms=2.8, mec='white', mew=0.4,
                            label=lab.split(' (')[0])
            ax.axhline(1, color=INK_2, lw=0.6, ls=':')
            logx(ax, [20, 50, 100, 200, 500])
            ax.set_ylim(0.4, 1.25)
            style(ax)
            if ri == 0:
                ax.set_title(f'{LABEL[fr]} ({int(nl[fr])} nuclei)')
            if ci == 0:
                ax.set_ylabel(f'{dn}: kept')
            if ri == len(DETS) - 1:
                ax.set_xlabel('compression ratio ($\\times$)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97), pad=0.3)
    fig.savefig(PAPER / 'figures' / 'fig_rate_detectability.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    qualitative_figure(cp)
    for k, v in N.items():
        print(f'{k:16s} {v}')

    # ---- claims the TEXT makes in words; fail loudly if the data stop supporting them
    mod = pj[pj.ratio < FAR]
    far = pj[pj.ratio >= FAR]
    sgn = lambda det: mod[mod.detector == det]['diff'].mean()
    fr_ = lambda det, reg: FL[(FL.det == det) & (FL.reg == reg)].iloc[0]
    claims = {
        'frame level, below 300x: 3D Cellpose favours splats (Holm p < .05, >= 9 of 12 frames)':
            fr_('cellpose3d', 'mod').p_holm < 0.05 and fr_('cellpose3d', 'mod').pos >= 9,
        'frame level, below 300x: watershed and 2D Cellpose favour JPEG2000 (Holm p < .05)':
            all(fr_(d, 'mod').p_holm < 0.05 and fr_(d, 'mod')['mean'] < 0 for d in ('watershed', 'cellpose')),
        'frame level, below 300x: LoG shows NO consistent difference (Holm p >= .05)':
            fr_('log', 'mod').p_holm >= 0.05,
        'frame level, beyond 300x: every detector favours JPEG2000 (Holm p < .05)':
            all(fr_(d, 'far').p_holm < 0.05 and fr_(d, 'far')['mean'] < 0 for d, *_ in DETS),
        'frame level: PSNR favours JPEG2000 on every frame in both regimes':
            all(fr_('psnr', r).neg == fr_('psnr', r).n for r in ('mod', 'far')),
        'segmenters (watershed, Cellpose 2D) never significantly favour Luxar':
            N['PairsLWS'] == '0' and N['PairsLCP'] == '0',
        'beyond ~300x JPEG2000 leads on average for every detector':
            all(far[far.detector == d]['diff'].mean() < 0 for d, *_ in DETS),
        'JPEG2000 has the higher PSNR at every matched size':
            N['PSNRjtkWins'].split(' of ')[0] == N['PSNRjtkWins'].split(' of ')[1],
        'every embryo-2 calibration recommends the same K* as embryo 1':
            len(set(ks)) == 1 and len(ks) == 4,
        'cliff: LoG 90% crossing <= 3.5 splats per nucleus on every frame':
            max(vals) <= 3.5,
        'PSNR rule "held on only X of 8" -- i.e. it failed (< 6)':
            int(N['RulePsnrOK'].split(' of ')[0]) < 6,
        '3D Cellpose: the 2D penalty is absent (gap >= 5 on fewer than 75% of frames)':
            int(N['RuleGapBigCPthree'].split(' of ')[0]) < 0.75 * int(N['RuleGapBigCPthree'].split(' of ')[1]),
        'watershed: gap < 5 on >= 75% of frames ("sits in between")':
            int(N['RuleGapSmallWS'].split(' of ')[0]) >= 0.75 * int(N['RuleGapSmallWS'].split(' of ')[1]),
    }
    bad = [c for c, ok in claims.items() if not ok]
    if bad:
        raise SystemExit('TEXT CLAIMS NO LONGER SUPPORTED -- rewrite paper/main.tex:\n  ' + '\n  '.join(bad))
    print(f'all {len(claims)} worded claims supported')


def qualitative_figure(cp, fr=('s01', 194)):
    """Same region from raw, JPEG-XL, JPEG2000 and Luxar at ~90x. Titles give WHOLE-FRAME
    nuclei kept (Cellpose 2D, 0.6 D); per-crop counts would be a cherry-pickable anecdote.
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

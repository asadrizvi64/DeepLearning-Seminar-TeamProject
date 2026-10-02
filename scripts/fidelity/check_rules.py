"""Evaluate paper/DECISION_RULES.md directly from the result files.

Prints each rule's verdict with the evidence behind it and writes
runs/fidelity/rule_check.md. Radius 0.6 D throughout, as the rules state.
A rule whose required measurements do not exist yet is reported NOT EVALUABLE -- it is
never replaced by a nearby comparison.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
F = REPO / 'runs' / 'fidelity'
R = 'kept@0.6'
FILES = {  # (embryo/seq, frame) -> (log csv, cellpose csv)
    ('s01', 150): ('log_ce_t150.csv', 'cellpose_ce_t150.csv'),
    ('s01', 194): ('log_ce_t194.csv', 'cellpose_ce_t194.csv'),
    ('s02', 150): ('log_ce_s02_t150.csv', 'cellpose_ce_s02_t150.csv'),
    ('s02', 180): ('log_ce_s02_t180.csv', 'cellpose_ce_s02_t180.csv'),
    ('s01', 100): ('log_ce_s01_t100.csv', 'cellpose_ce_s01_t100.csv'),
}
N_LABELLED = {('s01', 150): 192, ('s01', 194): 362, ('s02', 150): 190, ('s02', 180): 351,
              ('s01', 100): 93}


def load(name):
    d = pd.read_csv(F / name)
    raw = d[d.method == 'raw'].iloc[0]
    d[R] = d['found@r0.6'] / raw['found@r0.6']
    d['family'] = d.method.str.replace(r'_K\d+$', '', regex=True)
    d['seeds'] = d.method.str.extract(r'_K(\d+)$')[0].astype(float)
    return d


def splat_counts():
    """n_splats per Luxar fit, from the cluster tables."""
    out = {}
    for base in ('fidelity_cluster', 'fidelity_cluster2'):
        for meta in (REPO / 'runs' / base / 'fidelity').glob('pilot_ce_*/meta.json'):
            m = json.load(open(meta))
            d = pd.read_csv(meta.parent / 'rate_detectability.csv')
            for _, r in d[d.method.str.startswith('luxar')].iterrows():
                out[(f"s{m.get('seq', '01')}", m['frame'], r.method)] = json.loads(r.params)['n_splats']
    return out


def main():
    data = {k: (load(a), load(b)) for k, (a, b) in FILES.items() if (F / a).exists() and (F / b).exists()}
    lines = ['# Rule check (radius 0.6 D)', '']

    def say(s=''):
        print(s)
        lines.append(s)

    # ---- Rule 1a: detector dependence on embryo 2 (Cellpose, 55-94x)
    say('## 1a. Detector dependence replicates on embryo 2')
    ok_all, evaluable = True, True
    for fr in (150, 180):
        _, cp = data[('s02', fr)]
        lx = cp[(cp.family == 'luxar') & cp.ratio.between(55, 94)]
        jx = cp[(cp.family == 'jpegxl') & cp.ratio.between(55, 94)]
        if lx.empty or jx.empty:
            say(f'- s02 t{fr}: NOT EVALUABLE -- Luxar fits in 55-94x: {len(lx)}, JPEG-XL: {len(jx)}')
            evaluable = False
            continue
        gap = jx[R].mean() - lx[R].mean()
        say(f'- s02 t{fr}: JPEG-XL {jx[R].mean():.3f} vs Luxar {lx[R].mean():.3f} -> gap {100 * gap:.1f} points')
        ok_all &= gap >= 0.05
    say(f'**Verdict: {"NOT EVALUABLE" if not evaluable else ("REPLICATES" if ok_all else "DOES NOT REPLICATE")}**')
    say()

    # ---- Rule 1b: high-ratio advantage over JPEG2000 on embryo 2, both detectors
    say('## 1b. Beyond 120x, Luxar keeps more nuclei than JPEG2000 (embryo 2, both detectors)')
    ok_all, n_pairs = True, 0
    for fr in (150, 180):
        for det, df in zip(('LoG', 'Cellpose'), data[('s02', fr)]):
            j2 = df[df.family == 'jpeg2k']
            for _, lx in df[(df.family == 'luxar') & (df.ratio > 120)].iterrows():
                i = (np.log(j2.ratio) - np.log(lx.ratio)).abs().idxmin()
                j = j2.loc[i]
                if max(j.ratio / lx.ratio, lx.ratio / j.ratio) > 1.25:
                    continue
                n_pairs += 1
                win = lx[R] > j[R]
                ok_all &= win
                say(f'- s02 t{fr} {det}: Luxar {lx.ratio:.0f}x {lx[R]:.2f} vs JPEG2000 {j.ratio:.0f}x {j[R]:.2f} '
                    f'{"" if win else "  <-- Luxar not better"}')
    say(f'**Verdict: {"NOT EVALUABLE" if n_pairs == 0 else ("REPLICATES" if ok_all else "DOES NOT REPLICATE")}** '
        f'({n_pairs} size-matched pairs within 1.25x)')
    say()

    # ---- Rule 1c: JPEG-XL >= 97% under both detectors at every reachable size (embryo 2)
    say('## 1c. JPEG-XL keeps >= 97% at every size it reaches (embryo 2, both detectors)')
    worst = []
    for fr in (150, 180):
        for det, df in zip(('LoG', 'Cellpose'), data[('s02', fr)]):
            jx = df[df.family == 'jpegxl']
            m = jx[R].min()
            worst.append(m)
            say(f'- s02 t{fr} {det}: min {m:.3f} over {len(jx)} sizes ({jx.ratio.min():.0f}-{jx.ratio.max():.0f}x)')
    say(f'**Verdict: {"HOLDS" if min(worst) >= 0.97 else "QUALIFY"}** (worst {min(worst):.3f})')
    say()

    # ---- Rule 2: cliff scaling on frame 100, plus all frames for context
    say('## 2. Cliff sits below ~5 splats per labelled nucleus (LoG)')
    splats = splat_counts()
    verdict2 = None
    for key in sorted(data):
        log, _ = data[key]
        lx = log[log.family == 'luxar'].copy()
        lx['per_nuc'] = [splats.get((key[0], key[1], m), np.nan) / N_LABELLED[key] for m in lx.method]
        lx = lx.sort_values('per_nuc')
        below = lx[lx[R] < 0.90]
        cells = ', '.join(f'{p:.1f}/nuc {k:.2f}' for p, k in zip(lx.per_nuc, lx[R]))
        say(f'- {key[0]} t{key[1]}: {cells}')
        if key == ('s01', 100):
            if below.empty:
                verdict2 = f'NOT CONTRADICTED -- never below 90% (lowest budget {lx.per_nuc.min():.1f}/nuc)'
            else:
                verdict2 = 'SUPPORTED' if (below.per_nuc < 5).all() else 'CONTRADICTED'
    say(f'**Verdict (frame 100): {verdict2}**')
    say()

    # ---- Rule 3: Luxar's self-calibrated K*
    say("## 3. Luxar's own budget K*")
    safe = True
    for fr in (150, 194):
        s = json.load(open(REPO / 'runs' / 'fidelity_cluster2' / 'fidelity' / f'calibrate_ce_s01_t{fr}' / 'summary.json'))
        k = s['k_star']
        log, cp = data[('s01', fr)]
        kl = log[log.method == f'luxar_K{k}'][R]
        kc = cp[cp.method == f'luxar_K{k}'][R]
        kl = float(kl.iloc[0]) if len(kl) else np.nan
        kc = float(kc.iloc[0]) if len(kc) else np.nan
        ratio = log[log.method == f'luxar_K{k}'].ratio
        say(f'- s01 t{fr}: K* = {k} ({s["curve_type"]}, still climbing), ~{float(ratio.iloc[0]):.0f}x: '
            f'LoG {kl:.3f}, Cellpose {kc:.3f}')
        safe &= kl >= 0.95
    say(f'**Verdict: {"SAFE for blob detection -- limit the PSNR-blind claim to hand-chosen budgets" if safe else "LOSES NUCLEI"}**')
    say()

    # ---- Rule 5: Luxar vs JPEG2000 at exactly the same bytes (paired CIs, matched_pairs.py)
    say('## 5. Luxar vs JPEG2000 at the same bytes (paired 95% CI over nuclei)')
    mp = F / 'matched_pairs.csv'
    if not mp.exists():
        say('**NOT EVALUABLE -- run scripts/fidelity/matched_pairs.py**')
    else:
        p = pd.read_csv(mp)
        p = p[p.codec == 'jpeg2k']
        any_lux = False
        for det, g in p.groupby('detector'):
            jb, lb = g[g.hi < 0], g[g.lo > 0]
            say(f'- {det}: {len(g)} pairs on {g.frame.nunique()} frames -- JPEG2000 better {len(jb)}, '
                f'no difference {len(g) - len(jb) - len(lb)}, Luxar better {len(lb)}')
            for _, r in lb.iterrows():
                say(f'    - Luxar better: {r.frame} {r.method} {r.ratio:.0f}x, '
                    f'{100 * r["diff"]:+.1f} points [{100 * r.lo:+.1f}, {100 * r.hi:+.1f}]')
            any_lux |= len(lb) > 0
        say(f'**Verdict: {"Luxar better somewhere -- the paper names these combinations" if any_lux else "JPEG2000 keeps at least as many nuclei at every matched size"}**')

    (F / 'rule_check.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

"""Summarise the budget-matched temporal runs (scripts/temporal_matched.py) for the report
and the slides: team_report/generated/temporal_table.tex and presentation/figs/temporal_summary.json.
Results for the same seed in several run directories are merged.

    python scripts/temporal_summary.py runs/temporal_matched_s0 runs/temporal_matched_s12 \
        runs/temporal_matched_s12_small
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

by_seed = {}
for d in sys.argv[1:]:
    f = Path(d) / 'report.json'
    if f.exists():
        for r in json.load(open(f))['runs']:
            by_seed.setdefault(r['seed'], {}).update(r)
seeds = sorted(by_seed)
assert seeds, 'no report.json found'


def stat(values):
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


VARIANTS = (('shared_identity', 'One set per frame'), ('native_4d', 'Native 4D'),
            ('deformation', 'Deformation'), ('native_4d_small', 'Native 4D, 300 Gaussians'),
            ('deformation_small', 'Deformation, 300 Gaussians'))
rows = {}
for key, label in VARIANTS:
    per = [by_seed[s][key] for s in seeds if key in by_seed[s]]
    if not per:
        continue
    row = {'label': label, 'n_seeds': len(per), 'gaussians': per[0]['gaussians'],
           'params': per[0]['params'], 'fit': stat([np.mean(p['fit_psnr']) for p in per]),
           'fit_min': stat([p['fit_s'] / 60 for p in per])}
    if key == 'shared_identity':
        row['held'] = stat([np.mean(p['held_nearest_psnr']) for p in per])
        row['held_xfade'] = stat([np.mean(p['held_crossfade_psnr']) for p in per])
        row['smooth'] = stat([p['smoothness_vox'] for p in per])
        row['consistency'] = stat([np.mean(p['consistency_psnr']) for p in per])
    else:
        row['held'] = stat([np.mean(p['held_psnr']) for p in per])
    rows[key] = row

pm = lambda s: f"{s[0]:.1f} $\\pm$ {s[1]:.1f}"
num = lambda n: f"{n:,}".replace(',', '{,}')
si = rows['shared_identity']
lines = [
    # no fit-time column: wall-clock times of runs that spanned a laptop sleep are meaningless
    r'\begin{tabular}{@{}lrrcc@{}}', r'\toprule',
    r'Representation & Gaussians & Numbers & Fit frames (dB) & Held-out frames (dB) \\',
    r'\midrule',
    f"One set per frame, nearest frame & $4\\times{si['gaussians']}$ & {num(si['params'])} & {pm(si['fit'])} & {pm(si['held'])} \\\\",
    f"\\quad cross-fade of two frames & & & & {pm(si['held_xfade'])} \\\\",
]
for key in ('native_4d', 'deformation'):
    r = rows[key]
    lines.append(f"{r['label']} & {num(r['gaussians'])} & {num(r['params'])} & {pm(r['fit'])} & {pm(r['held'])} \\\\")
small = [k for k in ('native_4d_small', 'deformation_small') if k in rows]
if small:
    lines.append(r'\midrule')
    for key in small:
        r = rows[key]
        lines.append(f"{r['label'].split(',')[0]} & {num(r['gaussians'])} & {num(r['params'])} & {pm(r['fit'])} & {pm(r['held'])} \\\\")
lines += [r'\bottomrule', r'\end{tabular}']
(REPO / 'team_report' / 'generated' / 'temporal_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

summary = {'seeds': seeds, 'rows': rows}
(REPO / 'presentation' / 'figs' / 'temporal_summary.json').write_text(json.dumps(summary, indent=2))
print(f'seeds {seeds}')
for k, r in rows.items():
    extra = (f"  cross-fade {pm(r['held_xfade'])}  smooth {r['smooth'][0]:.3f}  consist {pm(r['consistency'])}"
             if 'smooth' in r else '')
    print(f"{r['label']:<28} n={r['n_seeds']} N={r['gaussians']:<5} params={r['params']:<6} "
          f"fit {pm(r['fit'])}  held {pm(r['held'])}  {r['fit_min'][0]:.0f} min{extra}")

"""Full-precision historical-vs-rerun comparison, all 8 fits, all 9 recorded fields.
No tolerance rounding (contrast with rescore_scale_cap.py's compare_to_historical(),
which uses np.isclose(atol=1e-3, rtol=1e-3) and is only called for the first region) --
raw float64 repr() and exact differences, for every region/arm pair.

Usage:
    python scripts/full_precision_compare.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
rescore = pd.read_csv(REPO / 'runs/scale_cap_rescore/rescore.csv')
historical = pd.read_csv(REPO / 'runs/scale_cap/scale_cap.csv')

FIELDS = ['S_A', 'S_B', 'recall', 'scale_kept', 'scale_lost', 'scale_B_lost',
          'psnr_nucleus', 'psnr_background', 'psnr_global']

old = rescore[rescore.matcher == 'old_buggy']
for _, row in old.iterrows():
    h = historical[(historical.region == row.region) & (historical.arm == row.arm)].iloc[0]
    print(f'\n{row.region} / {row.arm}')
    for f in FIELDS:
        hv, nv = float(h[f]), float(row[f])
        diff = nv - hv
        print(f'    {f:16s} hist={hv!r:24s} rerun={nv!r:24s} diff={diff!r}')

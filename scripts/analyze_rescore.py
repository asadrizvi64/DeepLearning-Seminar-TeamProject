"""Post-hoc analysis of runs/scale_cap_rescore/ -- the 4 reports requested during
review (historical-vs-rerun, old-vs-new matcher per fit, paired control->medium
gains/losses under each matcher, and reference-count/index correspondence checked
against this run's own saved outputs). No new fits -- pure analysis of already-saved
outputs (rescore.csv, peaks/*.npy, matches/*.npz).

Usage:
    python scripts/analyze_rescore.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / 'runs/scale_cap_rescore'

rescore = pd.read_csv(OUT / 'rescore.csv')
historical = pd.read_csv(REPO / 'runs/scale_cap/scale_cap.csv')

FIELDS = ['S_A', 'S_B', 'recall', 'scale_kept', 'scale_lost', 'scale_B_lost',
          'psnr_nucleus', 'psnr_background', 'psnr_global']

print('=' * 100)
print('REPORT 1: historical score vs rerun score, OLD matcher, every region/arm')
print('=' * 100)
old = rescore[rescore.matcher == 'old_buggy']
for _, row in old.iterrows():
    h = historical[(historical.region == row.region) & (historical.arm == row.arm)]
    if len(h) == 0:
        print(f'{row.region}/{row.arm}: no historical row found')
        continue
    h = h.iloc[0]
    diffs = []
    for f in FIELDS:
        hv, nv = float(h[f]), float(row[f])
        if not np.isclose(hv, nv, atol=1e-3, rtol=1e-3, equal_nan=True):
            diffs.append(f'{f}: hist={hv:.4f} rerun={nv:.4f}')
    status = 'EXACT MATCH' if not diffs else 'MISMATCH: ' + '; '.join(diffs)
    print(f'  {row.region:12s} {row.arm:8s}  {status}')

print()
print('=' * 100)
print('REPORT 2: old vs corrected matching, same reconstruction, every fit')
print('=' * 100)
for region in rescore.region.unique():
    for arm in rescore[rescore.region == region].arm.unique():
        o = rescore[(rescore.region == region) & (rescore.arm == arm) & (rescore.matcher == 'old_buggy')].iloc[0]
        n = rescore[(rescore.region == region) & (rescore.arm == arm) & (rescore.matcher == 'new_fixed')].iloc[0]
        old_m = np.load(OUT / 'matches' / f'{region}_{arm}_old.npz')
        new_m = np.load(OUT / 'matches' / f'{region}_{arm}_new.npz')
        n_old, n_new = len(old_m['target_idx']), len(new_m['target_idx'])
        diff = '  <-- DIFFERS' if n_old != n_new else ''
        print(f'  {region:12s} {arm:8s}  matched(old)={n_old:2d}  matched(new)={n_new:2d}  '
              f'S_A {o.S_A:.3f}->{n.S_A:.3f}  S_B {o.S_B:.3f}->{n.S_B:.3f}  '
              f'recall {o.recall:.3f}->{n.recall:.3f}{diff}')

print()
print('=' * 100)
print('REPORT 3: control -> medium paired gains/losses, under EACH matcher')
print('=' * 100)
for matcher_tag, npz_suffix in [('OLD (buggy)', 'old'), ('NEW (fixed)', 'new')]:
    print(f'\n--- matcher: {matcher_tag} ---')
    all_kept_found = all_regressed = all_rescued = all_still_missed = 0
    A_kept = A_reg = A_res = A_miss = 0
    B_kept = B_reg = B_res = B_miss = 0
    n_A_total = n_B_total = n_total = 0
    for region in rescore.region.unique():
        nuclei = np.load(OUT / 'peaks' / f'{region}_control_nuclei.npy')
        n_nuc = len(nuclei)
        ctrl_m = np.load(OUT / 'matches' / f'{region}_control_{npz_suffix}.npz')
        med_m = np.load(OUT / 'matches' / f'{region}_medium_{npz_suffix}.npz')
        got_ctrl = np.zeros(n_nuc, dtype=bool); got_ctrl[ctrl_m['target_idx']] = True
        got_med = np.zeros(n_nuc, dtype=bool); got_med[med_m['target_idx']] = True

        # group membership (A/B) -- recompute exactly as the run did (local_maxima init vs SEED_R)
        from scripts.scale_cap import SEED_R, SHAPE
        from volsplat.init import init_gaussians
        z, y, x = [int(v) for v in region.replace('z', '').replace('y', ' ').replace('x', ' ').split()]
        vol = np.load(REPO / 'runs/colleague_full_volume.npy').astype(np.float32)
        roi = np.ascontiguousarray(vol[z:z + SHAPE[0], y:y + SHAPE[1], x:x + SHAPE[2]])
        old_init = init_gaussians(roi, 250, strategy='local_maxima', init_scale=2.0, seed=0)
        p_old = old_init.positions.detach().cpu().numpy()[:, ::-1]
        d_old = np.linalg.norm(nuclei[:, None, :] - p_old[None, :, :], axis=-1).min(axis=1)
        in_A = d_old <= SEED_R

        kept_found = int((got_ctrl & got_med).sum())
        regressed = int((got_ctrl & ~got_med).sum())
        rescued = int((~got_ctrl & got_med).sum())
        still_missed = int((~got_ctrl & ~got_med).sum())
        all_kept_found += kept_found; all_regressed += regressed
        all_rescued += rescued; all_still_missed += still_missed
        n_total += n_nuc

        for grp_mask, pre in [(in_A, 'A'), (~in_A, 'B')]:
            kf = int((got_ctrl & got_med & grp_mask).sum())
            rg = int((got_ctrl & ~got_med & grp_mask).sum())
            rs = int((~got_ctrl & got_med & grp_mask).sum())
            sm = int((~got_ctrl & ~got_med & grp_mask).sum())
            if pre == 'A':
                A_kept += kf; A_reg += rg; A_res += rs; A_miss += sm; n_A_total += int(grp_mask.sum())
            else:
                B_kept += kf; B_reg += rg; B_res += rs; B_miss += sm; n_B_total += int(grp_mask.sum())

        print(f'  {region:12s} n={n_nuc:3d}  kept_found={kept_found:2d}  REGRESSED={regressed:2d}  '
              f'RESCUED={rescued:2d}  still_missed={still_missed:2d}  net={rescued-regressed:+d}')

    print(f'  {"ALL":12s} n={n_total:3d}  kept_found={all_kept_found:2d}  REGRESSED={all_regressed:2d}  '
          f'RESCUED={all_rescued:2d}  still_missed={all_still_missed:2d}  net={all_rescued-all_regressed:+d}')
    print(f'  {"Group A":12s} n={n_A_total:3d}  kept_found={A_kept:2d}  REGRESSED={A_reg:2d}  '
          f'RESCUED={A_res:2d}  still_missed={A_miss:2d}  net={A_res-A_reg:+d}')
    print(f'  {"Group B":12s} n={n_B_total:3d}  kept_found={B_kept:2d}  REGRESSED={B_reg:2d}  '
          f'RESCUED={B_res:2d}  still_missed={B_miss:2d}  net={B_res-B_reg:+d}')

print()
print('=' * 100)
print('REPORT 4: reference counts + coordinate correspondence, THIS rerun\'s own saved outputs')
print('=' * 100)
from volsplat.init import init_gaussians
from scripts.scale_cap import SHAPE
vol = np.load(REPO / 'runs/colleague_full_volume.npy').astype(np.float32)
total_n = total_exact = 0
for region in rescore.region.unique():
    nuclei = np.load(OUT / 'peaks' / f'{region}_control_nuclei.npy')  # SAVED output of this run
    n_nuc = len(nuclei)
    z, y, x = [int(v) for v in region.replace('z', '').replace('y', ' ').replace('x', ' ').split()]
    roi = np.ascontiguousarray(vol[z:z + SHAPE[0], y:y + SHAPE[1], x:x + SHAPE[2]])
    # fresh, cheap (no training) init call, same seed/settings the run actually used
    gs_init = init_gaussians(roi, 250, strategy='coverage', init_scale=2.0, seed=0)
    pos_init = gs_init.positions.detach().cpu().numpy()[:n_nuc, ::-1]  # xyz -> zyx
    d = np.linalg.norm(pos_init - nuclei, axis=1)
    n_exact = int((d == 0).sum())
    total_n += n_nuc
    total_exact += n_exact
    print(f'  {region:12s} n_nuc={n_nuc:3d}  max_dist={d.max():.4f}  exact_matches={n_exact}/{n_nuc}')
print(f'  {"TOTAL":12s} n_nuc={total_n:3d}  exact_matches={total_exact}/{total_n}')

"""Diagnostic: does the scale-cap's recorded benefit survive CORRECTED matching, on
FAITHFULLY REPRODUCED historical fits?

This is option (a) from the bug report, scoped exactly as agreed:
    - matching-algorithm fix ONLY (Hungarian-then-threshold -> max-cardinality-first)
    - detection settings UNCHANGED (legacy voxel-cubic, no physical spacing) -- the
      physical-spacing fix is option (b) and is a DIFFERENT fitting configuration
      (it can change which nuclei the coverage initializer even seeds), not scored here
    - same seeds, same regions, same init, same fitting settings as the original
      scripts/scale_cap.py -- nothing about the EXPERIMENT changes, only how the result
      is SCORED

Why this can't be pure post-hoc rescoring: `runs/scale_cap/per_nucleus.csv` saved only
booleans + scale/amp, and scale_cap.py never saved `recon` arrays or checkpoints. There
is nothing to rescore without refitting. Fits are deterministic under a fixed seed, so a
refit is the only way to recover the raw peak coordinates the matcher needs.

REPRODUCTION GATE: `runs/scale_cap/scale_cap.csv` was written with `metrics_fingerprint
0e1028ce85fc`, but hashing the CURRENTLY COMMITTED (git HEAD) cellmetrics.py/metrics.py/
ablation.py gives a DIFFERENT fingerprint (8709bdfb29e2) -- the dirty diff that produced
the original run was never committed and is not recoverable (checked: no stash, no
reflog entry, no dangling git objects, no .orig/.bak files). So "identical seeds" does
NOT guarantee the original code has been recovered, exactly the caution this script's
requester raised. This run uses the best available reconstruction (git HEAD's original
detect_cells + a verbatim copy of scale_cap.py's original inline matcher, see
ORIG_CELLMETRICS_PY below) and REPORTS the comparison against the historically recorded
numbers rather than assuming it -- if they diverge, that divergence is itself reported,
not hidden, and the old-vs-new-matcher delta computed WITHIN this run is still valid
evidence about the matching bug's effect even if it doesn't equal what historically ran.

Persists, per fit, into a NEW output directory (nothing in runs/scale_cap/ is touched):
    gaussians/{region}_{arm}.npz      -- positions, scales, quaternions, amplitudes
    peaks/{region}_{arm}_nuclei.npy   -- target peaks (voxel zyx)
    peaks/{region}_{arm}_pred.npy     -- predicted peaks (voxel zyx)
    matches/{region}_{arm}_old.npz    -- ri, ci from the ORIGINAL buggy matcher
    matches/{region}_{arm}_new.npz    -- ri, ci, dist from the FIXED matcher
    code_snapshot/                    -- byte-for-byte copies of every module in the
                                          fitting+scoring path AS ACTUALLY IMPORTED by
                                          this process, plus full `git status
                                          --porcelain` and `git diff` output (not just
                                          the dirty boolean) -- closes exactly the gap
                                          that made the original run unreproducible.
    rescore.csv                       -- one row per (region, arm, matcher) pair

Usage:
    python scripts/rescore_scale_cap.py                 # 4 regions x [control, medium]
    python scripts/rescore_scale_cap.py --full           # all 4 regions x 4 arms
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from volsplat.init import init_gaussians
from volsplat.ablation import fit_with_validation
from volsplat.cellmetrics import match_indices as match_indices_fixed
from volsplat.provenance import run_metadata
from volsplat import metrics as M
from scripts.scale_cap import nucleus_mask, REGIONS, SHAPE, SEED_R, NUC_VOXEL_R

OUT = REPO / 'runs/scale_cap_rescore'  # overridden in main() if --outdir is passed
HISTORICAL_CSV = REPO / 'runs/scale_cap/scale_cap.csv'
MATCH_R = 0.765 * 13  # identical to the historical run -- legacy voxel-space radius

# ---------------------------------------------------------------------------------
# Original (pre-fix) detect_cells and matcher, reproduced VERBATIM from git HEAD /
# the historical scripts/scale_cap.py, so the "old" column in this script is the
# actual historically-buggy behaviour, not a re-description of it.
# ---------------------------------------------------------------------------------
import importlib.util

_ORIG_PATH = None  # set by _materialize_original_cellmetrics(), relative to (possibly
                   # overridden) OUT -- must not be fixed at import time


def _materialize_original_cellmetrics():
    """Write git HEAD's cellmetrics.py to disk and import it under a private name, so
    detection + the original buggy matcher run EXACTLY as they did historically (best
    available reconstruction -- see REPRODUCTION GATE above for what this can and can't
    guarantee)."""
    global _ORIG_PATH
    OUT.mkdir(parents=True, exist_ok=True)
    _ORIG_PATH = OUT / '_orig_cellmetrics.py'
    content = subprocess.run(['git', 'show', 'HEAD:volsplat/cellmetrics.py'],
                             cwd=REPO, capture_output=True, text=True, check=True).stdout
    _ORIG_PATH.write_text(content, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('_orig_cellmetrics', _ORIG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def old_match_indices(pred, tgt, radius):
    """Verbatim reproduction of scale_cap.py's original `recovered()` matching body
    (Hungarian on raw distances, then threshold) -- the historically-run behaviour."""
    from scipy.optimize import linear_sum_assignment
    n_pred, n_tgt = len(pred), len(tgt)
    if n_pred == 0 or n_tgt == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty, np.empty(0, dtype=float)
    d = np.linalg.norm(pred[:, None, :] - tgt[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(d)
    keep = d[ri, ci] <= radius
    return ri[keep], ci[keep], d[ri, ci][keep]


def snapshot_code():
    """Copy every module actually imported for fitting/scoring, plus full git status
    and diff text (not just a dirty boolean), so THIS run's provenance is airtight."""
    snap = OUT / 'code_snapshot'
    snap.mkdir(parents=True, exist_ok=True)
    for rel in ('volsplat/cellmetrics.py', 'volsplat/init.py', 'volsplat/ablation.py',
               'volsplat/gaussians.py', 'volsplat/metrics.py', 'volsplat/provenance.py',
               'volsplat/tribolium.py', 'scripts/scale_cap.py',
               'scripts/rescore_scale_cap.py'):
        src = REPO / rel
        dst = snap / rel.replace('/', '__')
        shutil.copy2(src, dst)
    shutil.copy2(_ORIG_PATH, snap / '_orig_cellmetrics.py')
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO,
                            capture_output=True, text=True).stdout
    diff = subprocess.run(['git', 'diff', 'HEAD'], cwd=REPO,
                          capture_output=True, text=True).stdout
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    (snap / 'git_status.txt').write_text(f'HEAD={commit}\n\n{status}', encoding='utf-8')
    (snap / 'git_diff_HEAD.patch').write_text(diff, encoding='utf-8')


def compare_to_historical(row, tag):
    """Point 1: does THIS run's old-matcher score reproduce the historically-recorded
    value for the same region+arm? Reports the gap; does not abort on mismatch (the
    fit is already paid for, and the old-vs-new delta within this run is still
    informative even if it doesn't equal history -- see module docstring)."""
    if not HISTORICAL_CSV.exists():
        print(f'  [reproduction check] no historical CSV found, skipping')
        return
    hist = pd.read_csv(HISTORICAL_CSV)
    m = hist[(hist.region == row['region']) & (hist.arm == row['arm'])]
    if len(m) == 0:
        print(f'  [reproduction check] no historical row for {tag}')
        return
    h = m.iloc[0]
    fields = ['S_A', 'S_B', 'recall', 'scale_kept', 'scale_lost', 'scale_B_lost',
             'psnr_nucleus', 'psnr_background', 'psnr_global']
    print(f'  [reproduction check] {tag}  (historical vs this-run, OLD matcher)')
    all_close = True
    for f in fields:
        hv, nv = float(h[f]), float(row[f])
        close = np.isclose(hv, nv, atol=1e-3, rtol=1e-3, equal_nan=True)
        all_close &= bool(close)
        flag = 'ok' if close else '**MISMATCH**'
        print(f'      {f:16s} hist={hv:8.4f}  rerun={nv:8.4f}  [{flag}]')
    print(f'  [reproduction check] {tag} -> {"REPRODUCED" if all_close else "DID NOT REPRODUCE EXACTLY"}'
         f' (fingerprint mismatch is known -- see module docstring; this quantifies its effect)')


def score_arm(gs, roi, nuclei, nmask, in_A, cellmetrics_orig, region_tag, arm, cap, res):
    recon = gs.query_volume(roi.shape).cpu().numpy().astype(np.float32)
    pred = cellmetrics_orig.detect_cells(recon, mode='3d')

    ri_old, ci_old, dist_old = old_match_indices(pred, nuclei, MATCH_R)
    got_old = np.zeros(len(nuclei), dtype=bool)
    got_old[ci_old] = True

    ri_new, ci_new, dist_new = match_indices_fixed(pred, nuclei, MATCH_R)
    got_new = np.zeros(len(nuclei), dtype=bool)
    got_new[ci_new] = True

    pos = gs.positions.detach().cpu().numpy()
    sc_full = gs.scales.detach().cpu().numpy()  # (N, 3) -- full per-axis scales
    sc = sc_full.mean(axis=1)                   # (N,) -- scalar summary, matches the
                                                  # historical scale_kept/lost convention
    amp = gs.amplitudes.detach().cpu().numpy()
    quat = gs.quaternions.detach().cpu().numpy()
    n_nuc = len(nuclei)

    def build_row(got, matcher_tag):
        return dict(
            region=region_tag, arm=arm, cap=cap, matcher=matcher_tag,
            S_A=float(got[in_A].mean()) if in_A.any() else np.nan,
            S_B=float(got[~in_A].mean()) if (~in_A).any() else np.nan,
            recall=float(got.mean()),
            scale_kept=float(np.median(sc[:n_nuc][got])) if got.any() else np.nan,
            scale_lost=float(np.median(sc[:n_nuc][~got])) if (~got).any() else np.nan,
            scale_B_lost=float(np.median(sc[:n_nuc][(~in_A) & (~got)]))
                if ((~in_A) & (~got)).any() else np.nan,
            amp_final=float(np.median(amp[:n_nuc])),
            psnr_nucleus=M.psnr(recon[nmask], roi[nmask]),
            psnr_background=M.psnr(recon[~nmask], roi[~nmask]),
            psnr_global=M.psnr(recon, roi),
            val_psnr=res['val_psnr'],
        )

    return (recon, pred, got_old, got_new, ri_old, ci_old, ri_new, ci_new, dist_new,
            pos, sc, sc_full, amp, quat, build_row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--full', action='store_true', help='all 4 arms instead of [control, medium]')
    p.add_argument('--k', type=int, default=250)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--regions', type=int, default=4, help='use first N regions only (smoke-testing)')
    p.add_argument('--outdir', type=str, default=None, help='override output dir (smoke-testing)')
    args = p.parse_args()

    global OUT
    if args.outdir:
        OUT = Path(args.outdir)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'gaussians').mkdir(exist_ok=True)
    (OUT / 'peaks').mkdir(exist_ok=True)
    (OUT / 'matches').mkdir(exist_ok=True)

    cellmetrics_orig = _materialize_original_cellmetrics()
    snapshot_code()

    arms = [('control', None), ('mild', 12.0), ('medium', 10.0), ('strict', 8.0)] if args.full \
        else [('control', None), ('medium', 10.0)]

    vol = np.load(REPO / 'runs/colleague_full_volume.npy').astype(np.float32)
    rows = []

    for (z, y, x) in REGIONS[:args.regions]:
        tag = f'z{z}y{y}x{x}'
        roi = np.ascontiguousarray(vol[z:z + SHAPE[0], y:y + SHAPE[1], x:x + SHAPE[2]])
        nuclei = cellmetrics_orig.detect_cells(roi, mode='3d')
        nmask = nucleus_mask(roi.shape, nuclei)

        old_init = init_gaussians(roi, args.k, strategy='local_maxima', init_scale=2.0, seed=0)
        p_old = old_init.positions.detach().cpu().numpy()[:, ::-1]
        d_old = np.linalg.norm(nuclei[:, None, :] - p_old[None, :, :], axis=-1).min(axis=1)
        in_A = d_old <= SEED_R
        print(f'\n=== {tag}: {len(nuclei)} nuclei (A={int(in_A.sum())}, B={int((~in_A).sum())}) ===')

        for arm, cap in arms:
            gs, _, res = fit_with_validation(
                roi, num_gaussians=args.k, iterations=args.iters,
                init_strategy='coverage', parameterization='full', init_scale=2.0,
                seed=args.seed, max_scale=cap, max_scale_n=len(nuclei),
                full_recon=False)

            (recon, pred, got_old, got_new, ri_old, ci_old, ri_new, ci_new, dist_new,
             pos, sc, sc_full, amp, quat, build_row) = score_arm(
                gs, roi, nuclei, nmask, in_A, cellmetrics_orig, tag, arm, cap, res)

            row_old = build_row(got_old, 'old_buggy')
            row_new = build_row(got_new, 'new_fixed')
            meta = run_metadata(experiment='rescore_scale_cap', arm=arm, max_scale=cap,
                                seed=args.seed, region=tag, k=args.k, iters=args.iters,
                                init='coverage', parameterization='full',
                                init_scale=2.0, lr_position=0.0016)
            rows.append({**row_old, **meta})
            rows.append({**row_new, **meta})

            # BUG this replaces (found in review): saved only `scales_mean`, a per-
            # Gaussian scalar that loses the 3 individual axis scales -- insufficient
            # to reconstruct the fitted ellipsoids. Now saves the full (N,3) array too.
            np.savez(OUT / 'gaussians' / f'{tag}_{arm}.npz',
                    positions=pos, scales=sc_full, scales_mean=sc,
                    quaternions=quat, amplitudes=amp, cap=np.array(cap if cap is not None else np.nan))
            np.save(OUT / 'peaks' / f'{tag}_{arm}_nuclei.npy', nuclei)
            np.save(OUT / 'peaks' / f'{tag}_{arm}_pred.npy', pred)
            np.savez(OUT / 'matches' / f'{tag}_{arm}_old.npz', pred_idx=ri_old, target_idx=ci_old)
            np.savez(OUT / 'matches' / f'{tag}_{arm}_new.npz', pred_idx=ri_new, target_idx=ci_new, dist=dist_new)

            pd.DataFrame(rows).to_csv(OUT / 'rescore.csv', index=False)

            n_flip = int((got_old != got_new).sum())
            print(f'  {arm:8s} cap={str(cap):5s}  OLD: S_A={row_old["S_A"]:.3f} S_B={row_old["S_B"]:.3f} '
                 f'recall={row_old["recall"]:.3f}  |  NEW: S_A={row_new["S_A"]:.3f} S_B={row_new["S_B"]:.3f} '
                 f'recall={row_new["recall"]:.3f}  |  matcher flips={n_flip}', flush=True)

            if tag == f'z{REGIONS[0][0]}y{REGIONS[0][1]}x{REGIONS[0][2]}':
                compare_to_historical(row_old, f'{tag}/{arm}')

    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

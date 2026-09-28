"""Re-ranking experiment, v2 -- closes the loose ends left by rerank_vs_baselines.py.

WHAT v1 FOUND (runs/rerank/). Seeded at the top-500 raw candidates, a Gaussian fit
recovered 13-15 of 29 human-annotated nuclei at a matched budget of N=100, against 10 for
the raw image, in all 3 seeds.

WHAT v1 GOT WRONG OR LEFT OPEN
  1. MATCHER BUG. v1's recall_at() ran linear_sum_assignment on raw distances and THEN
     applied the radius. That minimises total cost, not match count, and can drop valid
     within-radius pairs. The same bug inflated the Tribolium scale-cap effect
     (runs/scale_cap_rescore/). v2 matches with cellmetrics.match_indices (maximum
     cardinality first) and records BOTH matchers for every arm, so the bug's effect on
     v1 is measured rather than assumed. Fits are seeded (torch.manual_seed), so the
     base / lr05 arms reproduce v1's fits.
  2. NOTHING WAS SAVED. v1 kept only recall numbers. v2 saves every arm's ranked
     candidates and every fit's Gaussian parameters, so a future scoring change is a
     rescore, not a refit.
  3. THE SCALE CAP WAS NEVER TESTED ON HUMAN LABELS -- only on Tribolium, against
     detector-derived targets.
  4. THE RECOMMENDED "HIGHER LR + CAP" COMBINATION WAS NEVER RUN.
  5. SAMPLER CONFOUND. 70% of each training batch is drawn in proportion to intensity,
     with no importance correction, so the effective objective over-weights bright
     voxels beyond plain MSE. A uniform-sampling arm tests whether faint-structure loss
     is partly a sampler effect rather than a loss effect.

ARMS -- all seeded at the SAME top-K raw candidates, so only the fit differs
    base       lr 0.0016   no cap   70% intensity sampling    v1 control, rescored
    cap        lr 0.0016   cap      70%                       the cap, on HUMAN labels
    uniform    lr 0.0016   no cap    0%                       sampler confound
    lr05_cap   lr 0.05     cap      70%                       the combined recipe
    lr05       lr 0.05     no cap   70%                       learning-rate factor
Loop order is SEED-major (every arm at seed 0, then seed 1, ...), so a run stopped early
still leaves a balanced, paired design.

CAP, PRE-REGISTERED FROM TRIBOLIUM, NOT TUNED HERE. The Tribolium medium cap was 10
voxels = 6.93 um in xy. The implementation clamps all three log-scales to one VOXEL value
(itself an instance of the voxel-units bug class), so the physical xy match on DRO
(0.406 um/voxel) is round(6.93 / 0.406) = 17 voxels -- 34.5 um along z, i.e. the cap
never binds along z, as on Tribolium (30 um). As on Tribolium, only the candidate-seeded
Gaussians are capped (max_scale_n = number of seeds); the fill Gaussians stay free.
Whether the cap actually BINDS is reported as a manipulation check.

RESOLUTION, STATED BEFORE THE RUN: 29 annotated nuclei, so one nucleus = 3.4 recall
points. Paired differences under ~3 nuclei, or not sign-consistent across seeds, are not
resolvable in this crop.

Saved Gaussian positions/scales are in the library's (x, y, z) order; candidate lists
are (z, y, x) voxel indices, strongest first.

Usage:
    python scripts/rerank_factorial.py --k 500 --iters 3000 --seeds 0 1 2
    python scripts/rerank_factorial.py --summarize-only     # re-summarise the saved CSV
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import zoom
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from volsplat.ctc import percentile_normalize
from volsplat.cellmetrics import detect_cells, match_indices
from volsplat import metrics as M

OUT = REPO / 'runs/rerank_factorial'
DRO = Path(r'D:/Download/train/Fluo-N3DL-DRO (1)/Fluo-N3DL-DRO')
SHAPE = (64, 128, 128)
ORIGIN = (32, 416, 544)                     # v1's placement holding the most nuclei
V = np.array([2.03, 0.406, 0.406])          # um per voxel (z, y, x)
MATCH_UM = 3.0
PROMINENCE = 0.02
SCALARS_PER_GAUSSIAN = 11
TRIB_CAP_UM_XY = 10 * 0.6934                # Tribolium medium cap, physical xy extent
CAP_VOX = int(round(TRIB_CAP_UM_XY / V[1]))  # = 17 on DRO
BUDGETS = [29, 50, 100, 200, 300, 500]

ARMS = [
    dict(name='base',     lr=0.0016, cap=None,    bias=0.7),
    dict(name='cap',      lr=0.0016, cap=CAP_VOX, bias=0.7),
    dict(name='uniform',  lr=0.0016, cap=None,    bias=0.0),
    dict(name='lr05_cap', lr=0.05,   cap=CAP_VOX, bias=0.7),
    dict(name='lr05',     lr=0.05,   cap=None,    bias=0.7),
]


def detect(vol):
    """Ranked candidates, strongest first, physically isotropic window."""
    return detect_cells(vol, mode='3d', threshold_abs=0.0, smoothing_sigma=0.8,
                        min_distance=2.0, prominence=PROMINENCE,
                        voxel_size_zyx=tuple(V))


def recall_fixed(pred_zyx, manual_zyx, n):
    top = pred_zyx[:n]
    if len(top) == 0:
        return 0.0, 0
    ri, _, _ = match_indices(top * V, manual_zyx * V, MATCH_UM)
    return len(ri) / len(manual_zyx), int(len(ri))


def recall_v1(pred_zyx, manual_zyx, n):
    """v1's recall_at, verbatim (Hungarian, THEN threshold). Kept only to measure how
    far the bug moved v1's numbers."""
    pred = pred_zyx[:n]
    if len(pred) == 0 or len(manual_zyx) == 0:
        return 0.0, 0
    d = np.linalg.norm((manual_zyx * V)[:, None, :] - (pred * V)[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(d)
    matched = int(sum(1 for a, b in zip(ri, ci) if d[a, b] <= MATCH_UM))
    return matched / len(manual_zyx), matched


def downsample_matched(vol, n_scalars):
    best = None
    for f in range(2, 17):
        dims = tuple(max(1, s // f) for s in vol.shape)
        n = int(np.prod(dims))
        if best is None or abs(n - n_scalars) < abs(best[1] - n_scalars):
            best = (f, n, dims)
    f, n, dims = best
    coarse = zoom(vol, tuple(d / s for d, s in zip(dims, vol.shape)), order=1)
    back = zoom(coarse, tuple(s / c for s, c in zip(vol.shape, coarse.shape)), order=1)
    return back[:vol.shape[0], :vol.shape[1], :vol.shape[2]].astype(np.float32), f, n


def score_row(cand, manual, n):
    r, m = recall_fixed(cand, manual, n)
    r1, m1 = recall_v1(cand, manual, n)
    return dict(N=n, recall=r, matched=m, recall_v1_matcher=r1, matched_v1_matcher=m1,
                n_cand=len(cand))


def summarize(df, meta):
    n_man = meta['n_manual']
    piv = df.pivot_table(index='arm', columns='N', values='matched', aggfunc='mean')
    print(f'\n=== matched nuclei (of {n_man}), mean over seeds, fixed matcher ===')
    print(piv.round(2).to_string())
    piv1 = df.pivot_table(index='arm', columns='N', values='matched_v1_matcher',
                          aggfunc='mean')
    print('\n=== same, v1 matcher (for comparison with runs/rerank) ===')
    print(piv1.round(2).to_string())
    ncand = df.groupby('arm').n_cand.mean()
    nseed = df[df.seed >= 0].groupby('arm').seed.nunique()
    print('\ncandidates supplied: ' + ', '.join(f'{a} {ncand[a]:.0f}' for a in ncand.index))
    print('seeds completed: ' + ', '.join(f'{a} {v}' for a, v in nseed.items()))

    g = df[df.seed >= 0]
    raw_m = df[df.arm == 'raw'].set_index('N').matched

    def paired(a, b, n):
        """Per-seed matched(a) - matched(b) at budget n; b may be 'raw' (deterministic)."""
        ga = g[(g.arm == a) & (g.N == n)].set_index('seed').matched
        if b == 'raw':
            return (ga - raw_m[n]).values
        gb = g[(g.arm == b) & (g.N == n)].set_index('seed').matched
        common = ga.index.intersection(gb.index)
        return (ga[common] - gb[common]).values

    contrasts = [
        ('fit vs raw image', 'base', 'raw'),
        ('cap vs none, lr 0.0016', 'cap', 'base'),
        ('uniform vs 70% intensity', 'uniform', 'base'),
        ('lr 0.05 vs 0.0016', 'lr05', 'base'),
        ('cap vs none, lr 0.05', 'lr05_cap', 'lr05'),
        ('combined recipe vs base', 'lr05_cap', 'base'),
    ]
    summary = dict(meta, contrasts={})
    present = set(g.arm)
    print('\n=== paired contrasts, in NUCLEI (resolution ~3, need sign consistency) ===')
    for label, a, b in contrasts:
        if a not in present or (b != 'raw' and b not in present):
            continue
        line, rec = [], {}
        for n in (100, 200):
            d = paired(a, b, n)
            if len(d) == 0:
                continue
            sign = ('all +' if (d > 0).all() else 'all -' if (d < 0).all() else
                    'all 0' if (d == 0).all() else 'mixed')
            rec[int(n)] = {'per_seed': [int(v) for v in d], 'mean': float(d.mean()),
                           'sign': sign}
            line.append(f'N={n}: {[int(v) for v in d]} mean {d.mean():+.2f} ({sign})')
        summary['contrasts'][label] = rec
        print(f'  {label:26s} ' + '   '.join(line))

    capped = [a for a in ('cap', 'lr05_cap') if a in present]
    if capped:
        fc = g[g.arm.isin(capped)].groupby('arm').frac_at_cap.mean()
        print('\nmanipulation check, fraction of seeded Gaussians at the cap: ' +
              ', '.join(f'{a} {v:.0%}' for a, v in fc.items()))
        summary['frac_at_cap'] = {k: float(v) for k, v in fc.items()}
    if len(g):
        ms = g.groupby('arm').median_seeded_max_scale.mean()
        print('median seeded max-scale (voxels): ' +
              ', '.join(f'{a} {v:.1f}' for a, v in ms.items()))
        vp = g.groupby('arm').val_psnr.mean()
        print('held-out val PSNR (dB): ' + ', '.join(f'{a} {v:.2f}' for a, v in vp.items()))
        summary['median_seeded_max_scale'] = {k: float(v) for k, v in ms.items()}
        summary['val_psnr'] = {k: float(v) for k, v in vp.items()}
    summary['matched_mean'] = {a: {int(n): float(v) for n, v in piv.loc[a].items()}
                               for a in piv.index}
    summary['matched_mean_v1_matcher'] = {a: {int(n): float(v) for n, v in
                                              piv1.loc[a].items()} for a in piv1.index}
    summary['n_cand'] = {k: float(v) for k, v in ncand.items()}
    summary['seeds_completed'] = {k: int(v) for k, v in nseed.items()}
    with open(OUT / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'\nOutputs -> {OUT}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=500)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--arms', nargs='+', default=[a['name'] for a in ARMS])
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--resume', action='store_true',
                   help='keep finished (arm, seed) fits from the saved CSV and skip them')
    args = p.parse_args()
    for sub in ('candidates', 'gaussians'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    cents = np.load(REPO / 'runs/dro_manual_centroids.npy')
    z, y, x = ORIGIN
    inside = ((cents[:, 0] >= z) & (cents[:, 0] < z + SHAPE[0]) &
              (cents[:, 1] >= y) & (cents[:, 1] < y + SHAPE[1]) &
              (cents[:, 2] >= x) & (cents[:, 2] < x + SHAPE[2]))
    manual = cents[inside] - np.array(ORIGIN)
    n_man = len(manual)
    meta = {'origin': list(ORIGIN), 'n_manual': int(n_man), 'cap_vox': CAP_VOX,
            'k': args.k, 'iters': args.iters, 'seeds': args.seeds}

    if args.summarize_only:
        summarize(pd.read_csv(OUT / 'rerank_factorial.csv'), meta)
        return

    from volsplat.ablation import fit_with_validation
    vol_full = percentile_normalize(
        tifffile.imread(str(DRO / '01/t000.tif')).astype(np.float32))
    roi = np.ascontiguousarray(vol_full[z:z + SHAPE[0], y:y + SHAPE[1], x:x + SHAPE[2]])
    print(f'ROI {ORIGIN} {roi.shape} | {n_man} HUMAN-annotated nuclei | '
          f'1 nucleus = {100 / n_man:.1f} recall points')
    print(f'pre-registered cap: {CAP_VOX} voxels (= {CAP_VOX * V[1]:.2f} um xy, '
          f'{CAP_VOX * V[0]:.1f} um z)\n', flush=True)

    rows = []
    done_rows = []
    if args.resume and (OUT / 'rerank_factorial.csv').exists():
        prev = pd.read_csv(OUT / 'rerank_factorial.csv')
        done_rows = prev[prev.seed >= 0].to_dict('records')
    done = {(r['arm'], int(r['seed'])) for r in done_rows}
    if done:
        print(f'resuming: {len(done)} fits already done: {sorted(done)}', flush=True)

    def save_rows():
        pd.DataFrame(rows).to_csv(OUT / 'rerank_factorial.csv', index=False)

    # ---- deterministic arms
    raw = detect(roi)
    np.save(OUT / 'candidates' / 'raw.npy', raw)
    ds, f, n_ds = downsample_matched(roi, args.k * SCALARS_PER_GAUSSIAN)
    dsc = detect(ds)
    np.save(OUT / 'candidates' / f'downsample_x{f}.npy', dsc)
    print('matched nuclei, v1 matcher -> fixed matcher:')
    for arm_name, cand in (('raw', raw), (f'downsample_x{f}', dsc)):
        cells = []
        for n in BUDGETS:
            row = score_row(cand, manual, n)
            rows.append(dict(arm=arm_name, seed=-1, **row))
            cells.append(f"@{n} {row['matched_v1_matcher']}->{row['matched']}")
        print(f'  {arm_name:16s} ({len(cand)} cand)  ' + '  '.join(cells), flush=True)
    rows.extend(done_rows)
    save_rows()

    # ---- Gaussian arms, all seeded at the same top-K raw candidates, SEED-major
    seed_pts = raw[:args.k]
    n_seed = len(seed_pts)
    meta['n_seeded'] = int(n_seed)
    print(f'\nseeding {n_seed} Gaussians at the top-{args.k} RAW candidates '
          f'(detector output, not annotation); {args.k - n_seed} free fill\n', flush=True)
    specs = [a for a in ARMS if a['name'] in args.arms]
    for sd in args.seeds:
        for spec in specs:
            if (spec['name'], sd) in done:
                continue
            gs, _, res = fit_with_validation(
                roi, num_gaussians=args.k, iterations=args.iters,
                init_strategy='oracle_coverage', init_kwargs={'nuclei': seed_pts},
                parameterization='full', init_scale=2.0, seed=sd,
                lr_position=spec['lr'], intensity_bias=spec['bias'],
                max_scale=spec['cap'], max_scale_n=n_seed if spec['cap'] else None,
                full_recon=False)
            recon = gs.query_volume(roi.shape).cpu().numpy().astype(np.float32)
            cand = detect(recon)
            sc = gs.scales.detach().cpu().numpy()
            seeded_max = sc[:n_seed].max(axis=1)
            at_cap = (float((seeded_max >= spec['cap'] * 0.999).mean())
                      if spec['cap'] else np.nan)
            np.save(OUT / 'candidates' / f"{spec['name']}_s{sd}.npy", cand)
            np.savez(OUT / 'gaussians' / f"{spec['name']}_s{sd}.npz",
                     positions_xyz=gs.positions.detach().cpu().numpy(), scales_xyz=sc,
                     quaternions=gs.quaternions.detach().cpu().numpy(),
                     amplitudes=gs.amplitudes.detach().cpu().numpy(), n_seeded=n_seed)
            psnr_full = float(M.psnr(recon, roi))
            cells = []
            for n in BUDGETS:
                row = score_row(cand, manual, n)
                rows.append(dict(arm=spec['name'], seed=sd, **row, lr=spec['lr'],
                                 cap=spec['cap'], intensity_bias=spec['bias'],
                                 frac_at_cap=at_cap,
                                 median_seeded_max_scale=float(np.median(seeded_max)),
                                 val_psnr=res['val_psnr'], psnr_full=psnr_full,
                                 fit_seconds=res.get('fit_seconds')))
                cells.append(f"@{n}={row['matched']}")
            save_rows()
            cap_note = f', {at_cap:.0%} at cap' if spec['cap'] else ''
            print(f"  s{sd} {spec['name']:9s} matched " + ' '.join(cells) +
                  f"  ({len(cand)} cand, val {res['val_psnr']:.2f} dB, median seeded "
                  f"max-scale {np.median(seeded_max):.1f} vox{cap_note})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'rerank_factorial.csv', index=False)
    summarize(df, meta)


if __name__ == '__main__':
    main()

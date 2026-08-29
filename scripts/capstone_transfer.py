"""The two remaining tests, answered together, against HUMAN annotation.

Everything below is scored on the 29 manually annotated nuclei inside a Drosophila ROI,
so no target is produced by our own detector. That removes the circularity that
invalidated the Tribolium coverage result.

TEST 1 -- cross-dataset transfer of the migration finding.
    On Tribolium, raising the position learning rate took migration 1% -> 32% and closed
    the initialization gap from +0.207 to +0.001 F1. Tribolium spatial rules have already
    failed to transfer once (the NMS-spacing rule), so this must be re-tested. DRO voxels
    are (2.03, 0.406, 0.406) um vs Tribolium (3.0, 0.6934, 0.6934), and the position
    learning rate is expressed in VOXELS, so the same numeric lr means a different
    physical step. Both a voxel-matched and a physically-matched lr are therefore tested.

TEST 2 -- do better candidates become a better FITTED result?
    Candidate ranking was validated as a DETECTION property. Detection quality is not
    automatically fitted-representation quality. Comparing a good initializer against a
    random one, scored on manual labels after fitting, tests whether it survives the fit.

Matching is in MICRONS throughout, and detection uses an offset-invariant prominence
criterion, per scripts/test_metrics.py.

Usage:
    python scripts/capstone_transfer.py --k 250 --iters 3000 --seeds 0 1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment

from volsplat.ctc import percentile_normalize
from volsplat.init import init_gaussians
from volsplat.ablation import fit_with_validation
from volsplat.cellmetrics import detect_cells

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/capstone'
DRO = Path(r'D:/Download/train/Fluo-N3DL-DRO (1)/Fluo-N3DL-DRO')
SHAPE = (64, 128, 128)
V_DRO = np.array([2.03, 0.406, 0.406])       # um per voxel (z, y, x)
V_TRIB = np.array([3.0, 0.6934, 0.6934])
MATCH_UM = 3.0
PROMINENCE = 0.02


def score_vs_manual(recon, manual_zyx):
    """Recall / precision / F1 of the reconstruction against HUMAN centroids, in um."""
    # min_distance / smoothing are MICRONS here, and the window is sized per axis.
    # The legacy cubic 13-voxel default recovers only 5/29 manual nuclei from the RAW
    # TARGET on this data (recall 0.172), so it caps any score computed with it.
    pred = detect_cells(recon, mode='3d', threshold_abs=0.0, smoothing_sigma=0.8,
                        min_distance=2.0, prominence=PROMINENCE,
                        voxel_size_zyx=tuple(V_DRO))
    if len(pred) == 0 or len(manual_zyx) == 0:
        return dict(recall=0.0, precision=0.0, f1=0.0, n_pred=int(len(pred)))
    d = np.linalg.norm((manual_zyx * V_DRO)[:, None, :] - (pred * V_DRO)[None, :, :],
                       axis=-1)
    ri, ci = linear_sum_assignment(d.T if d.shape[0] > d.shape[1] else d)
    matched = 0
    for a, b in zip(ri, ci):
        dist = d[b, a] if d.shape[0] > d.shape[1] else d[a, b]
        if dist <= MATCH_UM:
            matched += 1
    rec = matched / len(manual_zyx)
    prec = matched / len(pred)
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0.0
    return dict(recall=float(rec), precision=float(prec), f1=float(f1),
                n_pred=int(len(pred)), matched=int(matched))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=250)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    z, y, x = np.load(REPO / 'runs/dro_roi_origin.npy')
    man_all = np.load(REPO / 'runs/dro_manual_centroids.npy')
    vol = percentile_normalize(tifffile.imread(str(DRO / '01/t000.tif')).astype(np.float32))
    roi = np.ascontiguousarray(vol[z:z + SHAPE[0], y:y + SHAPE[1], x:x + SHAPE[2]])
    m = ((man_all[:, 0] >= z) & (man_all[:, 0] < z + SHAPE[0]) &
         (man_all[:, 1] >= y) & (man_all[:, 1] < y + SHAPE[1]) &
         (man_all[:, 2] >= x) & (man_all[:, 2] < x + SHAPE[2]))
    manual = man_all[m] - np.array([z, y, x])
    print(f'DRO ROI origin ({z},{y},{x})  {len(manual)} MANUALLY annotated nuclei')

    # a physically-matched lr: the same step in microns as Tribolium's winning 0.05
    # (Tribolium xy voxel 0.6934 um -> 0.05 vox = 0.03467 um; DRO xy voxel 0.406 um)
    lr_phys = 0.05 * V_TRIB[1] / V_DRO[1]
    lrs = [0.0016, 0.05, round(lr_phys, 4)]
    print(f'position lrs tested: {lrs}  (last = physically matched to Tribolium 0.05)\n')

    rows = []
    total = 2 * len(lrs) * len(args.seeds)
    n = 0
    for init in ['random', 'coverage']:
        for lr in lrs:
            for seed in args.seeds:
                n += 1
                gs0 = init_gaussians(roi, args.k, strategy=init, init_scale=2.0, seed=seed)
                p0 = gs0.positions.detach().cpu().numpy()
                gs, _, res = fit_with_validation(
                    roi, num_gaussians=args.k, iterations=args.iters,
                    init_strategy=init, parameterization='full', init_scale=2.0,
                    seed=seed, lr_position=lr, track_displacement=True,
                    full_recon=False)
                p1 = gs.positions.detach().cpu().numpy()
                recon = gs.query_volume(roi.shape).cpu().numpy().astype(np.float32)
                b = float(res.get('background_b') or 0.0)
                sc = score_vs_manual(recon + b, manual)

                # migration against MANUAL nuclei
                mn = manual[:, ::-1].astype(np.float32)          # -> (x, y, z)
                a0 = np.linalg.norm(p0[:, None, :] - mn[None, :, :], axis=-1).argmin(1)
                a1 = np.linalg.norm(p1[:, None, :] - mn[None, :, :], axis=-1).argmin(1)
                migrated = float((a0 != a1).mean())

                row = dict(init=init, lr=lr, seed=seed, migrated=migrated,
                           disp=res['disp_mean'], val_psnr=res['val_psnr'], **sc)
                rows.append(row)
                print(f'[{n}/{total}] {init:9s} lr={lr:<7g} s={seed}  '
                      f'migrated={migrated*100:5.1f}%  moved={res["disp_mean"]:5.2f}  '
                      f'recall={sc["recall"]:.3f}  F1={sc["f1"]:.3f}  '
                      f'nPred={sc["n_pred"]}', flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'capstone.csv', index=False)
    g = df.groupby(['init', 'lr']).agg(
        migrated=('migrated', 'mean'), disp=('disp', 'mean'),
        recall=('recall', 'mean'), f1=('f1', 'mean'),
        precision=('precision', 'mean'), n_pred=('n_pred', 'mean')).reset_index()
    print('\n=== scored against HUMAN annotation ===')
    print(g.round(3).to_string(index=False))

    print('\n=== TEST 1: does the migration finding transfer to Drosophila? ===')
    for lr in lrs:
        sub = g[g.lr == lr]
        if len(sub) == 2:
            gap = float(sub[sub.init == 'coverage'].f1.values[0]
                        - sub[sub.init == 'random'].f1.values[0])
            mig = float(sub[sub.init == 'random'].migrated.values[0])
            print(f'  lr={lr:<7g}  migration(random)={mig*100:5.1f}%   '
                  f'init gap (coverage - random) F1 = {gap:+.3f}')
    print('\n  Tribolium reference: migration 1.0% -> 32.0%, gap +0.207 -> +0.001')

    print('\n=== TEST 2: does a better initializer survive the fit? ===')
    for lr in lrs:
        sub = g[g.lr == lr]
        if len(sub) == 2:
            r = float(sub[sub.init == 'random'].f1.values[0])
            c = float(sub[sub.init == 'coverage'].f1.values[0])
            print(f'  lr={lr:<7g}  random F1={r:.3f}   coverage F1={c:.3f}   '
                  f'{"coverage helps" if c > r + 0.02 else "no clear benefit"}')

    json.dump(g.to_dict('records'), open(OUT / 'capstone_summary.json', 'w'), indent=2)
    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

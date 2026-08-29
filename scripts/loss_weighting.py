"""Causal test: are faint nuclei lost because voxelwise MSE under-weights them?

BACKGROUND. S_B (survival of faint, newly-covered nuclei) is 0.316-0.387 against
S_A ~ 0.83 for the rest. Removing the DC pedestal did NOT fix it -- a fixed scalar
background made S_B WORSE (0.382 -> 0.316) and freed no capacity (K_support 220.5 ->
217.25), so background competition is eliminated as the cause in its scalar form.

The remaining leading hypothesis is the objective itself. At contrast 0.03 a totally
missed nucleus costs ~9e-4 per voxel over ~5% of the volume; a 0.05 error on the bright
bulk costs ~2.5e-3 per voxel over ~95%. MSE has a rational incentive to ignore faint
structure -- roughly a 50x asymmetry.

INTERVENTION. Change ONLY how reconstruction errors are weighted. The data,
initialization, representation, background handling, position LR and iteration count are
untouched, so nothing else can explain a difference.

    A  mse                 control
    B  log                 || log(1+aI) - log(1+aI_hat) ||^2, compresses the bright end
    C  contrast_weighted   w(x)(I-I_hat)^2, w ~ 1/local_std, normalized to unit median
                           and CLIPPED to [0.25, 4] -- an unbounded 1/sigma would hand
                           near-flat regions absurd weight and manufacture a new artefact

PRE-REGISTERED (written before the run):
    S_B rises substantially   > +0.05
    S_A approximately held    |delta S_A| < 0.05

INTERPRETATION
    S_B up, S_A held      -> the failure is OBJECTIVE WEIGHTING, not capacity. Clean
                             result, and the smooth B(x) model becomes unnecessary.
    S_B flat              -> MSE weighting is not the main cause; spatial background
                             becomes the next test.
    S_B up, S_A collapses -> the optimization trade-off moved rather than the failure
                             being fixed. Informative, but not a rescue.

The positive-amplitude constraint is deliberately NOT varied here: changing two things
at once would make any improvement unattributable.

Usage:
    python scripts/loss_weighting.py --regions 4 --k 250 --iters 3000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from volsplat.init import init_gaussians
from volsplat.ablation import fit_with_validation
from volsplat.cellmetrics import detect_cells
from volsplat.provenance import run_metadata
from volsplat import metrics as M

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/loss_weighting'
SHAPE = (64, 128, 128)
REGIONS = [(0, 528, 144), (0, 624, 144), (0, 240, 336), (0, 528, 240)]
MATCH_R = 0.765 * 13
OBJ_R = 5.0
NUC_VOXEL_R = 8.0
# contrast_weighted is dropped from the replication: three regions already showed it
# HURTS (S_B 0.443 -> 0.228), and spending a third of the compute re-confirming a
# negative control is worse value than seeds on the comparison that matters.
ARMS = ['mse', 'log']


def nucleus_voxel_mask(shape, nuclei, radius=NUC_VOXEL_R):
    D, H, W = shape
    mask = np.zeros(shape, dtype=bool)
    r = int(np.ceil(radius))
    for z, y, x in nuclei.astype(int):
        z0, z1 = max(0, z - r), min(D, z + r + 1)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
        mask[z0:z1, y0:y1, x0:x1] |= (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return mask


def recovered(recon, nuclei):
    """Which target nuclei are recovered in this reconstruction."""
    pred = detect_cells(recon, mode='3d')
    got = np.zeros(len(nuclei), dtype=bool)
    if len(pred) and len(nuclei):
        d = np.linalg.norm(pred[:, None, :] - nuclei[None, :, :], axis=-1)
        ri, ci = linear_sum_assignment(d)
        for a, b in zip(ri, ci):
            if d[a, b] <= MATCH_R:
                got[b] = True
    return got, len(pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--regions', type=int, default=4)
    p.add_argument('--k', type=int, default=250)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    vol = np.load(REPO / 'runs/colleague_full_volume.npy').astype(np.float32)

    rows = []
    for (z, y, x) in REGIONS[:args.regions]:
        tag = f'z{z}y{y}x{x}'
        roi = np.ascontiguousarray(vol[z:z + SHAPE[0], y:y + SHAPE[1], x:x + SHAPE[2]])
        nuclei = detect_cells(roi, mode='3d')
        nmask = nucleus_voxel_mask(roi.shape, nuclei)

        # A / B groups defined exactly as in the Stage 3 and pedestal experiments:
        # A = already covered by the OLD initializer, B = only covered once suppression
        # was widened. B is the faint population under study.
        old = init_gaussians(roi, args.k, strategy='local_maxima', init_scale=2.0, seed=0)
        p_old = old.positions.detach().cpu().numpy()[:, ::-1]
        d_old = np.linalg.norm(nuclei[:, None, :] - p_old[None, :, :], axis=-1).min(axis=1)
        group_A = d_old < 3.0

        print(f'\n=== {tag}: {len(nuclei)} nuclei  (A={int(group_A.sum())}, '
              f'B={int((~group_A).sum())}) ===')

        for arm in ARMS:
            for seed in args.seeds:
                gs, _, res = fit_with_validation(
                    roi, num_gaussians=args.k, iterations=args.iters,
                    init_strategy='coverage', parameterization='full', init_scale=2.0,
                    seed=seed, loss_mode=arm, full_recon=False)
                recon = gs.query_volume(roi.shape).cpu().numpy().astype(np.float32)
                got, n_pred = recovered(recon, nuclei)

                pos = gs.positions.detach().cpu().numpy()
                scales = gs.scales.detach().cpu().numpy()
                dmat = np.linalg.norm(pos[:, None, :] - nuclei[:, ::-1][None, :, :],
                                      axis=-1)
                dn = dmat.min(axis=1)
                k_obj = int((dn <= OBJ_R).sum())
                prec = got.sum() / max(n_pred, 1)
                rec = float(got.mean())
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)

                # ---- THE MEDIATOR LINK. The scale-cap experiment shows lost nuclei
                # carry systematically LARGER Gaussians than kept ones, and that capping
                # scale raises S_B. If the log loss ALSO suppresses that broadening, the
                # two interventions act on one pathway:
                #     MSE weighting -> scale broadening -> local peak lost
                # For each nucleus take the scale of its nearest Gaussian, then split by
                # whether that nucleus survived.
                near = dmat.argmin(axis=0)                    # nearest Gaussian per nucleus
                nuc_scale = scales[near].mean(axis=1)         # mean of the 3 axis scales
                inB = ~group_A
                s_keptB = float(np.mean(nuc_scale[inB & got])) if (inB & got).any() else np.nan
                s_lostB = float(np.mean(nuc_scale[inB & ~got])) if (inB & ~got).any() else np.nan
                s_kept = float(np.mean(nuc_scale[got])) if got.any() else np.nan
                s_lost = float(np.mean(nuc_scale[~got])) if (~got).any() else np.nan

                row = dict(region=tag, loss=arm, seed=seed,
                           S_A=float(got[group_A].mean()) if group_A.any() else np.nan,
                           S_B=float(got[~group_A].mean()) if (~group_A).any() else np.nan,
                           recall=rec, precision=float(prec), f1=float(f1),
                           K_object=k_obj, K_support=args.k - k_obj,
                           scale_kept=s_kept, scale_lost=s_lost,
                           scale_keptB=s_keptB, scale_lostB=s_lostB,
                           scale_median=float(np.median(nuc_scale)),
                           psnr_nucleus=M.psnr(recon[nmask], roi[nmask]),
                           psnr_background=M.psnr(recon[~nmask], roi[~nmask]),
                           psnr_global=M.psnr(recon, roi),
                           val_psnr=res['val_psnr'])
                row.update(run_metadata(
                    experiment='loss_weighting', loss_mode=arm, seed=seed, region=tag,
                    k=args.k, iters=args.iters, init='coverage',
                    parameterization='full', init_scale=2.0, lr_position=0.0016))
                rows.append(row)
                # persist after EVERY fit -- the previous run was killed at 11/12 and
                # wrote nothing, because the CSV was only saved at the end.
                pd.DataFrame(rows).to_csv(OUT / 'loss_weighting.csv', index=False)
                print(f'  {arm:6s} s={seed}  S_A={row["S_A"]:.3f} S_B={row["S_B"]:.3f} '
                      f'rec={rec:.3f} prec={prec:.3f} F1={f1:.3f} '
                      f'scale kept/lost={s_kept:.2f}/{s_lost:.2f} '
                      f'(B: {s_keptB:.2f}/{s_lostB:.2f})', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'loss_weighting.csv', index=False)

    g = df.groupby('loss').agg(
        S_A=('S_A', 'mean'), S_B=('S_B', 'mean'), recall=('recall', 'mean'),
        precision=('precision', 'mean'), f1=('f1', 'mean'),
        K_support=('K_support', 'mean'), scale_keptB=('scale_keptB', 'mean'),
        scale_lostB=('scale_lostB', 'mean'), scale_median=('scale_median', 'mean'),
        psnr_nucleus=('psnr_nucleus', 'mean'),
        psnr_global=('psnr_global', 'mean')).reindex(ARMS)
    print('\n=== LOSS WEIGHTING (mean over regions/seeds) ===')
    print(g.round(3).to_string())

    # per-seed S_B, so the effect can be judged against seed noise rather than asserted
    print('\n=== per-seed S_B (pooled over regions) ===')
    ps = df.groupby(['loss', 'seed']).S_B.mean().unstack()
    print(ps.round(3).to_string())
    sd = df.groupby(['loss', 'seed']).S_B.mean().groupby('loss').std()
    print('  between-seed std: ' + ', '.join(f'{k}={v:.3f}' for k, v in sd.items()))

    ref = g.loc['mse']
    print('\n=== PRE-REGISTERED CRITERIA (vs mse control) ===')
    verdicts = {}
    for arm in ARMS[1:]:
        a = g.loc[arm]
        dS_B, dS_A = a.S_B - ref.S_B, a.S_A - ref.S_A
        dP = a.precision - ref.precision
        c1, c2 = dS_B > 0.05, abs(dS_A) < 0.05
        # Added after the first run: the rescue there came with enough extra false
        # positives that overall F1 stayed flat. A rescue that only trades recall for
        # precision is not the same claim, so precision is now reported as a named
        # endpoint rather than discovered afterwards.
        c3 = dP > -0.05
        if c1 and c2 and c3:
            v = 'OBJECTIVE WEIGHTING IS THE CAUSE'
        elif c1 and c2:
            v = 'S_B rescued BUT precision falls -- a trade, not a clean rescue'
        elif c1:
            v = 'trade-off moved, not a clean rescue'
        else:
            v = 'NOT confirmed'
        verdicts[arm] = v
        print(f'  {arm:8s} dS_B={dS_B:+.3f} {"PASS" if c1 else "FAIL"}   '
              f'dS_A={dS_A:+.3f} {"PASS" if c2 else "FAIL"}   '
              f'dPrec={dP:+.3f} {"PASS" if c3 else "FAIL"}')
        print(f'           -> {v}')
        # the mediator question
        b_broad_ref = ref.scale_lostB - ref.scale_keptB
        b_broad_arm = a.scale_lostB - a.scale_keptB
        print(f'           B-nucleus scale broadening (lost - kept): '
              f'mse {b_broad_ref:+.2f} -> {arm} {b_broad_arm:+.2f}'
              + ('  [broadening REDUCED -> same pathway as the scale cap]'
                 if b_broad_arm < b_broad_ref else '  [broadening NOT reduced]'))

    json.dump({'summary': g.reset_index().to_dict('records'), 'verdicts': verdicts},
              open(OUT / 'summary.json', 'w'), indent=2)
    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

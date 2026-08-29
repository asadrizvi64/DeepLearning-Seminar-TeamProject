"""Stage 0: what does one Gaussian actually represent?

A region holding ~30 nuclei is fitted with K=250 Gaussians and only ~11% of seeds sit on
a nucleus, so the pool is implicitly `K = K_object + K_support` rather than one Gaussian
per nucleus. This script tests whether that split is real, and whether the project's
premise -- one Gaussian <-> one nucleus -- actually holds.

Three measurements, in order of importance:

1. n_j -- HOW MANY fitted Gaussians are associated with each nucleus.
   n_j ~ 1  ->  G_i <-> N_j, the premise holds
   n_j > 1  ->  {G_i1, G_i2, ...} <-> N_j; identity belongs to a GROUP, which
                complicates tracking substantially

2. Role ablation. Reconstruct all / object-only / support-only and score each on
   PSNR restricted to NUCLEUS voxels, PSNR restricted to BACKGROUND voxels, and cell F1.
   Global PSNR alone hides which population is doing which job.

3. Threshold stability. The OBJECT/SUPPORT label depends on an arbitrary radius, so
   everything is recomputed at 3, 5 and 8 voxels. If conclusions move with the radius,
   the decomposition is not stable and should not be relied on.

Usage:
    python scripts/gaussian_roles.py --regions 2 --k 250 --iters 3000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from volsplat.gaussians import GaussianSet
from volsplat.ablation import fit_with_validation
from volsplat.cellmetrics import detect_cells, score_cells
from volsplat import metrics as M

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/gaussian_roles'
SHAPE = (64, 128, 128)
REGIONS = [(0, 528, 144), (0, 624, 144), (0, 240, 336), (0, 528, 240)]
RADII = [3.0, 5.0, 8.0]
NUCLEUS_VOXEL_RADIUS = 8.0     # defines which VOXELS count as nucleus territory


def subset(gs: GaussianSet, mask: np.ndarray) -> GaussianSet:
    idx = torch.from_numpy(np.where(mask)[0]).long()
    out = GaussianSet(torch.zeros(len(idx), 3), torch.ones(len(idx), 3))
    out.positions = nn.Parameter(gs.positions.detach()[idx].clone())
    out.log_scales = nn.Parameter(gs.log_scales.detach()[idx].clone())
    out.quaternions = nn.Parameter(gs.quaternions.detach()[idx].clone())
    out.amp_logits = nn.Parameter(gs.amp_logits.detach()[idx].clone())
    return out


def nucleus_voxel_mask(shape, nuclei_zyx, radius=NUCLEUS_VOXEL_RADIUS):
    """Boolean mask of voxels within `radius` of any nucleus centre."""
    D, H, W = shape
    mask = np.zeros(shape, dtype=bool)
    r = int(np.ceil(radius))
    for z, y, x in nuclei_zyx.astype(int):
        z0, z1 = max(0, z - r), min(D, z + r + 1)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
        d2 = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2
        mask[z0:z1, y0:y1, x0:x1] |= d2 <= radius ** 2
    return mask


def masked_psnr(pred, target, mask):
    if mask.sum() == 0:
        return float('nan')
    return M.psnr(pred[mask], target[mask])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--regions', type=int, default=2)
    p.add_argument('--k', type=int, default=250)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--init', default='coverage')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    vol = np.load(REPO / 'runs/colleague_full_volume.npy').astype(np.float32)

    ablation_rows, nj_rows, stability_rows = [], [], []

    for (z, y, x) in REGIONS[:args.regions]:
        tag = f'z{z}y{y}x{x}'
        roi = np.ascontiguousarray(vol[z:z + SHAPE[0], y:y + SHAPE[1], x:x + SHAPE[2]])
        nuclei = detect_cells(roi, mode='3d')
        nuc_mask = nucleus_voxel_mask(roi.shape, nuclei)
        bg_mask = ~nuc_mask

        gs, _, _ = fit_with_validation(
            roi, num_gaussians=args.k, iterations=args.iters,
            init_strategy=args.init, parameterization='full', init_scale=2.0,
            seed=args.seed, full_recon=False)

        pos = gs.positions.detach().cpu().numpy()          # (x, y, z)
        nuc_xyz = nuclei[:, ::-1].astype(np.float32)
        dmat = np.linalg.norm(pos[:, None, :] - nuc_xyz[None, :, :], axis=-1)
        dmin = dmat.min(axis=1)
        nearest = dmat.argmin(axis=1)

        scales = gs.scales.detach().cpu().numpy()
        amps = gs.amplitudes.detach().cpu().numpy()
        mass = amps * scales.prod(axis=1)

        print(f'\n=== {tag}: {len(nuclei)} nuclei, K={args.k}, '
              f'nucleus voxels {100*nuc_mask.mean():.1f}% of volume ===')

        # ---------------- 3. threshold stability + 1. n_j
        for R in RADII:
            is_obj = dmin <= R
            counts = np.bincount(nearest[is_obj], minlength=len(nuclei))
            stability_rows.append(dict(
                region=tag, radius=R, n_object=int(is_obj.sum()),
                n_support=int((~is_obj).sum()),
                mass_object=float(mass[is_obj].sum() / mass.sum()),
                nuclei_with_ge1=int((counts >= 1).sum()),
                nuclei_with_ge2=int((counts >= 2).sum()),
                median_nj=float(np.median(counts[counts > 0])) if (counts > 0).any() else 0.0,
                mean_nj=float(counts[counts > 0].mean()) if (counts > 0).any() else 0.0,
                max_nj=int(counts.max())))
            print(f'  radius {R:>3.0f} vox: OBJECT={int(is_obj.sum()):3d}  '
                  f'SUPPORT={int((~is_obj).sum()):3d}  mass_obj={mass[is_obj].sum()/mass.sum():.2f}  '
                  f'nuclei with >=1: {int((counts>=1).sum()):2d}/{len(nuclei)}  '
                  f'>=2: {int((counts>=2).sum()):2d}  median n_j='
                  f'{np.median(counts[counts>0]) if (counts>0).any() else 0:.1f}  max={counts.max()}')
            if R == 5.0:
                for j, c in enumerate(counts):
                    nj_rows.append(dict(region=tag, nucleus=j, n_gaussians=int(c)))

        # ---------------- 2. role ablation at the middle radius
        is_obj = dmin <= 5.0
        for label, mask in [('all', np.ones_like(is_obj)),
                            ('object_only', is_obj),
                            ('support_only', ~is_obj)]:
            if mask.sum() == 0:
                continue
            recon = subset(gs, mask).query_volume(roi.shape).cpu().numpy().astype(np.float32)
            cm = score_cells(recon, roi, mode='3d')
            row = dict(region=tag, variant=label, n=int(mask.sum()),
                       psnr_global=M.psnr(recon, roi),
                       psnr_nucleus=masked_psnr(recon, roi, nuc_mask),
                       psnr_background=masked_psnr(recon, roi, bg_mask),
                       cell_f1=cm['cell_f1'], recall=cm['recall'],
                       mass_frac=float(mass[mask].sum() / mass.sum()))
            ablation_rows.append(row)
            print(f'    {label:13s} n={row["n"]:3d}  global={row["psnr_global"]:6.2f}  '
                  f'nucleus={row["psnr_nucleus"]:6.2f}  bg={row["psnr_background"]:6.2f}  '
                  f'F1={row["cell_f1"]:.3f}')

    import pandas as pd
    ab = pd.DataFrame(ablation_rows); ab.to_csv(OUT / 'role_ablation.csv', index=False)
    nj = pd.DataFrame(nj_rows); nj.to_csv(OUT / 'gaussians_per_nucleus.csv', index=False)
    st = pd.DataFrame(stability_rows); st.to_csv(OUT / 'threshold_stability.csv', index=False)

    g = ab.groupby('variant').agg(
        n=('n', 'mean'), glob=('psnr_global', 'mean'), nuc=('psnr_nucleus', 'mean'),
        bg=('psnr_background', 'mean'), f1=('cell_f1', 'mean'),
        mass=('mass_frac', 'mean'))
    order = [v for v in ['all', 'object_only', 'support_only'] if v in g.index]
    g = g.loc[order]
    print('\n=== ROLE ABLATION (mean over regions, radius 5 vox) ===')
    print(f'{"population":14s} {"n":>5s} {"global":>8s} {"nucleus":>8s} {"backgnd":>8s} '
          f'{"F1":>6s} {"mass":>6s}')
    for v in order:
        r = g.loc[v]
        print(f'{v:14s} {int(r.n):5d} {r.glob:8.2f} {r.nuc:8.2f} {r.bg:8.2f} '
              f'{r.f1:6.3f} {r["mass"]:6.2f}')

    print('\n=== PREMISE CHECK: Gaussians per nucleus (radius 5 vox) ===')
    c = nj.n_gaussians.values
    print(f'  nuclei with 0: {int((c==0).sum())}   1: {int((c==1).sum())}   '
          f'2: {int((c==2).sum())}   3+: {int((c>=3).sum())}')
    print(f'  median over covered nuclei: {np.median(c[c>0]) if (c>0).any() else 0:.1f}  '
          f'mean: {c[c>0].mean() if (c>0).any() else 0:.2f}  max: {c.max()}')

    print('\n=== THRESHOLD STABILITY ===')
    print(st.groupby('radius').agg(n_object=('n_object', 'mean'),
                                   mass_object=('mass_object', 'mean'),
                                   median_nj=('median_nj', 'mean'),
                                   max_nj=('max_nj', 'max')).round(3).to_string())

    json.dump({'ablation': g.reset_index().to_dict('records'),
               'stability': st.to_dict('records')},
              open(OUT / 'roles_summary.json', 'w'), indent=2)

    # ---------------- figure
    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    cols = {'all': '#5C6670', 'object_only': '#0E7C6B', 'support_only': '#B4462A'}
    panels = [('nuc', 'PSNR on NUCLEUS voxels (dB)'),
              ('bg', 'PSNR on BACKGROUND voxels (dB)'),
              ('f1', 'cell-detection F1')]
    for ax, (key, lab) in zip(axes[:3], panels):
        vals = [g.loc[v, key] for v in order]
        bars = ax.bar(range(len(order)), vals, color=[cols[v] for v in order], width=0.6)
        for b, v, n in zip(bars, vals, [g.loc[o, 'n'] for o in order]):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}\n(n={int(n)})',
                    ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.replace('_', '\n') for o in order])
        ax.set_ylabel(lab, fontweight='bold')
        ax.set_title(lab, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    ax = axes[3]
    vals, cnts = np.unique(nj.n_gaussians.values, return_counts=True)
    ax.bar(vals, cnts, color='#0E7C6B', width=0.7)
    ax.set_xlabel('Gaussians associated with one nucleus', fontweight='bold')
    ax.set_ylabel('number of nuclei', fontweight='bold')
    ax.set_title('PREMISE: is it one-to-one?', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Stage 0 — what does one Gaussian represent?', fontsize=14,
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'gaussian_roles.png', dpi=150, bbox_inches='tight')
    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

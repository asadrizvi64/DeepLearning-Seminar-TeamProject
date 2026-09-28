"""Rerun: same real 24-nucleus ROI, control vs capped fit, found/missed marked.

Reproduces the exact test behind the presentation images (target_vs_recon_marked.png,
control_vs_capped_real.png). Deterministic given seed=0, so this is a reproducibility
check on the SAME fit, not a new sample -- run scale_cap.py for the aggregated,
seed-varied version across 4 regions.

Usage:
    python scripts/rerun_control_vs_capped.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linear_sum_assignment

from volsplat.ablation import fit_with_validation
from volsplat.cellmetrics import detect_cells

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/real_proof_rerun'
OUT.mkdir(parents=True, exist_ok=True)
MATCH_R = 0.765 * 13


def fit_and_score(vol, target_nuclei, max_scale, max_scale_n, tag):
    gs, _, res = fit_with_validation(
        vol, num_gaussians=250, iterations=3000, init_strategy='coverage',
        parameterization='full', init_scale=2.0, seed=0,
        max_scale=max_scale, max_scale_n=max_scale_n, full_recon=True)
    recon = gs.query_volume(vol.shape).cpu().numpy().astype(np.float32)
    pred = detect_cells(recon, mode='3d')
    found = np.zeros(len(target_nuclei), dtype=bool)
    if len(pred):
        d = np.linalg.norm(pred[:, None, :] - target_nuclei[None, :, :], axis=-1)
        ri, ci = linear_sum_assignment(d)
        for i, j in zip(ri, ci):
            if d[i, j] <= MATCH_R:
                found[j] = True
    print(f'{tag}: full_psnr={res["full_psnr"]:.2f} dB  val_psnr={res["val_psnr"]:.2f} dB  '
          f'recovered {found.sum()}/{len(target_nuclei)}', flush=True)
    return recon, found, res['full_psnr']


def main():
    vol = np.load(REPO / 'runs/tribolium_cells_64x128x128.npy').astype(np.float32)
    target_nuclei = detect_cells(vol, mode='3d')
    n_nuc = len(target_nuclei)
    print(f'target volume {vol.shape}, {n_nuc} real nuclei (Tribolium, Lund)', flush=True)

    recon_ctrl, found_ctrl, psnr_ctrl = fit_and_score(vol, target_nuclei, None, None, 'CONTROL (no cap)')
    recon_cap, found_cap, psnr_cap = fit_and_score(vol, target_nuclei, 10.0, n_nuc, 'CAPPED (medium=10 vox)')

    print(f'\nrecall: control {found_ctrl.sum()}/{n_nuc} -> capped {found_cap.sum()}/{n_nuc}', flush=True)
    flips = np.where(found_ctrl != found_cap)[0]
    for i in flips:
        arrow = 'MISS -> FOUND' if (not found_ctrl[i] and found_cap[i]) else 'FOUND -> MISS'
        print(f'  idx={i} zyx={target_nuclei[i]}  {arrow}', flush=True)
    if len(flips) == 0:
        print('  no nuclei flipped', flush=True)

    np.save(OUT / 'target.npy', vol)
    np.save(OUT / 'recon_ctrl.npy', recon_ctrl)
    np.save(OUT / 'recon_cap.npy', recon_cap)
    np.save(OUT / 'target_nuclei.npy', target_nuclei)
    np.save(OUT / 'found_ctrl.npy', found_ctrl)
    np.save(OUT / 'found_cap.npy', found_cap)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    for ax, img, title, found in [
        (axes[0], vol.max(0), 'REAL target', None),
        (axes[1], recon_ctrl.max(0), f'CONTROL (no cap): {found_ctrl.sum()}/{n_nuc} recovered  PSNR {psnr_ctrl:.2f} dB', found_ctrl),
        (axes[2], recon_cap.max(0), f'CAPPED (medium=10 vox): {found_cap.sum()}/{n_nuc} recovered  PSNR {psnr_cap:.2f} dB', found_cap)]:
        ax.imshow(img, cmap='magma', vmin=0, vmax=1); ax.axis('off')
        ax.set_title(title, fontsize=13, fontweight='bold')
        if found is not None:
            for i, (z, y, x) in enumerate(target_nuclei.astype(int)):
                c = '#2ca02c' if found[i] else '#ff3b3b'
                lw = 3.2 if i in flips else 1.6
                ax.add_patch(mpatches.Circle((x, y), 6, fill=False, edgecolor=c, linewidth=lw))
    fig.tight_layout()
    fig.savefig(OUT / 'control_vs_capped_rerun.png', dpi=145, bbox_inches='tight')
    print(f'\nOutputs -> {OUT}', flush=True)


if __name__ == '__main__':
    main()

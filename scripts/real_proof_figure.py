"""Actual proof: real microscopy target vs real reconstruction, with the failure marked.

Not a stat chart -- this fits the real Tribolium ROI with the exact "control" config
already reported in the scale-cap result (K=250, coverage init, full parameterization,
3000 iters, no scale cap), then shows:

    1. real target MIP / real reconstruction MIP / |error| MIP
    2. every target nucleus marked GREEN (recovered) or RED (missed) on the real tissue
    3. a zoomed crop on one specific missed nucleus: target vs reconstruction, so the
       failure is visible on the actual voxels, not asserted in a table
    4. the same for one RECOVERED nucleus, as the honest contrast case

Usage:
    python scripts/real_proof_figure.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linear_sum_assignment

from volsplat.ablation import fit_with_validation
from volsplat.cellmetrics import detect_cells

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/real_proof'
OUT.mkdir(parents=True, exist_ok=True)

MATCH_R = 0.765 * 13


def main():
    vol = np.load(REPO / 'runs/tribolium_cells_64x128x128.npy').astype(np.float32)
    print(f'target volume {vol.shape}, real Tribolium (Lund) light-sheet data')

    gs, _, res = fit_with_validation(
        vol, num_gaussians=250, iterations=3000, init_strategy='coverage',
        parameterization='full', init_scale=2.0, seed=0, full_recon=True)
    recon = gs.query_volume(vol.shape).cpu().numpy().astype(np.float32)
    print(f'fit done: full_psnr={res["full_psnr"]:.2f} dB  val_psnr={res["val_psnr"]:.2f} dB')

    target_nuclei = detect_cells(vol, mode='3d')
    pred_nuclei = detect_cells(recon, mode='3d')
    n_tgt = len(target_nuclei)
    print(f'{n_tgt} target nuclei detected in the REAL data')

    found = np.zeros(n_tgt, dtype=bool)
    if len(pred_nuclei) and n_tgt:
        d = np.linalg.norm(pred_nuclei[:, None, :] - target_nuclei[None, :, :], axis=-1)
        ri, ci = linear_sum_assignment(d)
        for i, j in zip(ri, ci):
            if d[i, j] <= MATCH_R:
                found[j] = True
    print(f'recovered {found.sum()}/{n_tgt}  (recall {found.mean():.3f})')

    np.save(OUT / 'target.npy', vol)
    np.save(OUT / 'recon.npy', recon)
    np.save(OUT / 'target_nuclei.npy', target_nuclei)
    np.save(OUT / 'found.npy', found)

    # ================================================================= FIGURE 1
    # target / reconstruction / error, MIP along z, every nucleus marked found/missed
    mip_t = vol.max(axis=0)
    mip_r = recon.max(axis=0)
    err = np.abs(vol - recon).max(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    axes[0].imshow(mip_t, cmap='magma', vmin=0, vmax=1)
    axes[0].set_title('REAL target (MIP)', fontsize=13, fontweight='bold')
    axes[1].imshow(mip_r, cmap='magma', vmin=0, vmax=1)
    axes[1].set_title(f'REAL reconstruction (MIP)  full PSNR {res["full_psnr"]:.2f} dB',
                      fontsize=13, fontweight='bold')
    im2 = axes[2].imshow(err, cmap='inferno', vmin=0, vmax=err.max())
    axes[2].set_title('|target − reconstruction| (MIP)', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes[:2]:
        for i, (z, y, x) in enumerate(target_nuclei.astype(int)):
            color = '#2ca02c' if found[i] else '#ff3b3b'
            circ = mpatches.Circle((x, y), 6, fill=False, edgecolor=color, linewidth=1.6)
            ax.add_patch(circ)
        ax.axis('off')
    axes[2].axis('off')

    handles = [mpatches.Patch(color='#2ca02c', label=f'recovered ({found.sum()})'),
              mpatches.Patch(color='#ff3b3b', label=f'MISSED ({(~found).sum()})')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=12,
              frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Real Tribolium light-sheet volume: every nucleus marked, found vs missed',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'target_vs_recon_marked.png', dpi=145, bbox_inches='tight')
    plt.close(fig)
    print('saved target_vs_recon_marked.png')

    # ================================================================= FIGURE 2
    # zoomed failure case: one missed nucleus, target vs reconstruction, actual voxels
    missed_idx = np.where(~found)[0]
    found_idx = np.where(found)[0]
    if len(missed_idx) == 0:
        print('no missed nuclei in this fit -- skipping failure zoom')
        return

    def crop_mips(center_zyx, half=16):
        z, y, x = center_zyx.astype(int)
        D, H, W = vol.shape
        z0, z1 = max(0, z - half), min(D, z + half + 1)
        y0, y1 = max(0, y - half), min(H, y + half + 1)
        x0, x1 = max(0, x - half), min(W, x + half + 1)
        return (vol[z0:z1, y0:y1, x0:x1], recon[z0:z1, y0:y1, x0:x1])

    # Exclude nuclei near the ROI boundary from example selection: a peak detector
    # can flag a genuine boundary/edge artifact (checked here after finding index 20
    # sat at voxel [z=4, y=127, x=20] -- y=127 is the LITERAL LAST ROW of a 128-row
    # volume), which is not a faint-nucleus loss and would misrepresent the mechanism.
    margin = 14
    D, H, W = vol.shape
    def interior(i):
        z, y, x = target_nuclei[i].astype(int)
        return (margin <= z <= D - margin and margin <= y <= H - margin
                and margin <= x <= W - margin)

    missed_interior = [i for i in missed_idx if interior(i)]
    found_interior = [i for i in found_idx if interior(i)]
    if not missed_interior:
        print('WARNING: no interior missed nucleus -- falling back to all missed')
        missed_interior = list(missed_idx)
    if not found_interior:
        found_interior = list(found_idx)

    # pick the missed nucleus with the highest local_maxima intensity among interior
    # candidates -- a clear failure, not a borderline detection-threshold miss
    tvals = [vol[tuple(target_nuclei[i].astype(int))] for i in missed_interior]
    worst = missed_interior[int(np.argmax(tvals))]
    # matched found example for honest contrast, similar (low) brightness rank
    fvals = [vol[tuple(target_nuclei[i].astype(int))] for i in found_interior]
    contrast_i = found_interior[int(np.argmin(fvals))]
    print(f'selected MISSED example: idx={worst} zyx={target_nuclei[worst]} '
          f'value={vol[tuple(target_nuclei[worst].astype(int))]:.3f}')
    print(f'selected RECOVERED example: idx={contrast_i} zyx={target_nuclei[contrast_i]} '
          f'value={vol[tuple(target_nuclei[contrast_i].astype(int))]:.3f}')

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for row, (idx, label, edgecolor) in enumerate(
            [(worst, 'MISSED nucleus', '#ff3b3b'), (contrast_i, 'RECOVERED (comparably faint)', '#2ca02c')]):
        tcrop, rcrop = crop_mips(target_nuclei[idx])  # default half=16
        tmip, rmip = tcrop.max(axis=0), rcrop.max(axis=0)
        errmip = np.abs(tcrop - rcrop).max(axis=0)
        for col, (img, title, cmap) in enumerate(
                [(tmip, 'target', 'magma'), (rmip, 'reconstruction', 'magma'),
                 (errmip, '|error|', 'inferno')]):
            ax = axes[row, col]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if col < 2 else errmip.max())
            if col == 0:
                ax.set_ylabel(label, fontsize=12, fontweight='bold', color=edgecolor)
            ax.set_title(title, fontsize=11)
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_color(edgecolor); spine.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Zoomed on real voxels: a missed faint nucleus vs a recovered one of similar brightness',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'failure_case_zoom.png', dpi=145, bbox_inches='tight')
    plt.close(fig)
    print('saved failure_case_zoom.png')

    # ================================================================= FIGURE 3
    # scale-broadening mechanism, made visible -- CAUTION built in: at K=250 with
    # ~24 nucleus-seeded Gaussians and ~226 support Gaussians, "nearest Gaussian by
    # raw position" can pick a SUPPORT Gaussian that happens to drift closer than the
    # one actually seeded on this nucleus, which would misattribute its scale. Bound
    # the search to a physically plausible capture radius instead of nearest-at-any-
    # distance; if nothing is within it, that absence is reported honestly rather than
    # silently attributing a distant Gaussian's scale to this nucleus. The reliable,
    # aggregated version of this result -- matched by seed INDEX, not nearest-position,
    # across many nuclei -- is runs/stage3_trajectory (lost 11.72 vs kept 9.07 vox,
    # p=0.0010); this figure is illustrative of that result, not a replacement for it.
    CAPTURE_R = 10.0
    pos = gs.positions.detach().cpu().numpy()          # (x,y,z)
    scale = gs.scales.detach().cpu().numpy().mean(axis=1)

    def nearest_within(idx):
        nuc_xyz = target_nuclei[idx][::-1]
        d = np.linalg.norm(pos - nuc_xyz, axis=1)
        gidx = int(np.argmin(d))
        return (gidx, float(d[gidx])) if d[gidx] <= CAPTURE_R else (None, float(d[gidx]))

    gi_worst, d_worst = nearest_within(worst)
    gi_found, d_found = nearest_within(contrast_i)
    print(f'MISSED nucleus: nearest Gaussian at {d_worst:.1f} vox'
          + (f', scale={scale[gi_worst]:.2f}' if gi_worst is not None else ' -- NONE within capture radius'))
    print(f'RECOVERED nucleus: nearest Gaussian at {d_found:.1f} vox'
          + (f', scale={scale[gi_found]:.2f}' if gi_found is not None else ' -- NONE within capture radius'))

    if gi_worst is None or gi_found is None:
        print('skipping broadening_visible.png -- no reliably-attributable Gaussian '
              'for one of the two examples; use runs/stage3_trajectory instead')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
        for ax, idx, gidx, label, edgecolor in [
                (axes[0], worst, gi_worst, 'MISSED', '#ff3b3b'),
                (axes[1], contrast_i, gi_found, 'RECOVERED', '#2ca02c')]:
            tcrop, rcrop = crop_mips(target_nuclei[idx], half=14)
            ax.imshow(rcrop.max(axis=0), cmap='magma', vmin=0, vmax=1)
            circ = mpatches.Circle((14, 14), scale[gidx], fill=False, edgecolor=edgecolor,
                                   linewidth=2.2, linestyle='--')
            ax.add_patch(circ)
            ax.set_title(f'{label}\nnearest Gaussian scale = {scale[gidx]:.1f} vox', fontsize=12,
                         fontweight='bold', color=edgecolor)
            ax.axis('off')
        fig.suptitle('One example (see runs/stage3_trajectory for the aggregated, matched result)',
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / 'broadening_visible.png', dpi=145, bbox_inches='tight')
        plt.close(fig)
        print('saved broadening_visible.png')

    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

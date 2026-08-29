"""Visual proof of the greedy search: baseline (isotropic) vs champion (full) fit
on the real Tribolium ROI, side by side with the ground truth.

Fits both configs, reconstructs, and renders actual / baseline / champion / error
across three orthogonal planes. Makes the +3.84 dB parameterization win tangible.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from volsplat.ablation import fit_with_validation

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/greedy_tribolium'


def fit(volume, **cfg):
    gs, _, res = fit_with_validation(volume, **cfg)
    recon = gs.query_volume(volume.shape).cpu().numpy().astype(np.float32)
    return recon, res


def main():
    volume = np.load(REPO / 'runs/tribolium_roi_64x128x128.npy').astype(np.float32)

    print('Fitting baseline (isotropic, k=50)...')
    base_recon, base_res = fit(
        volume, num_gaussians=50, iterations=1500,
        init_strategy='intensity_weighted', parameterization='isotropic', init_scale=2.0)

    print('Fitting champion (full, k=100, scale=4, 3000it)...')
    champ_recon, champ_res = fit(
        volume, num_gaussians=100, iterations=3000,
        init_strategy='intensity_weighted', parameterization='full', init_scale=4.0)

    D, H, W = volume.shape
    zc, yc, xc = D // 2, H // 2, W // 2
    planes = [('XY z-mid', np.s_[zc, :, :]), ('XZ y-mid', np.s_[:, yc, :]),
              ('YZ x-mid', np.s_[:, :, xc])]

    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    fig.suptitle(
        f'Real Tribolium: baseline (isotropic) vs champion (full)   '
        f'val PSNR {base_res["val_psnr"]:.2f} -> {champ_res["val_psnr"]:.2f} dB',
        fontsize=15, fontweight='bold')
    col_titles = ['Actual (GT)', f'Baseline isotropic\nval={base_res["val_psnr"]:.2f} dB',
                  f'Champion full\nval={champ_res["val_psnr"]:.2f} dB',
                  'Champion |error|']
    for row, (pname, sl) in enumerate(planes):
        gt_s, b_s, c_s = volume[sl], base_recon[sl], champ_recon[sl]
        err = np.abs(c_s - gt_s)
        for col, (img, cmap, vmax) in enumerate([
                (gt_s, 'magma', 1), (b_s, 'magma', 1), (c_s, 'magma', 1), (err, 'hot', None)]):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(pname, fontsize=11, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = OUT / 'champion_vs_baseline.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

    summary = {
        'baseline': {'config': 'isotropic k=50 1500it', **{k: base_res[k] for k in
                     ['val_psnr', 'full_ssim', 'val_mae', 'num_params']}},
        'champion': {'config': 'full k=100 scale4 3000it', **{k: champ_res[k] for k in
                     ['val_psnr', 'full_ssim', 'val_mae', 'num_params']}},
    }
    with open(OUT / 'champion_vs_baseline.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

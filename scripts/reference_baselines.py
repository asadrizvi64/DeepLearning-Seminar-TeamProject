"""Reference baselines for the Tribolium fit, scored on the SAME held-out voxels.

The greedy search reports improvement over its own arbitrary starting config
(isotropic, k=50). That is an internal baseline, not evidence the method is good in
absolute terms. This script adds honest external reference points:

  constant     predict the (train) mean everywhere -- the do-nothing floor
  downsample   classical compression at a MATCHED parameter budget: bin the volume to
               a coarse grid holding ~the same number of numbers as the Gaussian model,
               then trilinearly upsample back

All scored on the identical held-out validation voxels used by fit_with_validation,
so the numbers are directly comparable.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from volsplat.ablation import make_val_split
from volsplat import metrics as M

REPO = Path(__file__).resolve().parent.parent
VAL_FRACTION, VAL_SEED = 0.1, 1234


def grid_dims_for_budget(shape, budget):
    """Coarse grid dims proportional to `shape` holding about `budget` values."""
    shape = np.array(shape, dtype=float)
    scale = (budget / shape.prod()) ** (1 / 3)
    dims = np.maximum(np.round(shape * scale), 2).astype(int)
    return tuple(int(d) for d in dims)


def downsample_upsample(volume, budget):
    """Average-pool to a coarse grid of ~`budget` values, then trilinear upsample."""
    dims = grid_dims_for_budget(volume.shape, budget)
    t = torch.from_numpy(volume)[None, None]
    coarse = F.adaptive_avg_pool3d(t, dims)
    back = F.interpolate(coarse, size=volume.shape, mode='trilinear', align_corners=False)
    return back[0, 0].numpy(), dims, int(np.prod(dims))


def main():
    volume = np.load(REPO / 'runs/tribolium_roi_64x128x128.npy').astype(np.float32)
    D, H, W = volume.shape
    n_vox = D * H * W

    # identical split to fit_with_validation
    train_idx, val_idx = make_val_split(n_vox, VAL_FRACTION, VAL_SEED)
    flat = volume.reshape(-1)
    val_t = flat[val_idx.numpy()]
    train_mean = float(flat[train_idx.numpy()].mean())

    print(f'Volume {volume.shape} = {n_vox:,} voxels')
    print(f'  train voxels {train_idx.numel():,}   held-out val voxels {val_idx.numel():,}\n')

    rows = []

    # --- constant (mean) predictor
    const_pred = np.full_like(val_t, train_mean)
    rows.append({'method': 'constant (train mean)', 'params': 1,
                 'val_psnr': M.psnr(const_pred, val_t), 'val_mae': M.mae(const_pred, val_t)})

    # --- matched-budget classical compression
    for budget, label in [(1000, 'k=100 full'), (10000, 'k=1000 full')]:
        recon, dims, actual = downsample_upsample(volume, budget)
        pred = recon.reshape(-1)[val_idx.numpy()]
        rows.append({
            'method': f'downsample+upsample {dims} (matched to {label})',
            'params': actual,
            'val_psnr': M.psnr(pred, val_t), 'val_mae': M.mae(pred, val_t)})

    print(f'{"method":52s} {"params":>7s} {"val PSNR":>9s} {"val MAE":>8s}')
    print('-' * 80)
    for r in rows:
        print(f'{r["method"]:52s} {r["params"]:7d} {r["val_psnr"]:9.2f} {r["val_mae"]:8.4f}')

    # Gaussian results measured earlier, for side-by-side context
    print('\nGaussian splatting (this work, same held-out voxels):')
    print(f'{"  isotropic k=50 (internal baseline)":52s} {500:7d} {18.90:9.2f}')
    print(f'{"  full k=100 (greedy champion, 3000it)":52s} {1000:7d} {23.37:9.2f}')
    print(f'{"  full k=1000 (capacity peak, 1000it)":52s} {10000:7d} {23.07:9.2f}')

    with open(REPO / 'runs/capacity_tribolium/reference_baselines.json', 'w') as f:
        json.dump({'train_mean': train_mean, 'n_voxels': n_vox,
                   'n_train': int(train_idx.numel()), 'n_val': int(val_idx.numel()),
                   'baselines': rows}, f, indent=2)


if __name__ == '__main__':
    main()

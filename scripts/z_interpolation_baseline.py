"""What must Gaussian continuity beat to be useful along z?

The representation is continuous, so it CAN be sampled between slices. That is a
mathematical property, not a demonstrated benefit. The benefit only exists if predicting
an unseen z-slice from the Gaussians beats predicting it by ordinary interpolation.

This establishes the baseline BEFORE any Gaussian is fitted -- the same discipline that
caught the compression claim, where plain downsampling matched the Gaussians.

Protocol: drop every other z-slice, reconstruct it from the survivors by nearest /
linear / cubic interpolation, and score PSNR and SSIM on the held-out slices only.

If interpolation already scores high, axial super-resolution has little headroom and
should not be pursued. If it scores poorly, there is real missing information and a
continuous representation has something to prove.

Usage:
    python scripts/z_interpolation_baseline.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from volsplat import metrics as M

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/z_interpolation'


def evaluate(vol, name, z_um):
    D = vol.shape[0]
    keep = np.arange(0, D, 2)          # even slices survive
    held = np.arange(1, D - 1, 2)      # odd slices are hidden (skip last for cubic)
    rows = []
    print(f'\n=== {name}  shape {vol.shape}, z step {z_um} um ===')
    print(f'  {len(keep)} slices kept, {len(held)} held out '
          f'(effective z step becomes {2*z_um:.2f} um)')

    gt = vol[held]
    # a floor: predict the volume mean everywhere
    const = np.full_like(gt, vol[keep].mean())
    rows.append(('constant (mean)', M.psnr(const, gt), M.ssim3d(const, gt)))

    # copy the nearest kept slice
    nn = vol[keep][np.clip(np.searchsorted(keep, held), 0, len(keep) - 1)]
    rows.append(('nearest slice', M.psnr(nn, gt), M.ssim3d(nn, gt)))

    for kind in ['linear', 'cubic']:
        f = interp1d(keep, vol[keep], axis=0, kind=kind,
                     bounds_error=False, fill_value='extrapolate')
        pred = f(held).astype(np.float32)
        rows.append((f'{kind} interpolation', M.psnr(pred, gt), M.ssim3d(pred, gt)))

    print(f'  {"method":24s} {"PSNR":>8s} {"SSIM":>8s}')
    for n, p, s in rows:
        print(f'  {n:24s} {p:8.2f} {s:8.3f}')
    return [dict(dataset=name, method=n, psnr=float(p), ssim=float(s)) for n, p, s in rows]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    trib = np.load(REPO / 'runs/tribolium_cells_64x128x128.npy').astype(np.float32)
    results += evaluate(trib, 'Tribolium nuclei ROI', 3.0)

    dro_full = REPO / 'runs/colleague_full_volume.npy'
    import tifffile
    from volsplat.ctc import percentile_normalize
    DRO = Path(r'D:/Download/train/Fluo-N3DL-DRO (1)/Fluo-N3DL-DRO')
    man = np.load(REPO / 'runs/dro_manual_centroids.npy')
    vol = percentile_normalize(tifffile.imread(str(DRO / '01/t000.tif')).astype(np.float32))
    lo = np.maximum(man.min(0).astype(int) - 8, 0)
    roi = np.ascontiguousarray(vol[lo[0]:lo[0] + 64, lo[1]:lo[1] + 128, lo[2]:lo[2] + 128])
    results += evaluate(roi, 'Drosophila nuclei ROI', 2.03)

    json.dump(results, open(OUT / 'z_interpolation.json', 'w'), indent=2)
    print(f'\nThe best interpolation score is the bar a continuous representation must')
    print(f'clear to justify axial super-resolution.')
    print(f'Outputs -> {OUT}')


if __name__ == '__main__':
    main()

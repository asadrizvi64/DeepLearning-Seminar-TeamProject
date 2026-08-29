"""Is warm-start tracking feasible at all? Measured from human annotation, no fitting.

The proposed contribution is: fit frame t, warm-start frame t+1 from it, and let each
Gaussian follow its nucleus so correspondence comes free. That only works if a nucleus
moves LESS between frames than a Gaussian can travel during fitting.

Measured Gaussian travel under the current optimizer is 0.41-0.95 voxels, with a
displacement scale of order lr * iters = 0.0016 * 3000 = 4.8 voxels.

This script measures, from the 189 manually tracked Drosophila nuclei:
    1. per-frame displacement of real nuclei (voxels and microns)
    2. how that compares to the Gaussian travel budget -> is warm-start viable?
    3. what position learning rate WOULD be required
    4. crowding / ID-switch risk: how often two tracked nuclei come close enough that
       a nearest-neighbour assignment could swap them
    5. whether any divisions occur (mitosis handling)

Usage:
    python scripts/tracking_feasibility.py --frames 50
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/tracking_feasibility'
DRO = Path(r'D:/Download/train/Fluo-N3DL-DRO (1)/Fluo-N3DL-DRO')
VOXEL_ZYX_UM = np.array([2.03, 0.406, 0.406])

# current optimizer setting, for comparison
LR_POSITION = 0.0016
ITERS = 3000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=int, default=50)
    p.add_argument('--sequence', default='01')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tra_dir = DRO / f'{args.sequence}_GT/TRA'
    files = sorted(tra_dir.glob('man_track*.tif'))[:args.frames]
    print(f'Reading {len(files)} annotated frames from {tra_dir.name} ...')

    # centroid of every track id, per frame
    per_frame = {}
    for i, f in enumerate(files):
        lab = tifffile.imread(str(f))
        ids = np.unique(lab)
        ids = ids[ids > 0]
        cents = ndimage.center_of_mass(lab > 0, lab, ids)
        per_frame[i] = {int(t): np.asarray(c) for t, c in zip(ids, cents)}
        if i % 10 == 0:
            print(f'  frame {i:3d}: {len(ids)} tracked nuclei')

    # ---------------------------------------------------------------- 1. displacement
    disp_vox, disp_um = [], []
    for i in range(len(files) - 1):
        a, b = per_frame[i], per_frame[i + 1]
        for t in set(a) & set(b):
            d = b[t] - a[t]
            disp_vox.append(np.linalg.norm(d))
            disp_um.append(np.linalg.norm(d * VOXEL_ZYX_UM))
    disp_vox = np.array(disp_vox)
    disp_um = np.array(disp_um)

    budget = LR_POSITION * ITERS
    print(f'\n=== 1. how far do real nuclei move between consecutive frames? ===')
    print(f'  n = {len(disp_vox)} frame-to-frame observations')
    for q in [50, 90, 99]:
        print(f'  p{q}: {np.percentile(disp_vox, q):6.2f} voxels '
              f'= {np.percentile(disp_um, q):5.2f} um')
    print(f'  max: {disp_vox.max():.2f} voxels = {disp_um.max():.2f} um')

    print(f'\n=== 2. can a Gaussian keep up? ===')
    print(f'  Gaussian displacement scale (lr*iters) : {budget:.2f} voxels')
    print(f'  measured Gaussian travel               : 0.41 - 0.95 voxels')
    frac_ok_budget = float((disp_vox <= budget).mean())
    frac_ok_measured = float((disp_vox <= 0.95).mean())
    print(f'  nuclei moving <= {budget:.1f} vox (budget)   : {frac_ok_budget*100:.1f}%')
    print(f'  nuclei moving <= 0.95 vox (measured)  : {frac_ok_measured*100:.1f}%')

    # ---------------------------------------------------------------- 3. required lr
    p99 = np.percentile(disp_vox, 99)
    print(f'\n=== 3. what would the position lr need to be? ===')
    for label, need in [('median', np.median(disp_vox)), ('p99', p99),
                        ('max', disp_vox.max())]:
        print(f'  to cover {label:6s} motion ({need:5.2f} vox) in {ITERS} steps: '
              f'lr >= {need/ITERS:.5f}   ({need/ITERS/LR_POSITION:5.1f}x current)')

    # ---------------------------------------------------------------- 4. ID-switch risk
    print(f'\n=== 4. crowding / ID-switch risk ===')
    risk_rows = []
    for radius in [2.0, 3.0, 5.0]:
        close = 0
        total = 0
        for i in range(len(files) - 1):
            a, b = per_frame[i], per_frame[i + 1]
            shared = sorted(set(a) & set(b))
            if len(shared) < 2:
                continue
            pa = np.array([a[t] for t in shared]) * VOXEL_ZYX_UM
            pb = np.array([b[t] for t in shared]) * VOXEL_ZYX_UM
            # for each nucleus: is its own next position closer than another's?
            d_self = np.linalg.norm(pb - pa, axis=1)
            dmat = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
            np.fill_diagonal(dmat, np.inf)
            d_other = dmat.min(axis=1)
            close += int((d_other <= d_self).sum())
            total += len(shared)
        rate = close / max(total, 1)
        risk_rows.append(dict(radius_um=radius, swap_rate=rate))
        if radius == 2.0:
            print(f'  nearest-neighbour assignment would pick the WRONG nucleus: '
                  f'{close}/{total} = {rate*100:.2f}% of frame transitions')
            break

    # ---------------------------------------------------------------- 5. divisions
    print(f'\n=== 5. divisions present? ===')
    txt = (tra_dir / 'man_track.txt').read_text().strip().split('\n')
    tracks = [tuple(map(int, l.split())) for l in txt if len(l.split()) == 4]
    with_parent = [t for t in tracks if t[3] > 0]
    print(f'  {len(tracks)} tracks, {len(with_parent)} with a parent (division products)')
    if not with_parent:
        print('  => NO divisions in this sequence: tracking here tests FOLLOWING only')

    summary = dict(
        n_frames=len(files), n_observations=int(len(disp_vox)),
        disp_vox_median=float(np.median(disp_vox)),
        disp_vox_p99=float(p99), disp_vox_max=float(disp_vox.max()),
        disp_um_median=float(np.median(disp_um)),
        gaussian_budget_vox=budget,
        frac_within_budget=frac_ok_budget,
        frac_within_measured_travel=frac_ok_measured,
        lr_needed_p99=float(p99 / ITERS),
        lr_multiple_p99=float(p99 / ITERS / LR_POSITION),
        swap_rate=risk_rows[0]['swap_rate'],
        n_divisions=len(with_parent))
    json.dump(summary, open(OUT / 'tracking_feasibility.json', 'w'), indent=2)
    np.save(OUT / 'displacements_voxels.npy', disp_vox)
    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

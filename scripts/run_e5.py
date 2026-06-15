"""E5 - fidelity-vs-storage Pareto: 3DGS vs baselines (RQ2).

Places 3DGS on the same fidelity/storage front as compressed voxels, sparse point
clouds, and an implicit SIREN field. Each method contributes a curve (swept over its
quality knob); the plot shows where (if) 3DGS dominates.

Usage:
    python scripts/run_e5.py --shape 32 48 48 --num-blobs 20 --out-dir runs/e5

Caveats (read before interpreting):
  * The 3DGS curve uses a FIXED iteration count per budget, so larger budgets are
    under-converged and the curve can bend DOWN - the same confound flagged in the
    E3 review. For a valid 3DGS front, train each budget to a target PSNR (or scale
    iterations with budget). Reuse this harness on the converged real-data fit.
  * The phantom is a tiny, mostly-empty, trivially-compressible Gaussian sum, so
    compressed-voxel/implicit baselines dominate it. RQ2 ("does 3DGS dominate?") is
    only meaningful on DENSE real data - this run validates the harness, not the
    answer.
"""
import argparse
import json
import tempfile
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from volsplat.data import generate_phantom
from volsplat.train import train_static, evaluate_full
from volsplat.baselines import compressed_voxels, sparse_pointcloud, implicit_field


def gaussian_point(vol, volume_t, budget, iters, seed):
    gs, _ = train_static(vol, num_gaussians=budget, iterations=iters,
                         init_strategy='intensity_weighted', seed=seed, eval_every=0)
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp = f.name
    try:
        torch.save(gs.state_dict(), tmp)
        size = os.path.getsize(tmp)
    finally:
        os.unlink(tmp)
    return float(evaluate_full(gs, volume_t, subsample=100_000)), size


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shape', type=int, nargs=3, default=[32, 48, 48])
    p.add_argument('--num-blobs', type=int, default=20)
    p.add_argument('--sigma', type=float, default=2.0)
    p.add_argument('--gs-iters', type=int, default=2000)
    p.add_argument('--gs-budgets', type=int, nargs='+', default=[5, 10, 20, 50, 100],
                   help='Gaussian budget sweep. Default is appropriate for num-blobs<=20. '
                        'Larger budgets need proportionally more iters to converge.')
    p.add_argument('--out-dir', type=str, default='runs/e5')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    vol = generate_phantom(shape=tuple(args.shape), num_blobs=args.num_blobs,
                           blob_scale_range=(args.sigma, args.sigma + 1e-6),
                           blob_amp_range=(0.8, 1.2), seed=args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    volume_t = torch.from_numpy(vol.astype(np.float32)).to(device)
    print(f'Phantom {vol.shape}, {args.num_blobs} blobs')

    curves = {}
    print('compressed voxels...')
    curves['compressed voxels'] = [compressed_voxels(vol, b) for b in (2, 3, 4, 6, 8)]
    print('sparse point cloud...')
    curves['sparse point cloud'] = [sparse_pointcloud(vol, t) for t in (0.3, 0.15, 0.07, 0.03, 0.01)]
    print('implicit (SIREN)...')
    curves['implicit field'] = [implicit_field(vol, hidden=h, iters=1200, seed=args.seed)
                                for h in (16, 32, 48, 64)]
    print('3DGS...')
    curves['3DGS'] = [gaussian_point(vol, volume_t, b, args.gs_iters, args.seed)
                      for b in args.gs_budgets]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {'compressed voxels': 'o', 'sparse point cloud': 's',
               'implicit field': '^', '3DGS': 'D'}
    for name, pts in curves.items():
        pts = [pt for pt in pts if pt[1] > 0]
        xs = [b / 1024 for _, b in pts]; ys = [ps for ps, _ in pts]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], marker=markers[name],
                label=name, linewidth=1.8, markersize=7)
    ax.set_xscale('log'); ax.set_xlabel('Storage (KB, log)'); ax.set_ylabel('PSNR (dB)')
    ax.set_title('E5: fidelity vs storage (RQ2)')
    ax.grid(True, which='both', alpha=0.3, ls='--'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'pareto.png', dpi=140, bbox_inches='tight')

    with open(out / 'e5_report.json', 'w') as f:
        json.dump({'config': vars(args),
                   'curves': {k: [[float(a), int(b)] for a, b in v] for k, v in curves.items()}},
                  f, indent=2)

    print('\n=== E5 fidelity/storage (PSNR dB @ KB) ===')
    for name, pts in curves.items():
        s = '  '.join(f'{ps:.1f}dB@{b/1024:.1f}KB' for ps, b in pts if b > 0)
        print(f'{name:<20} {s}')
    print(f'\nPareto plot -> {out/"pareto.png"}')


if __name__ == '__main__':
    main()

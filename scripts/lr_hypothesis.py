"""Is position stuck because of an assignment barrier, or because the position
optimizer is under-scaled for voxel coordinates?

BACKGROUND
This repository uses a fixed `lr_position = 0.0016` with no coordinate-scale
normalization. The original 3DGS convention is `position_lr_init = 0.00016` multiplied
by a scene-dependent `spatial_lr_scale` (derived from camera/scene extent), and decayed
on a schedule. So the issue is not simply "our number is too small":

    we imported a position-LR convention without importing its coordinate-scale
    normalization

At the current rate, 3000 Adam updates provide only a few-voxel displacement scale --
Adam's update is eta * m_hat / (sqrt(v_hat) + eps), whose ratio is near 1 for a stable
gradient but is NOT mathematically bounded to 1, so this is a scale estimate, not a
hard ceiling. Either way it sits far below the ~20-voxel inter-nucleus spacing.

TWO COMPETING EXPLANATIONS for why good initialization dominates:
    A  an intrinsic assignment / local-minimum barrier
    B  the position optimizer is simply under-scaled
B has not been ruled out. This experiment separates them by varying ONLY the position
learning rate, holding volume, seeds, K, loss and iterations fixed.

THE DECISIVE MEASUREMENT is not PSNR and not F1. It is:

    does a badly-initialized Gaussian actually MIGRATE to a different nucleus
    when given the budget to do so?

measured as the fraction of Gaussians whose nearest target nucleus changes between
initialization and the end of training.

    migration rises with lr, and the good-vs-bad init gap closes  -> explanation B
    migration stays near zero however large lr gets               -> explanation A

Usage:
    python scripts/lr_hypothesis.py --lrs 0.0016 0.005 0.02 0.05 0.1 --k 100
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from volsplat.init import init_gaussians
from volsplat.ablation import fit_with_validation
from volsplat.cellmetrics import detect_cells

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/lr_hypothesis'
INITS = ['random', 'local_maxima']
COLORS = {'random': '#d62728', 'local_maxima': '#2ca02c'}


def assignment_stats(p_init_xyz, p_final_xyz, nuclei_zyx):
    """Nearest-nucleus assignment at init vs final, and how much it changed.

    Positions arrive as (x, y, z); nuclei as (z, y, x).
    """
    if len(nuclei_zyx) == 0:
        return dict(migrated_frac=np.nan, d_init=np.nan, d_final=np.nan)
    nuc = nuclei_zyx[:, ::-1].astype(np.float32)              # -> (x, y, z)
    d0 = np.linalg.norm(p_init_xyz[:, None, :] - nuc[None, :, :], axis=-1)
    d1 = np.linalg.norm(p_final_xyz[:, None, :] - nuc[None, :, :], axis=-1)
    a0, a1 = d0.argmin(axis=1), d1.argmin(axis=1)
    return dict(
        migrated_frac=float((a0 != a1).mean()),
        d_init=float(d0.min(axis=1).mean()),
        d_final=float(d1.min(axis=1).mean()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--volume', default='runs/tribolium_cells_64x128x128.npy')
    p.add_argument('--lrs', type=float, nargs='+',
                   default=[0.0016, 0.005, 0.02, 0.05, 0.1])
    p.add_argument('--k', type=int, default=100)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    volume = np.load(REPO / args.volume).astype(np.float32)
    nuclei = detect_cells(volume, mode='3d')
    print(f'Volume {volume.shape}, {len(nuclei)} target nuclei, k={args.k}, '
          f'{args.iters} iters, seeds {args.seeds}\n')

    rows = []
    total = len(args.lrs) * len(INITS) * len(args.seeds)
    n = 0
    for lr in args.lrs:
        for init in INITS:
            for seed in args.seeds:
                n += 1
                # init is deterministic given (strategy, seed) -- reproduce it to get p0
                gs0 = init_gaussians(volume, args.k, strategy=init,
                                     init_scale=2.0, seed=seed)
                p0 = gs0.positions.detach().cpu().numpy().copy()

                gs, _, res = fit_with_validation(
                    volume, num_gaussians=args.k, iterations=args.iters,
                    init_strategy=init, parameterization='full', init_scale=2.0,
                    seed=seed, lr_position=lr, track_displacement=True,
                    cell_metrics=True, cell_kwargs={'mode': '3d'})
                p1 = gs.positions.detach().cpu().numpy()

                st = assignment_stats(p0, p1, nuclei)
                rows.append(dict(lr=lr, init=init, seed=seed,
                                 disp=res['disp_mean'], disp_p90=res['disp_p90'],
                                 migrated=st['migrated_frac'],
                                 d_init=st['d_init'], d_final=st['d_final'],
                                 cell_f1=res['cell_f1'], recall=res['recall'],
                                 psnr=res['full_psnr']))
                print(f"[{n}/{total}] lr={lr:<7g} {init:13s} s={seed}  "
                      f"moved={res['disp_mean']:6.2f}  migrated={st['migrated_frac']*100:5.1f}%  "
                      f"d_nuc {st['d_init']:.1f}->{st['d_final']:.1f}  "
                      f"F1={res['cell_f1']:.3f}  PSNR={res['full_psnr']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'lr_hypothesis.csv', index=False)
    g = df.groupby(['lr', 'init']).agg(
        disp=('disp', 'mean'), migrated=('migrated', 'mean'),
        d_init=('d_init', 'mean'), d_final=('d_final', 'mean'),
        f1=('cell_f1', 'mean'), f1s=('cell_f1', 'std'),
        psnr=('psnr', 'mean')).reset_index()

    print('\n=== means over seeds ===')
    print(g.round(3).to_string(index=False))

    gaps = []
    for lr in args.lrs:
        a = g[(g.lr == lr) & (g.init == 'local_maxima')]
        b = g[(g.lr == lr) & (g.init == 'random')]
        if len(a) and len(b):
            gaps.append({'lr': float(lr),
                         'gap_f1': float(a.f1.values[0] - b.f1.values[0]),
                         'random_disp': float(b.disp.values[0]),
                         'random_migrated': float(b.migrated.values[0]),
                         'random_d_init': float(b.d_init.values[0]),
                         'random_d_final': float(b.d_final.values[0])})
    print('\n=== the decisive numbers (random init) ===')
    print(f'{"lr":>8s} {"travel":>8s} {"migrated":>10s} {"d_nuc init->final":>20s} {"F1 gap":>8s}')
    for r in gaps:
        print(f"{r['lr']:8g} {r['random_disp']:8.2f} {r['random_migrated']*100:9.1f}% "
              f"{r['random_d_init']:9.1f} ->{r['random_d_final']:7.1f} "
              f"{r['gap_f1']:+8.3f}")
    json.dump(gaps, open(OUT / 'gaps.json', 'w'), indent=2)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.4))

    ax = axes[0]
    for init, sub in g.groupby('init'):
        sub = sub.sort_values('lr')
        ax.plot(sub.lr, sub.disp, 'o-', lw=2, ms=8, color=COLORS[init], label=init)
    ax.axhline(20, color='k', ls='--', lw=1.6, label='nucleus spacing')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('position learning rate', fontweight='bold')
    ax.set_ylabel('mean travel (voxels)', fontweight='bold')
    ax.set_title('Does travel scale with the rate?', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for init, sub in g.groupby('init'):
        sub = sub.sort_values('lr')
        ax.plot(sub.lr, sub.migrated * 100, 'o-', lw=2, ms=8, color=COLORS[init],
                label=init)
    ax.set_xscale('log')
    ax.set_xlabel('position learning rate', fontweight='bold')
    ax.set_ylabel('% Gaussians changing nearest nucleus', fontweight='bold')
    ax.set_title('DECISIVE: do blobs migrate?', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    for init, sub in g.groupby('init'):
        sub = sub.sort_values('lr')
        ax.errorbar(sub.lr, sub.f1, yerr=sub.f1s, fmt='o-', lw=2, ms=8, capsize=4,
                    color=COLORS[init], label=init)
    ax.set_xscale('log')
    ax.set_xlabel('position learning rate', fontweight='bold')
    ax.set_ylabel('cell-detection F1', fontweight='bold')
    ax.set_title('Does a bad init catch up?', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[3]
    if gaps:
        ax.plot([r['lr'] for r in gaps], [r['gap_f1'] for r in gaps], 'o-', lw=2.5,
                ms=9, color='#1f77b4')
        ax.axhline(0, color='k', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('position learning rate', fontweight='bold')
    ax.set_ylabel('F1 advantage of good init', fontweight='bold')
    ax.set_title('B predicts this shrinks;\nA predicts it is flat', fontsize=12,
                 fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Assignment barrier, or under-scaled position optimizer?',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'lr_hypothesis.png', dpi=150, bbox_inches='tight')
    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

"""How does each initializer SPEND its Gaussian budget?

No fitting -- this measures the initialization step alone, so it isolates the
allocation policy from anything the optimizer does afterwards.

Reported per strategy:
    C  target coverage      fraction of target nuclei with a seed within `radius`
    D  duplicates           mean seeds landing on each covered nucleus
    E  seed efficiency      distinct nuclei covered / total seeds
    U  unassigned seeds     fraction of seeds not near any target nucleus

The claim under test is not merely that the old initializer misses nuclei, but that it
spends a large budget inefficiently: redundant seeds on bright structure while
low-contrast nuclei stay uncovered.

`oracle_coverage` is an experimental upper bound that consumes the target positions.
It is never a deployable method and is reported only to bound stage 3.

Usage:
    python scripts/seed_allocation.py --k 250
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from volsplat.init import init_gaussians
from volsplat.cellmetrics import detect_cells

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/seed_allocation'
SHAPE = (64, 128, 128)
REGIONS = [(0, 528, 144), (0, 624, 144), (0, 240, 336), (0, 528, 240)]
STRATEGIES = ['local_maxima', 'suppressed_topk', 'coverage', 'oracle_coverage']


def allocation_stats(seeds_xyz, nuclei_zyx, radius=3.0):
    """seeds as (x,y,z); nuclei as (z,y,x)."""
    if len(nuclei_zyx) == 0 or len(seeds_xyz) == 0:
        return {}
    nuc = nuclei_zyx[:, ::-1].astype(np.float32)          # -> (x, y, z)
    d = np.linalg.norm(seeds_xyz[:, None, :] - nuc[None, :, :], axis=-1)   # (S, N)
    near = d <= radius
    covered = near.any(axis=0)                             # per nucleus
    seeds_on_nuclei = near.any(axis=1)                     # per seed
    seeds_per_covered = near[:, covered].sum(axis=0) if covered.any() else np.array([0])
    return dict(
        coverage=float(covered.mean()),
        n_covered=int(covered.sum()),
        n_nuclei=int(len(nuclei_zyx)),
        duplicates=float(seeds_per_covered.mean()),
        dup_max=int(seeds_per_covered.max()) if covered.any() else 0,
        efficiency=float(covered.sum() / len(seeds_xyz)),
        unassigned=float(1.0 - seeds_on_nuclei.mean()),
        n_seeds=int(len(seeds_xyz)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=250)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--radius', type=float, default=3.0)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    vol = np.load(REPO / 'runs/colleague_full_volume.npy').astype(np.float32)

    rows = []
    for (z, y, x) in REGIONS:
        roi = np.ascontiguousarray(vol[z:z + SHAPE[0], y:y + SHAPE[1], x:x + SHAPE[2]])
        nuclei = detect_cells(roi, mode='3d')
        for strat in STRATEGIES:
            kw = {'nuclei': nuclei} if strat == 'oracle_coverage' else {}
            gs = init_gaussians(roi, args.k, strategy=strat, init_scale=2.0,
                                seed=args.seed, **kw)
            seeds = gs.positions.detach().cpu().numpy()
            st = allocation_stats(seeds, nuclei, radius=args.radius)
            st.update(strategy=strat, region=f'z{z}y{y}x{x}')
            rows.append(st)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'seed_allocation.csv', index=False)

    g = df.groupby('strategy').agg(
        coverage=('coverage', 'mean'), duplicates=('duplicates', 'mean'),
        dup_max=('dup_max', 'max'), efficiency=('efficiency', 'mean'),
        unassigned=('unassigned', 'mean'), n_nuclei=('n_nuclei', 'sum'),
        n_covered=('n_covered', 'sum')).reindex(STRATEGIES)

    print(f'k={args.k} seeds per region, 4 regions, {int(g.n_nuclei.iloc[0])} target nuclei total')
    print(f'(coverage radius {args.radius} voxels)\n')
    print(f'{"strategy":18s} {"C cover":>8s} {"covered":>9s} {"D dup/nuc":>10s} '
          f'{"E eff":>7s} {"U unassigned":>13s}')
    print('-' * 74)
    for s in STRATEGIES:
        r = g.loc[s]
        tag = '  <- upper bound only' if s == 'oracle_coverage' else ''
        print(f'{s:18s} {r.coverage:8.2f} {int(r.n_covered):4d}/{int(r.n_nuclei):<4d} '
              f'{r.duplicates:10.2f} {r.efficiency:7.3f} {r.unassigned:13.2f}{tag}')

    json.dump(g.reset_index().to_dict('records'), open(OUT / 'summary.json', 'w'), indent=2)
    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

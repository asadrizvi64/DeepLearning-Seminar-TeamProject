"""Sequential (greedy / coordinate-descent) hyperparameter search on real data,
every step backed by a held-out validation proof and (optionally) seed error bars.

Methodology ("keep one axis static until it stops improving, then move on"): start from
a baseline config, then optimize ONE axis at a time while all other axes stay fixed at
their current best. For each axis we fit every candidate, score it, lock in the winner
ONLY if it beats the current state, and emit a proof figure.

    1. init_strategy    random / intensity_weighted / local_maxima
    2. parameterization isotropic (spheres) -> diagonal (stretch) -> full (rotate/tilt)
    3. num_gaussians    how many splats
    4. init_scale       initial Gaussian size
    5. iterations       training budget

Selection metric (`--select-metric`):
    val_psnr   held-out photometric quality (default)
    cell_f1    nucleus recovery F1 -- detect cells in the reconstruction and match them
               to cells detected in the target. Use this when the goal is finding
               nuclei, not matching intensities; on this data the two are nearly
               uncorrelated, so they select different configurations.
    full_ssim  full-volume structural similarity

`--seeds` repeats every candidate at multiple seeds and selects on the MEAN, reporting
the standard deviation so differences can be judged against seed noise.

Outputs (into --out-dir):
    greedy_results.csv     one row per fit (stage, config, seed, all metrics)
    best_config.json       final ideal configuration + improvement path
    stage{N}_{axis}.png    per-stage proof: selection metric per candidate (+/- std)
    improvement_path.png   selection metric climbing as each stage locks its winner
    champion_curve.png     train vs val PSNR over iterations for the final config

Usage:
    # photometric objective
    python scripts/greedy_ablation.py --volume runs/tribolium_roi_64x128x128.npy

    # biological objective, 3 seeds for error bars
    python scripts/greedy_ablation.py --volume runs/tribolium_cells_64x128x128.npy \
        --select-metric cell_f1 --seeds 0 1 2 --out-dir runs/greedy_cells
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from volsplat.ablation import fit_with_validation

REPO = Path(__file__).resolve().parent.parent

METRIC_LABEL = {
    'val_psnr': 'held-out validation PSNR (dB)',
    'cell_f1': 'cell-detection F1 (nucleus recovery)',
    'full_ssim': 'full-volume SSIM',
}

BASELINE = dict(
    init_strategy='intensity_weighted',
    parameterization='isotropic',
    num_gaussians=50,
    init_scale=2.0,
    iterations=1500,
)


def build_stages(ks, scales, iters_values):
    return [
        ('init_strategy',    ['random', 'intensity_weighted', 'local_maxima']),
        ('parameterization', ['isotropic', 'diagonal', 'full']),
        ('num_gaussians',    ks),
        ('init_scale',       scales),
        ('iterations',       iters_values),
    ]


def run_fit(volume, cfg, val_fraction, seed, need_cells):
    return fit_with_validation(
        volume,
        num_gaussians=int(cfg['num_gaussians']),
        iterations=int(cfg['iterations']),
        init_strategy=cfg['init_strategy'],
        parameterization=cfg['parameterization'],
        init_scale=float(cfg['init_scale']),
        val_fraction=val_fraction,
        seed=seed,
        cell_metrics=need_cells,
    )


def evaluate_candidate(volume, cfg, val_fraction, seeds, need_cells, select):
    """Fit `cfg` at every seed. Returns (mean_score, std_score, rows, last_history)."""
    rows, scores, hist = [], [], None
    for s in seeds:
        _, h, res = run_fit(volume, cfg, val_fraction, s, need_cells)
        res = dict(res); res.update(cfg); res['seed'] = s
        rows.append(res)
        scores.append(res.get(select, float('nan')))
        hist = h
    return float(np.nanmean(scores)), float(np.nanstd(scores)), rows, hist


def stage_proof_figure(stage_idx, axis, cands, select, out_dir, locked_label):
    labels = [str(c['value']) for c in cands]
    means = [c['mean'] for c in cands]
    stds = [c['std'] for c in cands]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(6.5, 1.8 * len(labels)), 5))
    bars = ax.bar(x, means, yerr=stds if any(s > 0 for s in stds) else None,
                  capsize=5, color='#1f77b4', width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_xlabel(axis, fontweight='bold')
    ax.set_ylabel(METRIC_LABEL.get(select, select), fontweight='bold')
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + (s or 0), f'{m:.3f}',
                ha='center', va='bottom', fontsize=9)
    wi = int(np.nanargmax(means))
    ax.text(x[wi], means[wi] + (stds[wi] or 0), '★', ha='center', va='bottom',
            fontsize=20, color='goldenrod')
    ax.set_title(f'Stage {stage_idx}: optimize {axis}   ->  {locked_label}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_dir / f'stage{stage_idx}_{axis}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def improvement_path_figure(path, select, out_dir):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    stages = [p['stage_label'] for p in path]
    vals = [p['score'] for p in path]
    errs = [p.get('std', 0) for p in path]
    ax.errorbar(range(len(vals)), vals, yerr=errs, fmt='o-', lw=2.5, ms=9,
                capsize=5, color='#2ca02c')
    for i, v in enumerate(vals):
        ax.annotate(f'{v:.3f}', (i, v), textcoords='offset points', xytext=(0, 12),
                    ha='center', fontweight='bold', fontsize=9)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=20, ha='right')
    ax.set_ylabel(METRIC_LABEL.get(select, select), fontweight='bold')
    ax.set_title('Greedy improvement path (each step locks one axis)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'improvement_path.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def champion_curve_figure(history, out_dir):
    tr = [(h['iter'], h['train_psnr']) for h in history if 'train_psnr' in h]
    va = [(h['iter'], h['val_psnr']) for h in history if 'val_psnr' in h]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if tr:
        tr = np.array(tr); ax.plot(tr[:, 0], tr[:, 1], lw=2, label='train PSNR')
    if va:
        va = np.array(va); ax.plot(va[:, 0], va[:, 1], lw=2, label='validation PSNR')
    ax.set_xlabel('iteration', fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontweight='bold')
    ax.set_title('Champion config: train vs held-out validation',
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'champion_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--volume', default='runs/tribolium_roi_64x128x128.npy')
    p.add_argument('--out-dir', default='runs/greedy_tribolium')
    p.add_argument('--select-metric', default='val_psnr',
                   choices=['val_psnr', 'cell_f1', 'full_ssim'])
    p.add_argument('--ks', type=int, nargs='+', default=[10, 50, 100, 250])
    p.add_argument('--scales', type=float, nargs='+', default=[1.0, 2.0, 4.0])
    p.add_argument('--iters-values', type=int, nargs='+', default=[800, 1500, 3000])
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--base-iters', type=int, default=None)
    p.add_argument('--base-k', type=int, default=None)
    args = p.parse_args()

    select = args.select_metric
    need_cells = (select == 'cell_f1')

    out_dir = Path(REPO / args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    volume = np.load(REPO / args.volume).astype(np.float32)
    print(f'Volume {volume.shape}  |  select on: {select}  |  seeds: {args.seeds}')
    print(f'held-out validation fraction: {args.val_fraction}\n')

    if need_cells:
        from volsplat.cellmetrics import detect_cells
        n_t = len(detect_cells(volume, mode='3d'))
        print(f'Target contains {n_t} detected nuclei (image-derived, not manual GT)\n')

    stages = build_stages(args.ks, args.scales, args.iters_values)
    best = dict(BASELINE)
    if args.base_iters is not None:
        best['iterations'] = args.base_iters
    if args.base_k is not None:
        best['num_gaussians'] = args.base_k

    all_rows, path = [], []

    print(f'Baseline: {best}')
    m0, s0, rows0, hist0 = evaluate_candidate(volume, best, args.val_fraction,
                                              args.seeds, need_cells, select)
    for r in rows0:
        r.update(stage=0, stage_axis='baseline')
    all_rows += rows0
    print(f'  baseline {select} = {m0:.4f} +/- {s0:.4f}\n')
    path.append({'stage_label': 'baseline', 'score': m0, 'std': s0})
    champion_history = hist0

    for si, (axis, values) in enumerate(stages, start=1):
        candidates = list(values)
        if best[axis] not in candidates:
            candidates = [best[axis]] + candidates
        print(f'=== Stage {si}: {axis} over {candidates} ===')
        cand_summaries = []
        for val in candidates:
            cfg = dict(best); cfg[axis] = val
            m, s, rows, hist = evaluate_candidate(volume, cfg, args.val_fraction,
                                                  args.seeds, need_cells, select)
            for r in rows:
                r.update(stage=si, stage_axis=axis)
            all_rows += rows
            cand_summaries.append({'value': val, 'mean': m, 'std': s, 'hist': hist})
            extra = ''
            if need_cells and rows:
                extra = (f"  (recall={rows[-1].get('recall', float('nan')):.3f} "
                         f"prec={rows[-1].get('precision', float('nan')):.3f} "
                         f"psnr={rows[-1].get('full_psnr', float('nan')):.2f})")
            print(f'  {axis}={val!s:18s} {select}={m:.4f} +/- {s:.4f}{extra}')

        winner = max(cand_summaries, key=lambda c: c['mean'])
        current = path[-1]['score']
        if winner['mean'] > current + 1e-9:
            best[axis] = winner['value']
            new_score, new_std = winner['mean'], winner['std']
            champion_history = winner['hist']
            locked = f"LOCK {axis}={winner['value']} ({new_score - current:+.4f})"
        else:
            new_score, new_std = current, path[-1].get('std', 0.0)
            locked = f"no improvement; keep {axis}={best[axis]}"
        print(f'  -> {locked}\n')

        stage_proof_figure(si, axis, cand_summaries, select, out_dir, locked)
        path.append({'stage_label': f'{axis}={best[axis]}', 'score': new_score,
                     'std': new_std})

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / 'greedy_results.csv', index=False)
    improvement_path_figure(path, select, out_dir)
    champion_curve_figure(champion_history, out_dir)

    summary = {
        'select_metric': select,
        'seeds': args.seeds,
        'ideal_config': best,
        'baseline_score': path[0]['score'],
        'final_score': path[-1]['score'],
        'total_improvement': path[-1]['score'] - path[0]['score'],
        'improvement_path': [{'step': p_['stage_label'], 'score': p_['score'],
                              'std': p_.get('std', 0.0)} for p_ in path],
    }
    with open(out_dir / 'best_config.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('=' * 62)
    print(f'IDEAL CONFIG (selected on {select}):')
    for k, v in best.items():
        print(f'  {k:18s}: {v}')
    print(f"\nBaseline {select}: {path[0]['score']:.4f}")
    print(f"Final    {select}: {path[-1]['score']:.4f}")
    print(f"Improvement      : {summary['total_improvement']:+.4f}")
    print(f'\nOutputs -> {out_dir}')


if __name__ == '__main__':
    main()

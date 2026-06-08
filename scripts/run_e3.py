"""E3 - Gaussian budget sweep (P3).

Evaluates reconstruction quality as a function of Gaussian budget.
For each budget in [500, 1000, 2000, 5000, 10000, 20000]:
  * Run the same volume-fitting experiment (fixed phantom, fixed iterations,
    fixed init strategy) using the voxel-query supervisor (P2 decision).
  * Collect: num_gaussians, final_psnr, final_mae, fit_time_sec,
             peak_gpu_memory_mb, model_size_bytes, parameter_count.

Outputs:
    runs/e3_budget/
        e3_report.json
        plots/
            psnr_vs_gaussians.png
            psnr_vs_size.png
            psnr_vs_time.png
            psnr_vs_memory.png   (skipped if no CUDA)

Research questions answered:
  RQ-budget-1: "How many Gaussians are needed before PSNR saturates?"
  RQ-budget-2: "What is the quality / compression tradeoff?"

Usage:
    python scripts/run_e3.py --shape 32 48 48 --num-blobs 15 --sigma 2.0 \\
        --iterations 1500 --out-dir runs/e3_budget

    # Quick smoke test:
    python scripts/run_e3.py --budgets 500 1000 2000 --iterations 500

Design:
    This script is the Gaussian-budget counterpart to run_e2.py, which swept
    init strategies at a fixed budget. The outer loop here sweeps *budgets* at
    a fixed init strategy (intensity_weighted, the E2 winner). All other knobs
    (volume, iterations, seed) are held constant so the budget is the sole
    independent variable.

    Re-use ledger:
        * generate_phantom     - unchanged from E1/E2
        * train_static         - unchanged; we just vary num_gaussians
        * evaluate_full        - unchanged; used for PSNR
        * evaluate_mae         - NEW helper in volsplat/train.py (mirrors evaluate_full)
        * mae                  - NEW function in volsplat/losses.py
        * measure_model_size   - script-local; saves state_dict to tmpfile
        * count_parameters     - script-local one-liner
        * make_plots           - script-local; pure matplotlib
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')        # headless — no display required
import matplotlib.pyplot as plt

from volsplat.data import generate_phantom
from volsplat.train import train_static, evaluate_full, evaluate_mae


# ------------------------------------------------------------------ constants

# Default budget list: 40x range, roughly doubling each step.
BUDGETS_DEFAULT = [500, 1_000, 2_000, 5_000, 10_000, 20_000]

# E2 found intensity_weighted is the best strategy on this phantom
# (ties local_maxima on final PSNR, but is ~35% faster). Use it here so the
# budget is the sole independent variable.
INIT_STRATEGY = 'intensity_weighted'


# ------------------------------------------------------------------ metrics helpers

def measure_model_size(gs) -> int:
    """Serialize the GaussianSet state_dict to a temp file; return bytes.

    Uses torch.save so the number matches exactly what a researcher would
    measure when storing a trained model to disk.
    """
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name
    try:
        torch.save(gs.state_dict(), tmp_path)
        return os.path.getsize(tmp_path)
    finally:
        os.unlink(tmp_path)


def count_parameters(gs) -> int:
    """Total number of scalar parameters (not bytes) across all nn.Parameters."""
    return sum(p.numel() for p in gs.parameters())


# ------------------------------------------------------------------ per-budget run

def run_one_budget(
    vol: np.ndarray,
    volume_t: torch.Tensor,
    budget: int,
    args,
    device: str,
) -> dict:
    """Train a GaussianSet at one budget level and return all metrics.

    Parameters
    ----------
    vol       : numpy volume (CPU) — passed to train_static for training
    volume_t  : torch tensor (GPU/CPU) — used for final evaluation only
    budget    : number of Gaussians
    args      : parsed CLI args (iterations, init_scale, seed, etc.)
    device    : 'cuda' or 'cpu'

    Returns
    -------
    Dict with all E3 metrics for this budget.
    """
    use_cuda = (device == 'cuda') and torch.cuda.is_available()

    # ---- reset peak GPU stats BEFORE training so the measurement is clean ----
    # volume_t is already allocated; this baseline is excluded from the peak
    # window. The measurement will capture: (1) the internal vol copy train_static
    # creates, (2) all model parameters and gradients, (3) the optimizer state.
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # ---- train ---------------------------------------------------------------
    start = time.time()
    gs, _hist = train_static(
        vol,
        num_gaussians=budget,
        iterations=args.iterations,
        init_strategy=INIT_STRATEGY,
        init_scale=args.init_scale,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        device=device,
    )
    fit_time = time.time() - start

    # ---- peak GPU memory (0.0 on CPU so the report is always valid) ----------
    peak_mb = 0.0
    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)

    # ---- quality metrics (consistent subsample across all budgets) -----------
    final_psnr = evaluate_full(gs, volume_t, subsample=args.eval_subsample)
    final_mae  = evaluate_mae(gs, volume_t,  subsample=args.eval_subsample)

    # ---- compression metrics -------------------------------------------------
    model_bytes = measure_model_size(gs)
    param_count = count_parameters(gs)

    return {
        'num_gaussians':      int(gs.num_gaussians),
        'final_psnr':         float(final_psnr),
        'final_mae':          float(final_mae),
        'fit_time_sec':       float(fit_time),
        'peak_gpu_memory_mb': float(peak_mb),
        'model_size_bytes':   int(model_bytes),
        'parameter_count':    int(param_count),
    }


# ------------------------------------------------------------------ plotting

def _annotate_points(ax, xs, ys, labels, fontsize: int = 7):
    """Annotate each (x, y) point with its budget label."""
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(
            f'{lbl:,}',
            xy=(x, y),
            xytext=(0, 7),
            textcoords='offset points',
            ha='center',
            fontsize=fontsize,
            color='#444444',
        )


def _save_single_plot(
    xs, ys, point_labels,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    logx: bool = False,
):
    """Write one PSNR-vs-X scatter/line plot."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs, ys, 'o-', color='steelblue', linewidth=1.8,
            markersize=6, markeredgewidth=0.8, markeredgecolor='white')
    _annotate_points(ax, xs, ys, point_labels)
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.6)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot  -> {path}')


def make_plots(results: list, budgets: list, out_dir: Path) -> None:
    """Write all E3 plots to out_dir/plots/.

    Always writes:
      psnr_vs_gaussians.png  — answers "when does PSNR saturate?"
      psnr_vs_size.png       — quality / compression Pareto front
      psnr_vs_time.png       — quality / compute tradeoff

    Conditionally writes (CUDA only):
      psnr_vs_memory.png     — quality / GPU memory tradeoff
    """
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    psnrs      = [r['final_psnr']         for r in results]
    gaussians  = [r['num_gaussians']       for r in results]
    sizes_kb   = [r['model_size_bytes'] / 1024.0 for r in results]
    times      = [r['fit_time_sec']        for r in results]
    memories   = [r['peak_gpu_memory_mb']  for r in results]

    _save_single_plot(
        gaussians, psnrs, budgets,
        xlabel='Number of Gaussians (log scale)',
        ylabel='Final PSNR (dB)',
        title='E3: Reconstruction Quality vs Gaussian Budget',
        path=plots_dir / 'psnr_vs_gaussians.png',
        logx=True,
    )
    _save_single_plot(
        sizes_kb, psnrs, budgets,
        xlabel='Model File Size (KB, log scale)',
        ylabel='Final PSNR (dB)',
        title='E3: Quality / Compression Tradeoff',
        path=plots_dir / 'psnr_vs_size.png',
        logx=True,
    )
    _save_single_plot(
        times, psnrs, budgets,
        xlabel='Fit Time (seconds)',
        ylabel='Final PSNR (dB)',
        title='E3: Quality vs Fit Time',
        path=plots_dir / 'psnr_vs_time.png',
        logx=False,
    )

    has_memory = any(r['peak_gpu_memory_mb'] > 0.0 for r in results)
    if has_memory:
        _save_single_plot(
            memories, psnrs, budgets,
            xlabel='Peak GPU Memory (MB)',
            ylabel='Final PSNR (dB)',
            title='E3: Quality vs Peak GPU Memory',
            path=plots_dir / 'psnr_vs_memory.png',
            logx=False,
        )
    else:
        print('  psnr_vs_memory.png  — skipped (no CUDA detected)')


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(
        description='E3: Gaussian budget sweep for the volsplat density fitter.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- output ---
    p.add_argument('--out-dir', type=str, default='runs/e3_budget',
                   help='Directory for e3_report.json and plots/')

    # ---- phantom ---
    p.add_argument('--shape', type=int, nargs=3, default=[32, 48, 48],
                   metavar=('D', 'H', 'W'),
                   help='Volume shape (D H W) — same as E2 default for comparability')
    p.add_argument('--num-blobs', type=int, default=15,
                   help='Number of Gaussian blobs in the synthetic phantom')
    p.add_argument('--sigma', type=float, default=2.0,
                   help='Blob sigma in voxel units (narrow range for controlled test)')

    # ---- training ---
    p.add_argument('--iterations', type=int, default=1500,
                   help='Training iterations per budget (same for all budgets)')
    p.add_argument('--init-scale', type=float, default=2.0,
                   help='Initial isotropic scale for all Gaussians (voxels)')
    p.add_argument('--seed', type=int, default=0,
                   help='RNG seed — same for all budgets for fair comparison')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--eval-every', type=int, default=100,
                   help='Full-volume eval interval (passed to train_static)')

    # ---- sweep ---
    p.add_argument('--budgets', type=int, nargs='+', default=BUDGETS_DEFAULT,
                   metavar='N',
                   help='Gaussian budgets to sweep (space-separated)')

    # ---- eval ---
    p.add_argument('--eval-subsample', type=int, default=100_000,
                   help='Voxel subsample for PSNR/MAE eval (consistent across budgets)')

    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device       : {device}')
    print(f'Init strategy: {INIT_STRATEGY}  (E2 winner — held constant)')

    # ----------------------------------------------------------------- phantom
    vol = generate_phantom(
        shape=tuple(args.shape),
        num_blobs=args.num_blobs,
        blob_scale_range=(args.sigma, args.sigma + 1e-6),  # ~fixed sigma
        blob_amp_range=(0.8, 1.2),
        seed=args.seed,
    )
    print(f'Phantom      : shape={vol.shape}, sigma~{args.sigma}, '
          f'#blobs={args.num_blobs}, '
          f'range=[{vol.min():.3f}, {vol.max():.3f}]')

    # Pre-build volume tensor once — reused for all evaluation calls.
    # (train_static builds its own internal copy for training.)
    volume_t = torch.from_numpy(vol.astype(np.float32)).to(device)

    # --------------------------------------------------------- sweep budgets
    budgets = sorted(args.budgets)
    print(f'\nBudgets ({len(budgets)}): {budgets}')
    print(f'Iterations per budget: {args.iterations}')

    results = []
    for budget in budgets:
        print(f'\n[N={budget:,}]  training {args.iterations} iters '
              f'with {INIT_STRATEGY} init ...')
        r = run_one_budget(vol, volume_t, budget, args, device)
        results.append(r)
        print(
            f'  PSNR = {r["final_psnr"]:.2f} dB   '
            f'MAE = {r["final_mae"]:.4f}   '
            f'time = {r["fit_time_sec"]:.1f}s   '
            f'mem = {r["peak_gpu_memory_mb"]:.1f} MB   '
            f'size = {r["model_size_bytes"] // 1024} KB   '
            f'params = {r["parameter_count"]:,}'
        )

    # ------------------------------------------------------------------- save
    report = {
        'config': {
            'out_dir':       args.out_dir,
            'shape':         args.shape,
            'num_blobs':     args.num_blobs,
            'sigma':         args.sigma,
            'iterations':    args.iterations,
            'init_strategy': INIT_STRATEGY,
            'init_scale':    args.init_scale,
            'seed':          args.seed,
            'budgets':       budgets,
            'device':        device,
        },
        # List of dicts in budget order — use a list (not a dict keyed by int)
        # so JSON round-trip preserves ordering and type unambiguously.
        'results': results,
    }
    report_path = out / 'e3_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport saved -> {report_path}')

    # ------------------------------------------------------------------ plots
    print('\nGenerating plots ...')
    make_plots(results, budgets, out)

    # --------------------------------------------------------- summary table
    print('\n--- E3 summary ---')
    header = (
        f'{"budget":>8}  {"PSNR(dB)":>9}  {"MAE":>8}  '
        f'{"time(s)":>8}  {"mem(MB)":>8}  '
        f'{"size(KB)":>9}  {"params":>10}'
    )
    print(header)
    print('-' * len(header))
    for b, r in zip(budgets, results):
        print(
            f'{b:>8,}  '
            f'{r["final_psnr"]:>9.2f}  '
            f'{r["final_mae"]:>8.4f}  '
            f'{r["fit_time_sec"]:>8.1f}  '
            f'{r["peak_gpu_memory_mb"]:>8.1f}  '
            f'{r["model_size_bytes"] // 1024:>9}  '
            f'{r["parameter_count"]:>10,}'
        )
    print(f'\nE3 complete. All outputs in: {out}')


if __name__ == '__main__':
    main()

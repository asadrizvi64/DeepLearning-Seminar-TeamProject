"""E3 - Gaussian Budget Sweep (P3).

Evaluates reconstruction-quality tradeoffs as a function of Gaussian budget.

Per-budget metrics collected:
  num_gaussians      - actual count after training (may differ if densify enabled)
  final_psnr         - PSNR on fixed 100k-voxel subsample (dB)
  final_mae          - MAE on SAME subsample (intensity units, [0,1])
  final_ssim         - 3D SSIM on full voxelized volume (or slice-wise for large vols)
  fit_time_sec       - wall-clock training time
  peak_gpu_memory_mb - peak allocated GPU memory during training (0 if CPU)
  model_size_bytes   - torch.save() file size (actual on-disk cost)
  parameter_count    - scalar parameter count (budget × 11 for GaussianSet)
  final_iteration    - actual iteration at which training stopped
  converged          - bool; True if patience criterion triggered

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
matplotlib.use('Agg')   # headless — no display required
import matplotlib.pyplot as plt

from volsplat.data import generate_phantom
from volsplat.train import train_static, evaluate_metrics, evaluate_ssim


# ------------------------------------------------------------------ constants

BUDGETS_DEFAULT = [500, 1_000, 2_000, 5_000, 10_000, 20_000]


INIT_STRATEGY = 'intensity_weighted'


SSIM_MAX_VOXELS_3D = 5_000_000


# ------------------------------------------------------------------ data loading

def load_data(args) -> np.ndarray:
    """Load or generate the target volume.

    Returns a (D, H, W) float32 numpy array normalized to [0, 1].
    Applies ROI cropping if --roi-shape is specified.
    """
    if args.phantom:
        print('Phase A: generating synthetic phantom')
        vol = generate_phantom(
            shape=tuple(args.shape),
            num_blobs=args.num_blobs,
            blob_scale_range=(args.sigma, args.sigma + 1e-6),
            blob_amp_range=(0.8, 1.2),
            seed=args.seed,
        )
        print(f'  shape={vol.shape}, sigma~{args.sigma}, #blobs={args.num_blobs}')
        return vol

    # Real volume
    path = Path(args.volume)
    if not path.exists():
        raise FileNotFoundError(
            f'Volume not found: {path}\n'
            f'For the phantom sanity check, use --phantom instead.'
        )
    print(f'Phase B: loading real volume from {path}')


    if path.name.startswith('lund_'):
        from volsplat.tribolium import load_lund_volume
        vol = load_lund_volume(path, normalize=True)
    elif args.ctc_normalize:
        # CTC data: use percentile normalization like train_ctc.py does
        import tifffile
        from volsplat.ctc import percentile_normalize
        raw = tifffile.imread(str(path)).astype(np.float32)
        while raw.ndim > 3 and raw.shape[0] == 1:
            raw = raw[0]
        vol = percentile_normalize(raw)
    else:
        from volsplat.data import load_volume
        vol = load_volume(path)

    print(f'  loaded shape={vol.shape}, range=[{vol.min():.3f}, {vol.max():.3f}]')

    # ROI crop
    if args.roi_shape is not None:
        vol = center_crop(vol, tuple(args.roi_shape))
        print(f'  center-cropped to {vol.shape}')

    return vol


def center_crop(vol: np.ndarray, target: tuple) -> np.ndarray:
    """Center-crop a (D, H, W) volume to target shape.

    If a target dimension exceeds the volume dimension, the original size is kept
    (no padding). Logs a warning in that case.
    """
    D, H, W = vol.shape
    td, th, tw = target
    d0 = max(0, (D - td) // 2)
    h0 = max(0, (H - th) // 2)
    w0 = max(0, (W - tw) // 2)
    cropped = vol[d0:d0+td, h0:h0+th, w0:w0+tw]
    if cropped.shape != target:
        print(f'  [warn] center_crop: requested {target} but volume {vol.shape} '
              f'is smaller in one or more dims; actual crop = {cropped.shape}')
    return cropped


# ------------------------------------------------------------------ per-budget run

def measure_model_size(gs) -> int:
    """Save state_dict to a temp file; return file size in bytes.

    This measures actual on-disk cost (including torch serialization overhead),
    which is the meaningful number for compression comparisons.
    """
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp = f.name
    try:
        torch.save(gs.state_dict(), tmp)
        return os.path.getsize(tmp)
    finally:
        os.unlink(tmp)


def count_parameters(gs) -> int:
    """Total scalar parameter count across all nn.Parameters.

    For GaussianSet with G Gaussians:
        positions (G,3) + log_scales (G,3) + quaternions (G,4) + amp_logits (G,) = G×11
    So parameter_count = num_gaussians × 11.
    """
    return sum(p.numel() for p in gs.parameters())


def extract_psnr_curve(history: list) -> list:
    """Extract the full_psnr entries from a training history for convergence plots."""
    return [
        {'iter': h['iter'], 'psnr': h['full_psnr']}
        for h in history
        if isinstance(h, dict) and 'full_psnr' in h
    ]


def run_one_budget(
    vol: np.ndarray,
    volume_t: torch.Tensor,
    budget: int,
    args,
    device: str,
) -> dict:
    """Train a GaussianSet at one budget level and collect all E3 metrics.

    Parameters
    ----------
    vol      : numpy volume (CPU) — passed directly to train_static for training
    volume_t : torch tensor on device — used for evaluation after training
    budget   : Gaussian count for this run
    args     : parsed CLI args
    device   : 'cuda' or 'cpu'

    Memory measurement
    ------------------
    torch.cuda.reset_peak_memory_stats() is called BEFORE train_static.
    volume_t is pre-allocated and held in memory throughout the sweep, so it
    does NOT appear in the per-budget peak measurement (it predates the reset).
    The measurement captures: the internal volume_t copy created by train_static
    + all model parameters + gradients + Adam optimizer states. This is the
    memory cost of actually running the training, which is what matters for
    GPU capacity planning.
    """
    use_cuda = (device == 'cuda') and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # Build init_scale for anisotropic real-data init
    if args.init_scale_z is not None:
        init_scale = (args.init_scale_xy, args.init_scale_xy, args.init_scale_z)
    else:
        init_scale = args.init_scale_xy

    # ---- train ---------------------------------------------------------------
    t_start = time.time()
    gs, history = train_static(
        vol,
        num_gaussians=budget,
        iterations=args.iterations,
        batch_size=args.batch_size,
        init_strategy=INIT_STRATEGY,
        init_scale=init_scale,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        patience=args.patience,
        min_delta=args.min_delta,
        stop_metric=args.stop_metric,
        device=device,
    )
    fit_time = time.time() - t_start

    # ---- peak GPU memory -----------------------------------------------------
    peak_mb = 0.0
    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)

    # ---- PSNR + MAE (same coordinates, fixed seed) ---------------------------
    metrics = evaluate_metrics(
        gs, volume_t,
        subsample=args.eval_subsample,
        seed=42,    # fixed across all budgets: same test voxels
    )

    # ---- 3D SSIM (full voxelization) -----------------------------------------
    print(f'  computing SSIM (full voxelization of {volume_t.shape}) ...')
    t_ssim = time.time()
    ssim_val = evaluate_ssim(gs, volume_t, win_size=7, max_voxels_3d=SSIM_MAX_VOXELS_3D)
    print(f'  SSIM = {ssim_val:.4f}  ({time.time() - t_ssim:.1f}s)')

    # ---- compression metrics -------------------------------------------------
    model_bytes = measure_model_size(gs)
    param_count = count_parameters(gs)

    # ---- convergence info from history ---------------------------------------
    summary = next(
        (h for h in reversed(history)
         if isinstance(h, dict) and 'elapsed_seconds' in h),
        {}
    )
    psnr_curve = extract_psnr_curve(history)

    return {
        'num_gaussians':         int(gs.num_gaussians),
        'final_psnr':            float(metrics['psnr']),
        'final_mae':             float(metrics['mae']),
        'final_ssim':            float(ssim_val),
        'fit_time_sec':          float(fit_time),
        'peak_gpu_memory_mb':    float(peak_mb),
        'model_size_bytes':      int(model_bytes),
        'parameter_count':       int(param_count),
        'final_iteration':       int(summary.get('final_iter', args.iterations - 1)),
        'converged':             summary.get('converged', None),
        # Improvement 1: best-checkpoint provenance
        'best_full_psnr':        summary.get('best_full_psnr'),
        'best_full_ssim':        summary.get('best_full_ssim'),
        'best_iteration':        summary.get('best_iteration'),
        # Diagnostic only: highest PSNR observed at ANY eval checkpoint, even
        # one whose state_dict was not saved (relevant when stop_metric='ssim',
        # where checkpoint selection is driven by SSIM, not PSNR). This value
        # does NOT necessarily describe the returned/restored model — use
        # best_full_psnr for that.
        'highest_psnr_observed': summary.get('highest_psnr_observed'),
        # Improvement 3: convergence diagnostics
        'convergence_metric':    summary.get('convergence_metric', 'psnr'),
        'best_metric_value':     summary.get('best_metric_value'),
        'stagnant_evaluations':  summary.get('stagnant_evaluations'),
        'total_evaluations':     summary.get('total_evaluations'),
        # Convergence curve for psnr_vs_iterations plot
        'psnr_curve':            psnr_curve,
    }


# ------------------------------------------------------------------ plotting

def _annotate(ax, xs, ys, labels, fontsize: int = 7):
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(f'{lbl:,}', xy=(x, y), xytext=(0, 7),
                    textcoords='offset points', ha='center', fontsize=fontsize,
                    color='#333333')


def _scatter_plot(
    xs, ys, labels,
    xlabel: str, ylabel: str, title: str,
    path: Path,
    logx: bool = False,
    color: str = 'steelblue',
):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, ys, 'o-', color=color, linewidth=1.8,
            markersize=6.5, markeredgewidth=0.8, markeredgecolor='white',
            zorder=3)
    _annotate(ax, xs, ys, labels)
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
    print(f'  plot  → {path}')


def _convergence_plot(results: list, budgets: list, path: Path):
    """Overlay PSNR-vs-iteration curves for all budgets on one figure.

    This is the key diagnostic for checking that each budget is fairly trained:
    - If a curve is still rising steeply at its final point → under-trained.
    - If a curve has reached a clear plateau → converged, comparison is fair.
    """
    cmap = plt.cm.viridis
    n = len(budgets)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (budget, r) in enumerate(zip(budgets, results)):
        curve = r.get('psnr_curve', [])
        if not curve:
            continue
        iters = [pt['iter'] for pt in curve]
        psnrs = [pt['psnr'] for pt in curve]
        color = cmap(i / max(n - 1, 1))
        label = f'N={budget:,}'
        if r.get('converged'):
            label += ' ✓'
        ax.plot(iters, psnrs, '-', color=color, linewidth=1.5, label=label)
        # Mark the final point
        if iters:
            marker = 'v' if not r.get('converged') else 'o'
            ax.plot(iters[-1], psnrs[-1], marker=marker, color=color,
                    markersize=7, markeredgewidth=0.8, markeredgecolor='white',
                    zorder=5)

    ax.set_xlabel('Training Iteration', fontsize=10)
    ax.set_ylabel('Full-volume PSNR (dB)', fontsize=10)
    ax.set_title('E3: Convergence Curves by Gaussian Budget\n'
                 '(● = converged,  ▼ = hit iteration cap)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot  → {path}')


def _convergence_summary_plot(results: list, budgets: list, path: Path):
    """Bar chart: training iterations used per budget

    Two series per budget:
      - Total iterations run (final_iteration + 1): how long training ran.
      - Best-checkpoint iteration: when the best quality was first achieved.

    The gap between bars answers: "how many extra iterations ran after the
    best point was reached?" A large gap suggests the patience window is wide.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(budgets))
    w = 0.38

    final_iters = [r['final_iteration'] + 1 for r in results]
    best_iters  = [
        (r.get('best_iteration') if r.get('best_iteration') is not None
         else r['final_iteration']) + 1
        for r in results
    ]

    ax.bar(x - w / 2, final_iters, w, label='Total iterations run',
           color='steelblue', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x + w / 2, best_iters, w, label='Best-checkpoint iteration',
           color='darkorange', alpha=0.85, edgecolor='white', linewidth=0.5)

    # Flag non-converged budgets
    y_max = max(final_iters) if final_iters else 1
    for i, r in enumerate(results):
        if r.get('converged') is False:
            ax.text(x[i], y_max * 1.04, '⚠', ha='center',
                    fontsize=11, color='crimson', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{b:,}' for b in budgets], fontsize=9)
    ax.set_xlabel('Gaussian Budget', fontsize=10)
    ax.set_ylabel('Training Iterations', fontsize=10)
    ax.set_title(
        'E3: Optimization Difficulty by Gaussian Budget\n'
        '(⚠ = hit iteration cap without converging)',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.6)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot  → {path}')


def make_plots(results: list, budgets: list, out_dir: Path) -> None:
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    psnrs     = [r['final_psnr']         for r in results]
    ssims     = [r['final_ssim']         for r in results]
    sizes_kb  = [r['model_size_bytes'] / 1024.0 for r in results]
    times     = [r['fit_time_sec']       for r in results]
    memories  = [r['peak_gpu_memory_mb'] for r in results]
    gaussians = [r['num_gaussians']      for r in results]

    _scatter_plot(
        gaussians, psnrs, budgets,
        xlabel='Number of Gaussians (log scale)',
        ylabel='Final PSNR (dB)',
        title='E3: Reconstruction Quality vs Gaussian Budget',
        path=plots_dir / 'psnr_vs_gaussians.png',
        logx=True,
    )
    _scatter_plot(
        gaussians, ssims, budgets,
        xlabel='Number of Gaussians (log scale)',
        ylabel='3D SSIM',
        title='E3: Structural Similarity vs Gaussian Budget',
        path=plots_dir / 'ssim_vs_gaussians.png',
        logx=True,
        color='darkorange',
    )
    _scatter_plot(
        sizes_kb, psnrs, budgets,
        xlabel='Model File Size (KB, log scale)',
        ylabel='Final PSNR (dB)',
        title='E3: Quality / Compression Pareto Front',
        path=plots_dir / 'psnr_vs_size.png',
        logx=True,
    )
    _scatter_plot(
        times, psnrs, budgets,
        xlabel='Fit Time (seconds)',
        ylabel='Final PSNR (dB)',
        title='E3: Quality vs Fit Time',
        path=plots_dir / 'psnr_vs_time.png',
        logx=False,
    )

    has_memory = any(r['peak_gpu_memory_mb'] > 0.0 for r in results)
    if has_memory:
        _scatter_plot(
            memories, psnrs, budgets,
            xlabel='Peak GPU Memory (MB)',
            ylabel='Final PSNR (dB)',
            title='E3: Quality vs Peak GPU Memory',
            path=plots_dir / 'psnr_vs_memory.png',
            logx=False,
        )
    else:
        print(f'  psnr_vs_memory.png  — skipped (CPU-only run)')

    _convergence_plot(results, budgets, plots_dir / 'psnr_vs_iterations.png')
    _convergence_summary_plot(results, budgets, plots_dir / 'convergence_summary.png')


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(
        description=(
            'E3: Gaussian budget sweep for the volsplat density fitter.\n'
            'Use --phantom for Phase A (sanity check) or --volume PATH for Phase B (primary).'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- data source ----
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--phantom', action='store_true',
                     help='Phase A: use a synthetic phantom (always available)')
    src.add_argument('--volume', type=str, metavar='PATH',
                     help='Phase B: path to a real microscopy TIFF or .npy')

    # ---- phantom knobs (only used when --phantom) ----
    p.add_argument('--shape', type=int, nargs=3, default=[32, 48, 48],
                   metavar=('D', 'H', 'W'))
    p.add_argument('--num-blobs', type=int, default=15)
    p.add_argument('--sigma', type=float, default=2.0)

    # ---- real-data knobs (only used when --volume) ----
    p.add_argument('--roi-shape', type=int, nargs=3, default=None,
                   metavar=('D', 'H', 'W'),
                   help='Center-crop real volume to this shape')
    p.add_argument('--ctc-normalize', action='store_true',
                   help='Use percentile normalization (for CTC data)')

    # ---- training ----
    p.add_argument('--iterations', type=int, default=5000,
                   help='Maximum training iterations per budget (safety cap only '
                        'when --patience > 0)')
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--init-scale-xy', type=float, default=2.0,
                   dest='init_scale_xy')
    p.add_argument('--init-scale-z', type=float, default=None,
                   dest='init_scale_z',
                   help='Z init scale (set for anisotropic real data; '
                        'e.g. 0.46 for Lund Tribolium, 0.4 for DRO). '
                        'If not set, isotropic init-scale-xy is used.')
    p.add_argument('--seed', type=int, default=0)

    # ---- convergence ----
    p.add_argument('--patience', type=int, default=10,
                   help='Early-stopping patience (# non-improving eval intervals). '
                        '0 disables early stopping (fixed iterations). '
                        'Recommended: 8-12 for phantom, 10-15 for real data.')
    p.add_argument('--min-delta', type=float, default=0.05,
                   help='Min improvement in stop-metric to count as progress. '
                        'For PSNR (dB): 0.05 is typical. '
                        'For SSIM (0-1): consider 0.002.')
    p.add_argument('--eval-every', type=int, default=200,
                   help='Full-volume eval interval (iterations). '
                        'patience × eval-every = total stagnant iterations before stop.')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--stop-metric', type=str, default='psnr',
                   choices=['psnr', 'ssim'], dest='stop_metric',
                   help='Metric used for convergence / patience decisions. '
                        '"psnr" (default): fast, no extra compute per eval. '
                        '"ssim": more accurate but calls evaluate_ssim() at every '
                        'eval checkpoint — O(D·H·W·G) per check. Only use with '
                        'small volumes or large --eval-every.')

    # ---- sweep ----
    p.add_argument('--budgets', type=int, nargs='+', default=BUDGETS_DEFAULT,
                   metavar='N')
    p.add_argument('--eval-subsample', type=int, default=100_000,
                   help='Voxels for PSNR/MAE evaluation (same set for all budgets).')

    # ---- output ----
    p.add_argument('--out-dir', type=str, default='runs/e3_budget')

    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    budgets = sorted(args.budgets)

    print(f'Device         : {device}')
    print(f'Init strategy  : {INIT_STRATEGY}  (E2 winner, held constant)')
    print(f'Convergence    : patience={args.patience}, min_delta={args.min_delta}, '
          f'stop_metric={args.stop_metric}, eval_every={args.eval_every} iters')
    print(f'Iteration cap  : {args.iterations}')
    print(f'Budgets        : {budgets}')

    # ---- load data ----
    vol = load_data(args)
    print(f'\nVolume shape   : {vol.shape}')
    n_voxels = vol.shape[0] * vol.shape[1] * vol.shape[2]
    dense_size_mb = n_voxels * 4 / (1024 ** 2)
    print(f'Dense voxel size (float32): {dense_size_mb:.1f} MB  '
          f'({n_voxels:,} voxels) — Gaussian rep target to beat')

    if n_voxels > SSIM_MAX_VOXELS_3D:
        print(f'[ssim] {n_voxels:,} > {SSIM_MAX_VOXELS_3D:,} threshold — '
              f'will use slice-wise 2D SSIM')

    # Pre-build volume tensor once — reused by all evaluation calls.
    # train_static creates its own internal copy; this one is for post-training eval only.
    volume_t = torch.from_numpy(vol.astype(np.float32)).to(device)

    # ---- sweep ----
    print(f'\n{"="*60}')
    results = []
    for budget in budgets:
        print(f'\n[N={budget:,}]  {INIT_STRATEGY} init, '
              f'patience={args.patience}, max_iters={args.iterations}')
        r = run_one_budget(vol, volume_t, budget, args, device)
        results.append(r)

        conv_status = (
            f'converged at iter {r["final_iteration"]}'
            if r['converged']
            else f'hit cap ({r["final_iteration"]+1} iters)'
            if r['converged'] is not None
            else f'{r["final_iteration"]+1} iters (no patience)'
        )
        print(
            f'  PSNR = {r["final_psnr"]:.2f} dB  '
            f'SSIM = {r["final_ssim"]:.4f}  '
            f'MAE = {r["final_mae"]:.4f}\n'
            f'  time = {r["fit_time_sec"]:.1f}s  '
            f'mem = {r["peak_gpu_memory_mb"]:.1f} MB  '
            f'size = {r["model_size_bytes"]//1024} KB  '
            f'params = {r["parameter_count"]:,}\n'
            f'  → {conv_status}'
        )

    # ---- convergence validity gate + report ----------------------------------
    print(f'\n{"="*60}')
    under_trained_budgets = [
        b for b, r in zip(budgets, results)
        if r.get('converged') is False
    ]
    if under_trained_budgets:
        print(
            f'[WARNING] Budgets hit iteration cap WITHOUT converging: '
            f'{under_trained_budgets}\n'
            f'  Their PSNR/SSIM may be underestimated — increase --iterations '
            f'or --patience and re-run for a valid comparison.'
        )
    else:
        print('[OK] All budgets converged or patience was disabled.')
    # ---- save report ---------------------------------------------------------
    report = {
        # Improvement 6: scientific caveats that survive independent of plots/code
        'experiment_notes': {
            'stop_metric':                  args.stop_metric,
            'phantom_used_as_sanity_check': args.phantom,
            'primary_dataset':              'phantom' if args.phantom else str(args.volume),
            'metrics_reported':             ['PSNR', 'MAE', 'SSIM'],
            'evaluation_basis':             'best_checkpoint',
            'evaluation_note': (
                'final_psnr / final_mae / final_ssim are measured on the '
                'best-checkpoint model (restored by train_static before return). '
                'best_full_psnr comes from periodic eval-every checkpoints '
                '(50k-voxel subsample, unseeded); final_psnr from evaluate_metrics '
                '(100k-voxel, seed=42). Minor numerical differences between them '
                'are expected from different sampling — they measure the same model.'
            ),
            'ssim_method': (
                '3D sliding-window SSIM via skimage for volumes ≤5M voxels; '
                'slice-wise 2D SSIM (averaged over z) for larger volumes'
            ),
            'convergence_warning': (
                f'Budgets {under_trained_budgets} hit the iteration cap without '
                f'converging. Their metrics may be underestimated. '
                f'Increase --iterations or --patience and re-run.'
            ) if under_trained_budgets else None,
        },
        'config': {
            'data_source':    'phantom' if args.phantom else str(args.volume),
            'volume_shape':   list(vol.shape),
            'n_voxels':       int(n_voxels),
            'dense_size_mb':  float(dense_size_mb),
            'budgets':        budgets,
            'iterations_cap': args.iterations,
            'patience':       args.patience,
            'min_delta':      args.min_delta,
            'eval_every':     args.eval_every,
            'stop_metric':    args.stop_metric,
            'init_strategy':  INIT_STRATEGY,
            'init_scale_xy':  args.init_scale_xy,
            'init_scale_z':   args.init_scale_z,
            'batch_size':     args.batch_size,
            'seed':           args.seed,
            'eval_subsample': args.eval_subsample,
            'device':         device,
        },
        'results': results,
    }

    report_path = out / 'e3_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport  → {report_path}')

    # ---- plots ---------------------------------------------------------------
    print('\nGenerating plots ...')
    make_plots(results, budgets, out)

    # ---- summary table -------------------------------------------------------
    print('\n--- E3 summary ---')
    hdr = (f'{"budget":>8}  {"PSNR":>8}  {"best PSNR":>9}  {"SSIM":>7}  '
           f'{"MAE":>8}  {"time(s)":>8}  {"mem(MB)":>7}  '
           f'{"size(KB)":>8}  {"iters":>6}  {"best@":>6}  {"cvg":>5}')
    print(hdr)
    print('-' * len(hdr))
    for b, r in zip(budgets, results):
        cvg     = ('yes' if r['converged']
                   else 'CAP' if r['converged'] is False
                   else ' — ')
        best_p  = r.get('best_full_psnr')
        best_it = r.get('best_iteration')
        best_p_str  = f'{best_p:9.2f}' if best_p  is not None else f'{"—":>9}'
        best_it_str = f'{best_it+1:6}' if best_it is not None else f'{"—":>6}'
        print(
            f'{b:>8,}  '
            f'{r["final_psnr"]:>8.2f}  '
            f'{best_p_str}  '
            f'{r["final_ssim"]:>7.4f}  '
            f'{r["final_mae"]:>8.4f}  '
            f'{r["fit_time_sec"]:>8.1f}  '
            f'{r["peak_gpu_memory_mb"]:>7.1f}  '
            f'{r["model_size_bytes"]//1024:>8}  '
            f'{r["final_iteration"]+1:>6}  '
            f'{best_it_str}  '
            f'{cvg:>5}'
        )

    print(f'\nE3 complete. All outputs in: {out}')


if __name__ == '__main__':
    main()
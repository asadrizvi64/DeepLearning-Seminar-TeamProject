"""E1 - integration-bias study: voxel-query vs alpha-blending projection.

Trains two supervisors on the same phantom with matched budgets:
  [A] Voxel-query (P1 default; expected unbiased)
  [B] Alpha-blending projection (the gsplat-style render-and-compare; expected
      to inflate amplitudes by ~sqrt(2*pi)*sigma per the closed-form prediction)

Reports density-field MAE, RMSE, and signed mean error stratified by Gaussian-
overlap count, and the ratio of recovered density to true density at each
ground-truth blob center (the falsifiable bias prediction).

Usage:
    python scripts/run_e1.py --out-dir runs/e1_v1
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from volsplat.data import generate_phantom
from volsplat.train import train_static, train_static_projection, evaluate_full
from volsplat.bias import bias_diagnostic


def _last_proj_psnr(history) -> float:
    """Pull the last logged projection PSNR from a train_static_projection history."""
    for h in reversed(history):
        if isinstance(h, dict) and 'proj_psnr' in h:
            return float(h['proj_psnr'])
    return float('nan')


@torch.no_grad()
def amp_inflation(gs, volume_t, blob_positions: np.ndarray) -> dict:
    """At each ground-truth blob center, compare pred density vs the volume value.

    Voxel-query model should give ratio ~1; alpha-blend model should give the
    closed-form inflation factor.
    """
    centers = torch.from_numpy(blob_positions).float().to(gs.positions.device)
    pred = gs.query_density(centers).cpu().numpy()
    D, H, W = volume_t.shape
    xs = centers[:, 0].round().long().clamp(0, W - 1)
    ys = centers[:, 1].round().long().clamp(0, H - 1)
    zs = centers[:, 2].round().long().clamp(0, D - 1)
    target = volume_t[zs, ys, xs].cpu().numpy()
    ratios = pred / np.clip(target, 1e-6, None)
    return {
        'mean_ratio':   float(np.mean(ratios)),
        'median_ratio': float(np.median(ratios)),
        'mean_pred':    float(np.mean(pred)),
        'mean_target':  float(np.mean(target)),
        'n':            int(len(ratios)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=str, default='runs/e1')
    p.add_argument('--shape', type=int, nargs=3, default=[32, 48, 48],
                   metavar=('D', 'H', 'W'))
    p.add_argument('--num-blobs', type=int, default=20)
    p.add_argument('--sigma', type=float, default=2.0,
                   help='Isotropic per-blob sigma in voxel units (~fixed via '
                        'a narrow blob_scale_range to keep the closed-form '
                        'bias prediction clean)')
    p.add_argument('--num-gaussians', type=int, default=500)
    p.add_argument('--iterations', type=int, default=800)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- phantom
    vol, params = generate_phantom(
        shape=tuple(args.shape),
        num_blobs=args.num_blobs,
        blob_scale_range=(args.sigma, args.sigma + 1e-6),  # ~fixed sigma
        blob_amp_range=(0.8, 1.2),
        seed=args.seed,
        return_params=True,
    )
    expected_factor = math.sqrt(2 * math.pi) * args.sigma
    print(f'Phantom: shape={vol.shape}, sigma~{args.sigma}, '
          f'#blobs={args.num_blobs}, range=[{vol.min():.3f},{vol.max():.3f}]')
    print(f'Closed-form predicted inflation for alpha-blend: '
          f'sqrt(2*pi)*sigma = {expected_factor:.3f}x')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    volume_t = torch.from_numpy(vol.astype(np.float32)).to(device)

    # ------------------------------------------------------------ train A: VQ
    print(f'\n[A] Voxel-query supervisor: {args.iterations} iters, '
          f'{args.num_gaussians} Gaussians')
    gs_vq, hist_vq = train_static(
        vol, num_gaussians=args.num_gaussians, iterations=args.iterations,
        seed=args.seed, eval_every=0,
    )
    psnr_vq = evaluate_full(gs_vq, volume_t, subsample=100_000)
    print(f'  final density PSNR = {psnr_vq:.2f} dB')

    # ------------------------------------------------------------ train B: AB
    print(f'\n[B] Alpha-blending projection supervisor: {args.iterations} iters, '
          f'{args.num_gaussians} Gaussians')
    gs_ab, hist_ab = train_static_projection(
        vol, num_gaussians=args.num_gaussians, iterations=args.iterations,
        seed=args.seed, eval_every=0,
    )
    psnr_ab = evaluate_full(gs_ab, volume_t, subsample=100_000)
    proj_psnr_ab = _last_proj_psnr(hist_ab)
    print(f'  final density PSNR  = {psnr_ab:.2f} dB')
    print(f'  final projection PSNR (for context) = {proj_psnr_ab:.2f} dB')

    # ---------------------------------------------------------------- diagnose
    diag_vq = bias_diagnostic(gs_vq, volume_t, n_samples=50_000, seed=args.seed)
    diag_ab = bias_diagnostic(gs_ab, volume_t, n_samples=50_000, seed=args.seed)
    amp_vq = amp_inflation(gs_vq, volume_t, params['positions'])
    amp_ab = amp_inflation(gs_ab, volume_t, params['positions'])

    # ----------------------------------------------------------------- report
    def _print(name, diag, psnr_val, amp):
        print(f'\n=== {name} ===')
        print(f'  density-field PSNR  = {psnr_val:6.2f} dB')
        print(f'  overall MAE         = {diag["mae"]:.4f}')
        print(f'  overall RMSE        = {diag["rmse"]:.4f}')
        print(f'  overall signed mean = {diag["signed_mean"]:+.4f}')
        print(f'  at-blob density ratio: mean={amp["mean_ratio"]:.3f}x '
              f'median={amp["median_ratio"]:.3f}x '
              f'(predicted ~{expected_factor:.3f}x for alpha-blend)')
        print(f'  per-overlap signed mean:')
        for row in diag['per_overlap']:
            print(f'    overlap {row["bin"]:>5}: n={row["count"]:>6}, '
                  f'signed={row["signed_mean"]:+.4f}, mae={row["mae"]:.4f}')

    _print('A: voxel-query', diag_vq, psnr_vq, amp_vq)
    _print('B: alpha-blend', diag_ab, psnr_ab, amp_ab)

    # ------------------------------------------------------------------- save
    report = {
        'config': vars(args),
        'expected_inflation_factor': expected_factor,
        'voxel_query': {
            'final_density_psnr': float(psnr_vq),
            'diagnostic':         diag_vq,
            'at_blob_ratio':      amp_vq,
        },
        'alpha_blend': {
            'final_density_psnr':    float(psnr_ab),
            'final_projection_psnr': float(proj_psnr_ab),
            'diagnostic':            diag_ab,
            'at_blob_ratio':         amp_ab,
        },
    }
    with open(out / 'e1_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    def _save(gs, path):
        torch.save({
            'positions':   gs.positions.detach().cpu(),
            'log_scales':  gs.log_scales.detach().cpu(),
            'quaternions': gs.quaternions.detach().cpu(),
            'amp_logits':  gs.amp_logits.detach().cpu(),
        }, path)
    _save(gs_vq, out / 'gs_vq.pt')
    _save(gs_ab, out / 'gs_ab.pt')
    print(f'\nE1 report -> {out / "e1_report.json"}')


if __name__ == '__main__':
    main()

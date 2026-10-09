"""Budget-matched comparison of the three temporal representations, with a held-out test.

Fixes two problems of compare_temporal.py (E7): there all variants used the same number of
Gaussians, so shared-identity stored 3.4x more parameters; and its "interpolation PSNR"
compared a half-frame render with a neighbouring *training* frame.

Here a 4D phantom with 2T-1 frames is generated; every method is fitted on the T even frames
only and scored on the T-1 odd frames it never saw. All three use about the same number of
stored parameters (shared-identity: T sets of N Gaussians x 11 numbers; native-4D: 13 per
Gaussian; deformation: 17 per Gaussian). Shared-identity has no continuous time, so its
held-out score is that of the nearest earlier frame (what a frame slider shows) and of a
50/50 cross-fade of the two neighbouring frames.

    python scripts/temporal_matched.py --seeds 0 1 2 --out runs/temporal_matched
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from volsplat.losses import psnr
from volsplat.temporal import (
    generate_phantom_4d, fit_timeseries_shared_identity, fit_native_4d, fit_deformation,
    temporal_smoothness, frame_to_frame_consistency,
)


def _params(module):
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def _grid(shape):
    D, H, W = shape
    z, y, x = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij')
    return torch.stack([x, y, z], -1).reshape(-1, 3).float()


@torch.no_grad()
def _render(model, pts, t=None):
    return (model.query_density(pts) if t is None else model.query_density(pts, float(t))).cpu().numpy()


def run(seed, args):
    full, tracks = generate_phantom_4d(shape=tuple(args.shape), num_blobs=args.num_blobs,
                                       num_frames=2 * args.frames - 1,
                                       division_frame=args.division_frame, seed=seed)
    train, held = full[0::2], full[1::2]            # fit on even frames, test on odd ones
    pts = _grid(train[0].shape)
    flat = lambda v: v.reshape(-1)
    T, N = len(train), args.num_gaussians
    out = {'seed': seed, 'blobs_per_frame': [int(t.shape[0]) for t in tracks]}

    budget = T * N * 11                               # numbers stored by shared identity
    if 'si' in args.variants:
        t0 = time.time()
        sets, _ = fit_timeseries_shared_identity(train, num_gaussians=N, iters_first=1200,
                                                 iters_warm=300, seed=seed)
        si_time = time.time() - t0
        si_r = [_render(s.cpu(), pts) for s in sets]
        out['shared_identity'] = {
            'params': sum(_params(s) for s in sets), 'gaussians': N, 'fit_s': si_time,
            'fit_psnr': [psnr(si_r[k], flat(train[k])) for k in range(T)],
            'held_nearest_psnr': [psnr(si_r[k], flat(held[k])) for k in range(T - 1)],
            'held_crossfade_psnr': [psnr(0.5 * (si_r[k] + si_r[k + 1]), flat(held[k])) for k in range(T - 1)],
            'smoothness_vox': temporal_smoothness(sets),
            'consistency_psnr': frame_to_frame_consistency(sets, train, device='cpu'),
        }
    # matched storage, and a control with the same N per model as shared identity per frame
    todo = [('native_4d', fit_native_4d, int(round(budget / 13)), '4d'),
            ('deformation', fit_deformation, int(round(budget / 17)), 'deform'),
            ('native_4d_small', fit_native_4d, N, '4d_small'),
            ('deformation_small', fit_deformation, N, 'deform_small')]
    for name, fit, n, flag in todo:
        if flag not in args.variants:
            continue
        t0 = time.time()
        m = fit(train, num_gaussians=n, iterations=args.iters, seed=seed).cpu()
        out[name] = {
            'params': _params(m), 'gaussians': n, 'fit_s': time.time() - t0,
            'fit_psnr': [psnr(_render(m, pts, k), flat(train[k])) for k in range(T)],
            'held_psnr': [psnr(_render(m, pts, k + 0.5), flat(held[k])) for k in range(T - 1)],
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shape', type=int, nargs=3, default=[16, 32, 32])
    p.add_argument('--frames', type=int, default=4, help='training frames (2*frames-1 generated)')
    p.add_argument('--num-blobs', type=int, default=8)
    p.add_argument('--num-gaussians', type=int, default=300, help='per frame, shared-identity')
    p.add_argument('--division-frame', type=int, default=4, help='in the full 2T-1 sequence')
    p.add_argument('--iters', type=int, default=2500, help='native-4D and deformation')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--variants', nargs='+', default=['si', '4d', 'deform'],
                   choices=['si', '4d', 'deform', '4d_small', 'deform_small'],
                   help='*_small: same N as one shared-identity frame (control for N)')
    p.add_argument('--out', type=str, default='runs/temporal_matched')
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    results = []
    for s in args.seeds:
        r = run(s, args)
        results.append(r)
        print(f"seed {s}: " + "  ".join(
            f"{k} fit {np.mean(v['fit_psnr']):.1f} held "
            f"{np.mean(v.get('held_psnr', v.get('held_nearest_psnr'))):.1f} dB ({v['params']} params)"
            for k, v in r.items() if isinstance(v, dict)), flush=True)
        with open(out / 'report.json', 'w') as f:     # after every seed: fits take ~1 h on a CPU
            json.dump({'config': vars(args), 'runs': results}, f, indent=2)
    print('->', out / 'report.json')


if __name__ == '__main__':
    main()

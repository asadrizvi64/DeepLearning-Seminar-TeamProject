"""P1 exit criterion: fit a GaussianSet to the synthetic phantom end-to-end.

Run from the project root:

    python scripts/make_phantom.py --shape 64 128 128 --num-blobs 50
    python scripts/train_phantom.py --num-gaussians 2000 --iterations 3000

Outputs in runs/p1_phantom/:
    final.pt      - final Gaussian set parameters
    history.json  - training log
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from volsplat.data import load_volume
from volsplat.train import train_static, evaluate_full


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--volume', type=str, default='data/phantom/phantom.tif')
    p.add_argument('--out-dir', type=str, default='runs/p1_phantom')
    p.add_argument('--num-gaussians', type=int, default=2000)
    p.add_argument('--iterations', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--init-scale', type=float, default=2.0)
    p.add_argument('--densify-every', type=int, default=0, help='0 disables densification')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol = load_volume(args.volume)
    print(f'Loaded volume: shape={vol.shape}, range=[{vol.min():.4f}, {vol.max():.4f}]')

    gs, history = train_static(
        volume=vol,
        num_gaussians=args.num_gaussians,
        iterations=args.iterations,
        batch_size=args.batch_size,
        densify_every=args.densify_every,
        init_scale=args.init_scale,
        seed=args.seed,
    )

    # Final full-volume eval
    volume_t = torch.from_numpy(vol.astype(np.float32)).to(gs.positions.device)
    final_psnr = evaluate_full(gs, volume_t, subsample=100_000)
    history.append({'final_psnr': float(final_psnr)})

    # Save checkpoint
    torch.save({
        'positions':   gs.positions.detach().cpu(),
        'log_scales':  gs.log_scales.detach().cpu(),
        'quaternions': gs.quaternions.detach().cpu(),
        'amp_logits':  gs.amp_logits.detach().cpu(),
        'num_gaussians': gs.num_gaussians,
    }, out_dir / 'final.pt')

    # Save history
    def _default(o):
        if hasattr(o, 'item'):
            return o.item()
        return str(o)

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2, default=_default)

    print()
    print(f'Done. Final N={gs.num_gaussians}, final full-volume PSNR={final_psnr:.2f} dB')
    print(f'Outputs in: {out_dir}')


if __name__ == '__main__':
    main()

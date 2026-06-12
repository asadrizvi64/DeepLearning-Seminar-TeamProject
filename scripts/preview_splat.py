"""Render a static 3D preview of a fitted GaussianSet (point cloud of centers).

Three orthographic panels (XY, XZ, YZ), each point a Gaussian center colored by
amplitude and sized by mean scale. A quick "what do the splats look like" check
that needs no browser.

Usage:
    python scripts/preview_splat.py --checkpoint runs/ctc_t000_gpu/final.pt \
        --out runs/ctc_t000_gpu/splat_preview.png
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--max-points', type=int, default=20000,
                   help='Subsample for legibility if more Gaussians than this')
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    pos = ckpt['positions'].float().numpy()
    scale = torch.exp(ckpt['log_scales'].float()).mean(-1).numpy()
    amp = F.softplus(ckpt['amp_logits'].float()).numpy()

    if pos.shape[0] > args.max_points:
        idx = np.random.choice(pos.shape[0], args.max_points, replace=False)
        pos, scale, amp = pos[idx], scale[idx], amp[idx]

    c = np.clip(amp / (np.percentile(amp, 99) or 1.0), 0, 1)
    s = 2 + 18 * (scale / (scale.max() or 1.0))

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / 'splat_preview.png'
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (a, b, an, bn) in zip(axes, [(0, 1, 'x', 'y'), (0, 2, 'x', 'z'), (1, 2, 'y', 'z')]):
        ax.scatter(pos[:, a], pos[:, b], c=c, s=s, cmap='magma', alpha=0.6, linewidths=0)
        ax.set_xlabel(an); ax.set_ylabel(bn); ax.set_title(f'{an}{bn} view')
        ax.set_aspect('equal'); ax.invert_yaxis()
    fig.suptitle(f'{pos.shape[0]} Gaussian splats (color = amplitude, size = scale)')
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=110)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

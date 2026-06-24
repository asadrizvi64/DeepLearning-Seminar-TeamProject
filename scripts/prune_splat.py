"""Drop oversized 'floater' Gaussians from a .splat file (post-hoc display cleanup).

3DGS fits on dense microscopy frequently grow a handful of giant, thin Gaussians
that model the smooth background/DC level rather than any nucleus. They dominate
the viewer as a glowing blob while contributing little real structure. This reads
the antimatter15 .splat binary directly (32 bytes/Gaussian: pos[3]f32, scale[3]f32,
rgba[4]u8, rot[4]u8), removes Gaussians whose RMS scale exceeds a threshold, and
writes a cleaned .splat. No checkpoint or GPU needed.

Usage:
    python scripts/prune_splat.py in.splat out.splat --max-scale 8
    python scripts/prune_splat.py splats/ cleaned/ --max-scale 8   # whole folder
"""
import argparse
from pathlib import Path

import numpy as np

SP = 32  # bytes per splat


def prune_one(src: Path, dst: Path, max_scale: float) -> tuple[int, int]:
    raw = np.fromfile(src, dtype=np.uint8)
    n = len(raw) // SP
    rec = raw[: n * SP].reshape(n, SP)
    scl = rec[:, 12:24].copy().view(np.float32).reshape(n, 3)
    smag = np.sqrt((scl ** 2).sum(1) / 3.0)          # RMS scale per Gaussian
    keep = smag <= max_scale
    dst.parent.mkdir(parents=True, exist_ok=True)
    rec[keep].tofile(dst)
    return int(keep.sum()), int((~keep).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src', help='input .splat file or a directory of them')
    p.add_argument('dst', help='output .splat file or directory')
    p.add_argument('--max-scale', type=float, default=8.0,
                   help='drop Gaussians with RMS scale (voxels) above this')
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if src.is_dir():
        files = sorted(src.glob('*.splat'))
        if not files:
            print(f'No .splat files in {src}')
            return
        for f in files:
            kept, dropped = prune_one(f, dst / f.name, args.max_scale)
            print(f'{f.name}: kept {kept}, dropped {dropped} floater(s)')
    else:
        kept, dropped = prune_one(src, dst, args.max_scale)
        print(f'{src.name}: kept {kept}, dropped {dropped} floater(s) -> {dst}')


if __name__ == '__main__':
    main()

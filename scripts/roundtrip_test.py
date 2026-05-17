"""P0 exit criterion: generate a phantom, save it, reload it, verify identity within tolerance.

Run from the project root:

    python scripts/roundtrip_test.py

Should print 'P0 round-trip: PASS' on success.
"""
import argparse

import numpy as np

from volsplat.data import generate_phantom, save_volume, load_volume


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, default='data/phantom/_roundtrip.tif')
    p.add_argument('--shape', type=int, nargs=3, default=[32, 64, 64])
    p.add_argument('--num-blobs', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    vol = generate_phantom(
        shape=tuple(args.shape),
        num_blobs=args.num_blobs,
        seed=args.seed,
    )
    save_volume(vol, args.path)
    reloaded = load_volume(args.path)

    # load_volume normalizes to [0, 1]; replicate the same normalization on the original
    # for a fair comparison.
    vmin, vmax = float(vol.min()), float(vol.max())
    normed = (vol - vmin) / (vmax - vmin) if vmax > vmin else vol

    assert vol.shape == reloaded.shape, f'shape mismatch: {vol.shape} vs {reloaded.shape}'
    diff = float(np.abs(normed - reloaded).max())
    print(f'shape match: OK  ({vol.shape})')
    print(f'max |diff| after normalization: {diff:.3e}')

    tol = 1e-5
    assert diff < tol, f'round-trip failed: max diff {diff:.3e} >= tol {tol}'
    print('P0 round-trip: PASS')


if __name__ == '__main__':
    main()

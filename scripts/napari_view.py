"""View a single 3D microscopy volume in napari with correct voxel anisotropy.

Per-dataset voxel sizes are looked up automatically. napari's `scale` parameter takes
(z, y, x) in microns, so the embryo isn't visually squashed along z.

Usage:
    # Tribolium (Lund) - 4.33x z-anisotropy
    python scripts/napari_view.py --path /path/to/lund_i000022_oi_000096.tif \\
        --dataset tribolium

    # CTC Drosophila - 5x z-anisotropy
    python scripts/napari_view.py --path /path/to/Fluo-N3DL-DRO/01/t000.tif \\
        --dataset drosophila

    # Custom voxel sizes
    python scripts/napari_view.py --path some.tif --voxel-zyx 3.0 0.7 0.7

Install
-------
    pip install "napari[all]"

    # On Linux you may also need:
    sudo apt install libxcb-cursor0
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, required=True)
    p.add_argument(
        '--dataset', type=str, default='tribolium',
        choices=['tribolium', 'drosophila', 'custom'],
        help='Used to look up voxel size if --voxel-zyx is not given',
    )
    p.add_argument(
        '--voxel-zyx', type=float, nargs=3, default=None,
        help='Voxel size (z, y, x) in microns; overrides --dataset',
    )
    p.add_argument('--no-normalize', action='store_true')
    p.add_argument(
        '--gamma', type=float, default=1.0,
        help='napari contrast gamma (try 0.5 to brighten faint structure)',
    )
    args = p.parse_args()

    try:
        import napari
    except ImportError:
        print('napari is not installed. Install with:')
        print('    pip install "napari[all]"')
        print('On Linux you may also need:  sudo apt install libxcb-cursor0')
        return

    # Resolve voxel size
    if args.voxel_zyx is not None:
        scale_zyx = tuple(args.voxel_zyx)
    elif args.dataset == 'tribolium':
        from volsplat.tribolium import VOXEL_SIZE_LUND_UM
        vx, vy, vz = VOXEL_SIZE_LUND_UM
        scale_zyx = (vz, vy, vx)
    elif args.dataset == 'drosophila':
        from volsplat.ctc import VOXEL_SIZE_DRO_UM
        vx, vy, vz = VOXEL_SIZE_DRO_UM
        scale_zyx = (vz, vy, vx)
    else:
        scale_zyx = (1.0, 1.0, 1.0)
        print('Using isotropic scale (1, 1, 1). Pass --voxel-zyx to set physical units.')

    # Load
    print(f'Loading {args.path}...')
    if args.dataset == 'tribolium':
        from volsplat.tribolium import load_lund_volume
        vol = load_lund_volume(args.path, normalize=not args.no_normalize)
    else:
        import tifffile
        vol = tifffile.imread(str(args.path)).astype(np.float32)
        while vol.ndim > 3 and vol.shape[0] == 1:
            vol = vol[0]
        if not args.no_normalize:
            from volsplat.ctc import percentile_normalize
            vol = percentile_normalize(vol)

    print(f'  shape (D, H, W) = {vol.shape}')
    print(f'  range = [{vol.min():.4f}, {vol.max():.4f}]')
    print(f'  voxel size (z, y, x) microns = {scale_zyx}')

    viewer = napari.Viewer()
    viewer.add_image(
        vol,
        name=Path(args.path).stem,
        scale=scale_zyx,
        colormap='magma',
        contrast_limits=[float(vol.min()), float(vol.max())],
        gamma=args.gamma,
    )
    # Default to a 3D rendered view rather than the 2D slice view
    viewer.dims.ndisplay = 3
    napari.run()


if __name__ == '__main__':
    main()

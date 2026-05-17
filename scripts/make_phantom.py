"""Generate a synthetic phantom volume and save it to disk."""
import argparse
from volsplat.data import generate_phantom, save_volume


def main():
    p = argparse.ArgumentParser(description='Generate a synthetic microscopy-like phantom.')
    p.add_argument('--out', type=str, default='data/phantom/phantom.tif')
    p.add_argument('--shape', type=int, nargs=3, default=[64, 128, 128], metavar=('D', 'H', 'W'))
    p.add_argument('--num-blobs', type=int, default=50)
    p.add_argument('--noise-std', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    vol = generate_phantom(
        shape=tuple(args.shape),
        num_blobs=args.num_blobs,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    save_volume(vol, args.out)
    print(f'Wrote phantom: {args.out}')
    print(f'  shape = {vol.shape}')
    print(f'  range = [{vol.min():.4f}, {vol.max():.4f}]')
    print(f'  mean  = {vol.mean():.4f}')


if __name__ == '__main__':
    main()

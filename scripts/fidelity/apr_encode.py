"""Adaptive particle representation (APR; Cheeseman et al., Nat. Commun. 9:5160, 2018) of one
raw volume, at a sweep of relative error bounds E, for scoring with the fidelity pipeline.

Runs in its own environment (pyapr pulls a different numpy/scipy):
    python -m venv aprenv && aprenv/Scripts/pip install pyapr
    aprenv/Scripts/python scripts/fidelity/apr_encode.py raw.npy --voxel 1.0 0.09 0.09 --out apr_out

Each setting is written to an .apr file, read back and reconstructed from the file, so the
byte count is what is stored on disk and the reconstruction is what a reader would get.
Particle intensities are stored either losslessly or with pyapr's lossy (within-noise)
particle compression. Writes <out>/<name>_<bytes>.npy (uint8/uint16 reconstructions, as the
raw dtype) and <out>/apr_runs.json; apr_to_recon.py turns them into recon/*.npz.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyapr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('raw')
    p.add_argument('--voxel', type=float, nargs=3, required=True, help='z y x, um')
    p.add_argument('--errors', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.4])
    p.add_argument('--sigma-th', type=float, default=None,
                   help='floor of the local intensity scale; default: pyapr auto_parameters')
    p.add_argument('--grad-th', type=float, default=None, help='gradients below are ignored')
    p.add_argument('--tag', default='apr', help='name prefix of the outputs')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    img = np.load(args.raw)
    # The integer converters overflow in their B-spline gradient filter on this data, so the
    # particle layout is computed in float and the intensities re-sampled from the raw image
    # in its own type (1 byte per particle for 8-bit data).
    Parts = {np.dtype('uint8'): pyapr.ByteParticles, np.dtype('uint16'): pyapr.ShortParticles}[img.dtype]
    runs = []
    for E in args.errors:
        par = pyapr.APRParameters()
        par.auto_parameters = args.sigma_th is None
        if args.sigma_th is not None:
            par.sigma_th, par.grad_th = args.sigma_th, args.grad_th
        par.rel_error = E
        par.dz, par.dy, par.dx = args.voxel
        t0 = time.time()
        apr, _ = pyapr.converter.get_apr(img.astype(np.float32), converter=pyapr.converter.FloatConverter(),
                                         params=par)
        parts = Parts()
        parts.sample_image(apr, img)
        t_conv = time.time() - t0
        n_part = int(apr.total_number_particles())
        for lossy in (False, True):
            name = f"{args.tag}{'q' if lossy else ''}-e{E:g}"
            f = out / f'{name}.apr'
            parts.set_compression_type(1 if lossy else 0)
            pyapr.io.write(str(f), apr, parts, write_tree=False)
            nbytes = f.stat().st_size
            apr2, parts2 = pyapr.io.read(str(f))
            rec = np.asarray(pyapr.reconstruction.reconstruct_smooth(apr2, parts2))
            rec = np.clip(np.rint(rec), 0, np.iinfo(img.dtype).max).astype(img.dtype).reshape(img.shape)
            np.save(out / f'{name}_{nbytes}.npy', rec)
            err = np.abs(rec.astype(np.float32) - img.astype(np.float32))
            runs.append(dict(name=name, rel_error=E, lossy_particles=lossy, particles=n_part,
                             computational_ratio=img.size / n_part, bytes=nbytes,
                             ratio=img.nbytes / nbytes, convert_s=t_conv,
                             max_abs_err=float(err.max()), mean_abs_err=float(err.mean())))
            print(f"{name:14s} particles {n_part:9,d} (CR {img.size / n_part:6.1f})  "
                  f"{nbytes / 1024:8.1f} KiB  {img.nbytes / nbytes:6.1f}x  "
                  f"mean|err| {err.mean():.2f}  convert {t_conv:.1f}s", flush=True)
    json.dump(runs, open(out / 'apr_runs.json', 'w'), indent=2)


if __name__ == '__main__':
    main()

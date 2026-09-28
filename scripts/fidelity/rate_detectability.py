"""Rate-detectability audit: does lossy volume compression keep the nuclei?

Every Gaussian-splat compressor for microscopy/medical volumes we found (Luxar, GSToken,
GaussianPile, fixed-budget Gaussian encoding) chooses its budget and reports its quality
with PSNR. Luxar's own notes find nuclei-masked PSNR "flat ... across a 37x range in
count". This script asks the question PSNR cannot answer: at a given number of bytes on
disk, how many REAL nuclei can a standard detector still find?

METHODS, all scored at MATCHED BYTES ON DISK against the raw uint16 volume:
    luxar_K        Luxar gsplats fit with K seeds, stored as its own .gsplats.zarr
    jpeg2k         JPEG2000 (the DICOM lossy standard), 3D, quality bisected to the size
    jpegxl         JPEG-XL per z-slice, distance bisected to the size
    zfp            ZFP fixed-rate (error-bounded scientific-data compressor)
    downsample     integer downsample + trilinear upsample, stored uncompressed

SCORING (identical for every reconstruction, including the raw volume itself):
    1. Candidates: local maxima with a physically isotropic window (microns).
    2. Two rankings of those candidates: brightness, and multi-scale scale-normalised
       LoG (sigmas fixed in advance: 1.5, 2, 2.5, 3 um) -- LoG ranking beat our own
       Gaussian-fit re-ranking, so it is the honest downstream detector.
    3. recall@N against HUMAN-annotated nuclei, one-to-one matching within 3 um
       (cellmetrics.match_indices, the max-cardinality matcher).
    4. Faint vs bright: GT nuclei split at the median of their LoG response in the RAW
       volume. recall@N reported per half -- the loss PSNR hides lives in the faint half.
    5. Survival (needs no labels): the raw volume's top-S LoG-ranked candidates are the
       reference objects; the fraction still matched by the reconstruction's top-2S.
    6. PSNR on the common normalisation, for the PSNR-vs-detectability comparison.

Reconstructions are saved (npz) so any change to scoring is a rescore, not a refit.

Usage (local CPU is slow for Luxar; use the cluster for real budgets):
    python scripts/fidelity/rate_detectability.py --out runs/fidelity/pilot_dro \
        --luxar-seeds 500 2000 8000 --n-iters 1000 --device cuda
    python scripts/fidelity/rate_detectability.py --out runs/fidelity/codecs_only \
        --target-kib 16 32 64 128            # codecs only, no Luxar
"""
import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import imagecodecs as ic
from scipy.ndimage import gaussian_laplace, zoom

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from volsplat.cellmetrics import detect_cells, match_indices  # noqa: E402

DRO = Path(r'D:/Download/train/Fluo-N3DL-DRO (1)/Fluo-N3DL-DRO')
V = np.array([2.03, 0.406, 0.406])            # um per voxel (z, y, x)
MATCH_UM = 3.0
PROMINENCE = 0.02
LOG_SIGMAS_UM = (1.5, 2.0, 2.5, 3.0)          # fixed in advance, not tuned per method
BUDGETS = (50, 100, 200)
SURVIVAL_S = 100


# ------------------------------------------------------------------ data
def load_crop(dro_root, frame, origin, shape):
    full = tifffile.imread(str(Path(dro_root) / '01' / f't{frame:03d}.tif'))
    lo, hi = np.percentile(full, 0.5), np.percentile(full, 99.5)
    z, y, x = origin
    crop = np.ascontiguousarray(full[z:z + shape[0], y:y + shape[1], x:x + shape[2]])
    return crop, float(lo), float(hi)


def manual_in_crop(origin, shape):
    cents = np.load(REPO / 'runs/dro_manual_centroids.npy')
    o = np.array(origin)
    inside = np.all((cents >= o) & (cents < o + np.array(shape)), axis=1)
    return cents[inside] - o


# ------------------------------------------------------------------ codecs
def bisect_param(encode, lo, hi, target, increasing, iters=18):
    """Find the codec parameter whose output size is closest to `target` bytes.
    `increasing`: size grows with the parameter."""
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        n = encode(mid)[1]
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (mid, n)
        if (n < target) == increasing:
            lo = mid
        else:
            hi = mid
    return best[0]


def jpeg2k(crop, target):
    enc = lambda q: (b := ic.jpeg2k_encode(crop, level=q), len(b))
    q = bisect_param(enc, 30.0, 120.0, target, increasing=True)
    b = ic.jpeg2k_encode(crop, level=q)
    return ic.jpeg2k_decode(b).astype(np.float32), len(b), {'level_db': q}


def jpegxl(crop, target):
    def enc(d):
        bs = [ic.jpegxl_encode(s, distance=d) for s in crop]
        return bs, sum(map(len, bs))
    d = bisect_param(enc, 0.05, 25.0, target, increasing=False)
    bs, n = enc(d)
    rec = np.stack([ic.jpegxl_decode(b) for b in bs]).astype(np.float32)
    return rec, n, {'distance': d}


def zfp(crop, target):
    rate = max(target * 8.0 / crop.size, 0.02)
    b = ic.zfp_encode(crop.astype(np.float32), mode='r', level=rate)
    return ic.zfp_decode(b).astype(np.float32), len(b), {'bits_per_value': rate}


def downsample(crop, target):
    best = None
    for f in range(2, 33):
        dims = tuple(max(1, s // f) for s in crop.shape)
        n = int(np.prod(dims)) * 2                      # stored as uint16
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (f, n, dims)
    f, n, dims = best
    coarse = zoom(crop.astype(np.float32), [d / s for d, s in zip(dims, crop.shape)], order=1)
    back = zoom(coarse, [s / c for s, c in zip(crop.shape, coarse.shape)], order=1)
    return back[:crop.shape[0], :crop.shape[1], :crop.shape[2]].astype(np.float32), n, {'factor': f}


CODECS = {'jpeg2k': jpeg2k, 'jpegxl': jpegxl, 'zfp': zfp, 'downsample': downsample}


# ------------------------------------------------------------------ luxar
def luxar_fit(crop, seeds, n_iters, device, zarr_path, refit):
    from luxar.gsplats import fit_gaussian_splats
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.rendering import render_to_volume
    t = time.time()
    if zarr_path.exists() and not refit:
        res = GSplatData.load(str(zarr_path))
        fit_s = float('nan')
    else:
        # output_space='voxel' is REQUIRED: with voxel_size given, Luxar's default
        # 'real' space stores centers/Cholesky in microns, and render_to_volume on a
        # voxel grid then misplaces every splat (measured: render-vs-data r = 0.08,
        # vs 0.93 once mapped back to voxels). voxel_size still informs the fit.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = fit_gaussian_splats(crop.astype(np.float32), seeds=int(seeds), n_iters=n_iters,
                                      device=device, verbose=False, voxel_size=list(V),
                                      output_space='voxel')
        fit_s = time.time() - t
        shutil.rmtree(zarr_path, ignore_errors=True)
        res.save(str(zarr_path))
    nbytes = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
    rec = render_to_volume(res, shape=crop.shape, device=device).astype(np.float32)
    align_r = float(np.corrcoef(rec.ravel(), crop.astype(np.float32).ravel())[0, 1])
    if align_r < 0.5:
        raise RuntimeError(f'Luxar render does not line up with the data (r={align_r:.2f}); '
                           'check output_space / voxel_size before trusting any score')
    # Luxar renders in its own normalised units (floor-subtracted, [0,1]); map back to
    # raw counts by least squares so every method sees the same intensity scale. The
    # map is affine, so rankings are unchanged; it only makes thresholds comparable.
    a, b = np.polyfit(rec.ravel(), crop.astype(np.float32).ravel(), 1)
    return a * rec + b, nbytes, {'seeds': int(seeds), 'n_splats': int(res.centers.shape[0]),
                                 'fit_s': fit_s, 'affine_a': float(a), 'affine_b': float(b),
                                 'align_r': align_r}


# ------------------------------------------------------------------ scoring
def normalise(vol, lo, hi):
    return np.clip((vol.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def log_response(img):
    out = None
    for s in LOG_SIGMAS_UM:
        r = -(s ** 2) * gaussian_laplace(img, sigma=tuple(s / V))
        out = r if out is None else np.maximum(out, r)
    return out


def ranked_candidates(img):
    cand = detect_cells(img, mode='3d', threshold_abs=0.0, smoothing_sigma=0.8, min_distance=2.0,
                        prominence=PROMINENCE, voxel_size_zyx=tuple(V))
    L = log_response(img)
    if len(cand) == 0:
        return cand, cand, L
    s = L[tuple(np.round(cand).astype(int).T)]
    return cand, cand[np.argsort(-s)], L


def found_mask(pred, tgt, n):
    top = pred[:n]
    hit = np.zeros(len(tgt), bool)
    if len(top) and len(tgt):
        _, ci, _ = match_indices(top * V, tgt * V, MATCH_UM)
        hit[ci] = True
    return hit


def score(img_norm, raw_norm, manual, faint_mask, ref_objs, ref_faint):
    bright_order, log_order, _ = ranked_candidates(img_norm)
    row = {'n_cand': int(len(bright_order)),
           'psnr': float(10 * np.log10(1.0 / max(np.mean((img_norm - raw_norm) ** 2), 1e-12)))}
    for n in BUDGETS:
        hb = found_mask(bright_order, manual, n)
        hl = found_mask(log_order, manual, n)
        row[f'gt_bright_rank@{n}'] = int(hb.sum())
        row[f'gt_log_rank@{n}'] = int(hl.sum())
        row[f'gt_log_rank@{n}_faint'] = int(hl[faint_mask].sum())
        row[f'gt_log_rank@{n}_bright'] = int(hl[~faint_mask].sum())
    surv = found_mask(log_order, ref_objs, 2 * SURVIVAL_S)
    row['survival'] = float(surv.mean())
    row['survival_faint'] = float(surv[ref_faint].mean())
    row['survival_bright'] = float(surv[~ref_faint].mean())
    return row


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--dro', default=str(DRO))
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--origin', type=int, nargs=3, default=[32, 416, 544])
    p.add_argument('--shape', type=int, nargs=3, default=[64, 128, 128])
    p.add_argument('--luxar-seeds', type=int, nargs='*', default=[])
    p.add_argument('--n-iters', type=int, default=1000)
    p.add_argument('--device', default='cpu')
    p.add_argument('--refit', action='store_true')
    p.add_argument('--target-kib', type=float, nargs='*', default=[])
    p.add_argument('--codecs', nargs='*', default=list(CODECS))
    args = p.parse_args()
    out = Path(args.out)
    (out / 'recon').mkdir(parents=True, exist_ok=True)
    (out / 'luxar').mkdir(parents=True, exist_ok=True)

    crop, lo, hi = load_crop(args.dro, args.frame, args.origin, args.shape)
    manual = manual_in_crop(args.origin, args.shape) if args.frame == 0 else np.zeros((0, 3))
    raw_norm = normalise(crop, lo, hi)
    _, raw_log_order, L_raw = ranked_candidates(raw_norm)
    gt_strength = L_raw[tuple(np.clip(np.round(manual).astype(int), 0, np.array(crop.shape) - 1).T)]
    faint_mask = gt_strength <= np.median(gt_strength) if len(manual) else np.zeros(0, bool)
    ref_objs = raw_log_order[:SURVIVAL_S]
    ref_strength = L_raw[tuple(np.round(ref_objs).astype(int).T)]
    ref_faint = ref_strength <= np.median(ref_strength)
    meta = dict(origin=args.origin, shape=args.shape, frame=args.frame, n_manual=int(len(manual)),
                raw_bytes=int(crop.nbytes), norm_lo=lo, norm_hi=hi, voxel_um=V.tolist(),
                match_um=MATCH_UM, log_sigmas_um=LOG_SIGMAS_UM, survival_S=SURVIVAL_S)
    json.dump(meta, open(out / 'meta.json', 'w'), indent=2)
    print(f'crop {args.origin} {crop.shape} | {len(manual)} human nuclei '
          f'({int(faint_mask.sum())} faint) | raw {crop.nbytes / 1024:.0f} KiB', flush=True)

    rows = []

    def add(method, rec, nbytes, params):
        rn = normalise(rec, lo, hi)
        np.savez_compressed(out / 'recon' / f'{method}_{nbytes}.npz', rec=rn.astype(np.float16))
        r = score(rn, raw_norm, manual, faint_mask, ref_objs, ref_faint)
        r.update(method=method, bytes=int(nbytes), ratio=crop.nbytes / nbytes,
                 bits_per_voxel=8.0 * nbytes / crop.size, params=json.dumps(params))
        rows.append(r)
        pd.DataFrame(rows).to_csv(out / 'rate_detectability.csv', index=False)
        print(f"  {method:11s} {nbytes / 1024:8.1f} KiB {r['ratio']:7.1f}x  PSNR {r['psnr']:5.2f}  "
              f"GT@100 log {r['gt_log_rank@100']:2d} (faint {r['gt_log_rank@100_faint']:2d})  "
              f"GT@200 log {r['gt_log_rank@200']:2d}  survival {r['survival']:.2f} "
              f"(faint {r['survival_faint']:.2f})", flush=True)

    add('raw', crop.astype(np.float32), crop.nbytes, {})
    targets = [k * 1024 for k in args.target_kib]
    for K in args.luxar_seeds:
        rec, nb, prm = luxar_fit(crop, K, args.n_iters, args.device,
                                 out / 'luxar' / f'K{K}.gsplats.zarr', args.refit)
        add(f'luxar_K{K}', rec, nb, prm)
        targets.append(nb)
    for tgt in targets:
        for name in args.codecs:
            rec, nb, prm = CODECS[name](crop, tgt)
            prm['target_bytes'] = int(tgt)
            add(name, rec, nb, prm)
    print(f'\nOutputs -> {out}')


if __name__ == '__main__':
    main()

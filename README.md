# volsplat — 3DGS for Volumetric Microscopy

CMS-Team Project, TU Dresden (Chair of ML for Spatial Understanding). Onboarding
notes for the second team member — read this first, then `PLAN.md` for the research
framing.

## Status

- **P0** (env + loaders + phantom): done — `python scripts/roundtrip_test.py`.
- **P1** (static pipeline, sampled voxel-query supervision): done on phantom
  (~48 dB) and verified end-to-end on real CTC data (climbing PSNR on a real ROI).
- **CTC bring-up**: loader, percentile normalization, ROI cropping, anisotropic
  init, inspection script with per-axis MIPs, real-data training script — all in
  place and tested against the Keller-lab *Fluo-N3DL-DRO* dataset.
- **P2** (orthographic alpha-blending rasterizer + E1 integration-bias study): **done.**
  The rasterizer reproduces the closed-form R²-Gaussian bias on an isolated Gaussian
  (predicted √(2π)·σ ≈ 5.013, measured 5.018). The full E1 run shows a clean
  monotonic positive bias growing with Gaussian-overlap count for the alpha-blend
  supervisor (signed mean +0.006 at overlap 0 → +12.5 at overlap 16+), while the
  voxel-query supervisor stays unbiased (at-blob ratio 0.96× vs 14.8× for alpha-blend).
  **Decision: voxel-query is the primary supervision for P3 onward.** See `runs/e1_v1/`
  and `runs/e1_sigma*/` for the artifacts; `PLAN.md` §11 for the writeup.
- **P3** (static characterization — E2 init / E3 budget / E4 densify / E5 baselines):
  in progress. **E2 done**: three init strategies in `volsplat/init.py`
  (`random`, `intensity_weighted`, `local_maxima`); structured init beats random by
  ~7 dB final PSNR; `local_maxima` ties `intensity_weighted` on the sparse phantom
  (expected to pull ahead on dense real data). Default stays `intensity_weighted`.
  E3 budget sweep on the `MAE/Gaus_budget` branch. See `runs/e2_v1/`, `PLAN.md` §12.
- **P4** (export + viewer): **done.** `volsplat/export.py` →
  `.ply` (INRIA-3DGS) + `.splat` (antimatter15) via `scripts/export_splat.py`;
  **E6 round-trip is exact** (0.0 position/scale error). Density amplitude maps to
  grayscale + opacity so nuclei show in web viewers. `scripts/preview_splat.py` for
  a no-browser 3D preview; **`viewer/index.html`** is a self-contained WebGL splat
  viewer with a frame slider for 3D+t scrubbing (`python -m http.server`-served).
- **P5** (temporal, 3D+t): **shared-identity + native-4D done; E7 comparison built.**
  `volsplat/temporal.py`: 4D moving-blob phantom with controlled division events (E8
  testbed); **shared-identity** per-frame fitting (warm-start carries Gaussian identity,
  O(T·N), ~37 dB/frame, supports tracking); **native-4D** `GaussianSet4D` (one set with
  a time axis, O(N), continuous-t interpolation per E10); E7 metrics (temporal
  smoothness, frame-to-frame consistency). `scripts/compare_temporal.py` tabulates the
  trade-off (fidelity+tracking vs storage+interpolation); `scripts/train_temporal.py`
  exports per-frame `.splat`s for the viewer. Deformation-field variant still to come.
- **HPC**: TU Dresden Alpha (A100) wired up — `scripts/hpc/` (setup + SLURM job).
  Converged real-data fit runs there; CPU is phantom-only.

## Setup

Python 3.10+, CUDA strongly recommended for real volumes (one frame is ~96M voxels
= ~380 MB float32 — CPU is fine for the phantom only).

```bash
python -m venv .venv && source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows
pip install -e .
```

`pyproject.toml` pulls everything you need including `imagecodecs` (required —
the CTC TIFFs are LZW-compressed and `tifffile` cannot decode them without it).
`gsplat` is not a hard dependency yet; we use our own PyTorch density evaluator
through P1. It lands at P2 with the slab-projection rasterizer.

## Repo layout

```
volsplat/
  __init__.py
  gaussians.py    # GaussianSet nn.Module: positions, scales, rotations, amplitudes; query_density
  data.py         # TIFF/.npy IO, synthetic phantom, intensity-weighted sampling
  losses.py       # MSE, PSNR
  densify.py      # clone / split / prune + per-param-group Adam
  train.py        # static training loop, anisotropic init, full-volume eval
  ctc.py          # Cell Tracking Challenge loader (Fluo-N3DL-DRO targeted, generic 3D CTC)
scripts/
  roundtrip_test.py   # P0 exit criterion
  make_phantom.py     # synthetic volume generator
  train_phantom.py    # P1 exit criterion (synthetic)
  inspect_ctc.py      # MIPs + tracking-GT diagnostic on any CTC frame
  train_ctc.py        # real-data training entry point
pyproject.toml
PLAN.md             # project plan: goals, RQs, phases, experiment matrix (separate doc)
```

## Three workflows you'll actually run

### 1. P0 — verify the environment

```bash
python scripts/roundtrip_test.py
```

Should print `P0 round-trip: PASS`. Builds a 32x64x64 phantom, saves to TIFF,
reloads, bit-equal within float32 tolerance.

### 2. P1 — fit the synthetic phantom

```bash
python scripts/make_phantom.py --out data/phantom/phantom.tif --shape 64 128 128 --num-blobs 50
python scripts/train_phantom.py --volume data/phantom/phantom.tif --num-gaussians 2000 --iterations 3000
```

Outputs in `runs/p1_phantom/`: `final.pt` (parameters), `history.json` (loss/PSNR/N
over training). Target: PSNR past ~30 dB within a few thousand iterations. The
ground truth is literally a Gaussian sum so anything lower means a bug.

### 3. Real CTC data — first look, first fit

```bash
# Eyeball frame 0 (per-axis MIPs with light-sheet aspect correction + GT summary)
python scripts/inspect_ctc.py --root /path/to/Fluo-N3DL-DRO \
    --sequence 01 --frame 0 --crop --out-dir runs/inspect_t000

# Modest first fit (recenters to a fixed ROI; anisotropic init by default)
python scripts/train_ctc.py --root /path/to/Fluo-N3DL-DRO \
    --sequence 01 --frame 0 --crop --target-shape 128 256 256 \
    --num-gaussians 20000 --iterations 5000 \
    --out-dir runs/ctc_t000
```

CPU-only is impractical on real data — one iteration on a 24x80x80 ROI with 3000
Gaussians is ~2 s; you want a GPU. Anisotropic init is on by default:
`--init-scale-xy 2.0 --init-scale-z 0.4` (in voxel units) which is ~0.81 µm
isotropic in physical units for DRO's 0.406×0.406×2.03 µm voxels.

## Conventions (the only ones you have to memorize)

- **Volume shape**: `(D, H, W)` indexed as `V[z, y, x]`.
- **Gaussian positions**: `(x, y, z)` Cartesian order.
- The conversion happens once inside the loader/evaluator. Everything else uses
  `(x, y, z)`.
- Coordinates are in **voxel units**, not normalized. For very large volumes
  consider normalizing positions to `[-1, 1]` before optimizing.
- Volumes from `load_volume` are normalized to `[0, 1]` (min-max). CTC volumes
  from `load_ctc_frame` use **percentile normalization** (0.5/99.5) because
  16-bit microscopy has long-tailed intensity distributions — min-max would
  squash everything into a narrow band near zero.

## What the supervision actually does in P1

At each iteration: sample a batch of voxel coordinates (70% intensity-biased,
30% uniform), evaluate the predicted density field at those exact points, MSE
against the ground-truth voxel values. This is the simplest defensible v1 —
it's *sampled voxel-query*, not true projection rasterization.

P2 will add the two paths E1 actually compares:

1. **Slab-projection rasterizer** — render axis-aligned 2D slabs (the true
   render-and-compare path, equivalent to what `gsplat` does for images).
2. **Differentiable voxelizer** — analytic, bias-free reference (à la R²-Gaussian).

E1 then asks whether vanilla render-and-compare carries the integration bias
R²-Gaussian found on X-ray data.

## Densification

Off by default in P1 (`--densify-every 0`); the phantom has plenty of static
capacity. Enable with `--densify-every 500` to exercise clone/split/prune. The
thresholds (`grad_threshold`, `scale_threshold`, `amp_threshold` in
`densify.py`) are tuned for photographic scenes and will need re-tuning for
microscopy in P3/E4 — that's a deliberate todo, not a bug.

## Findings from the real data you should know about

These came out of running `inspect_ctc.py` + a smoke fit on Fluo-N3DL-DRO
frame 0. Worth absorbing before doing the real runs:

- **The blastoderm is a *dense shell* of nuclei**, not sparse blobs in empty
  space. After percentile-normalization: mean ~0.14, ~40% of voxels above 0.01.
  This contradicts the project plan's "sparse signal in mostly empty space"
  framing and weakens the RQ2 compression argument. Open question for the team.
- **`find_intensity_bbox` is ineffective on this data** at the default 95th
  percentile — the background is too diffuse. Z is not cropped at all, XY barely.
  Use `--target-shape D H W` to recenter to a fixed crop; don't rely on the
  bbox alone.
- **Voxel anisotropy is the dominant fact.** DRO is 0.406×0.406×2.03 µm
  (z is 5× coarser than xy). This *is* essentially the PSF. Always use
  anisotropic `init_scale` for this data — the defaults in `train_ctc.py`
  already do.
- **No division events in the DRO tracking GT.** All 189 tracks span frame 0–49
  with no parents. E8 (mitosis stress test) cannot use this GT directly — it'll
  need synthetic phantoms with controlled divisions. `inspect_ctc.py` prints
  this diagnostic automatically.

## Things I haven't done yet but you might want to

- **Densification on microscopy.** The current heuristics are 3DGS-paper
  defaults; nothing about them is microscopy-aware. Live in `densify.py`.
- **Position normalization for large volumes.** Adam learning rates are tuned
  for voxel-unit coordinates on phantom-sized volumes. On the full 125×603×1272
  frame, consider rescaling positions to `[-1, 1]` and adjusting `lr_position`
  accordingly.
- **GT-position initialization** for the shared-identity temporal variant:
  `volsplat.ctc.cell_centroids_from_labels(label_map)` returns the centroid of
  every tracked cell in `(x, y, z)` order. Wire that into
  `init_gaussians_from_volume` when we get to P5.

## Where to look when things break

- `roundtrip_test.py` fails: dependencies. `pip install -e .` in a clean venv.
- Real CTC load throws `LZW requires imagecodecs`: `pip install imagecodecs`
  (should be automatic via `pyproject.toml`).
- Phantom PSNR plateaus below 30 dB: try halving `lr_position`, or check
  `--init-scale` (too large = blurry init that can't sharpen).
- Real-data PSNR plateaus very low (<15 dB): inspect the residual MIPs;
  capacity is probably going into background fluorescence rather than nuclei.
  That's an E2/E4 problem, not a v1 bug.
- `OSError: [Errno 9]` from `tifffile` on a network mount: known
  `flush()`-on-FUSE quirk; switch the file to `.npy`, the loader handles both.

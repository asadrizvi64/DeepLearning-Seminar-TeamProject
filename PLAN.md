# 3D Gaussian Splatting for Volumetric Microscopy — Project Plan

*CMS-Team Project · Chair of Machine Learning for Spatial Understanding, TU Dresden*

---

## 1. Context & objective

3D Gaussian Splatting (3DGS) represents a scene as a set of anisotropic 3D Gaussians optimized to fit data and rasterized in real time. Light-sheet microscopy volumes of *Tribolium* and *Drosophila* embryos are **sparse** — signal concentrated on nuclei, membranes, and labelled cells in mostly empty space — which is exactly the regime where an adaptive primitive representation should beat a dense voxel grid on compactness while remaining browser-viewable.

The project investigates **how well 3DGS represents sparse volumetric microscopy data**, and how to extend the formulation to time-resolved (3D+t) acquisitions so biologists can scrub through development interactively.

Key framing point: this is **not** novel-view synthesis. We already have the 3D ground truth (the volume itself), so this is a *density-fitting* problem, not inverse rendering from posed photographs. That distinction drives the central design decisions below.

> **Note (added after real-data bring-up):** the "sparse signal in mostly empty space" framing does not hold for stage-5 Drosophila blastoderm in Fluo-N3DL-DRO — the embryo is a *dense shell* of nuclei with substantial diffuse background. See README "Findings from the real data" and revisit RQ2 framing before P3.

---

## 2. Project goals (from the brief)

1. **Static volume fitting** — a PyTorch + `gsplat` density-fitting pipeline for a single volume; characterize reconstruction quality vs. number of Gaussians, file size, fit time, GPU memory.
2. **Export & visualization** — convert the fitted representation to `.ply` / `.splat` and inspect it in an existing browser viewer (antimatter15/splat, SuperSplat).
3. **Time-resolved extension** — adapt to 3D time series; compare *per-frame Gaussian sets with shared identity* vs. *fully 4D Gaussians with a temporal covariance dimension* on storage, smoothness, frame-to-frame consistency.
4. **Time-series renderer** — extend a browser viewer to play back the 4D representation for interactive scrubbing.

---

## 3. Core research questions

These are the questions the project should *answer*, not just the features it should build.

- **RQ1 — Integration bias.** Does the stock `gsplat` (render-and-compare) pipeline exhibit a systematic intensity error when fitting a density volume, as R²-Gaussian found for X-ray rasterization? If so, does a direct differentiable voxelizer remove it?
- **RQ2 — Compression vs. baselines.** Does 3DGS actually dominate the obvious alternatives (compressed dense voxels, sparse point clouds, implicit fields) on the fidelity–storage Pareto front, and *where* does it win (storage, render speed, editability)?
- **RQ3 — Temporal representation trade-off.** Among per-frame/shared-identity, native-4D, and deformation-based representations, which gives the best storage–smoothness–consistency balance for embryo development?
- **RQ4 — Mitosis failure modes.** How does each temporal representation fail at cell-division events (topology change), and can the failure be quantified?
- **RQ5 — Downstream-task equivalence.** Can biological analysis tasks (nucleus counting, segmentation, lineage tracking) be performed on the Gaussian representation as well as on the raw volume?

RQ1 and RQ4 are the two findings most worth building the project narrative around: both are concrete and likely novel on this data.

---

## 4. Phases & milestones

Owners and durations are placeholders — adjust to team size and semester schedule. Phase 4 runs in parallel with 3 and 5.

| Phase | Milestone | Key activities | Exit criterion | Owner | Effort |
|-------|-----------|----------------|----------------|-------|--------|
| **P0** | Environment & data ready | `gsplat` + PyTorch env; obtain one *Tribolium* volume; build slice/voxel loaders and a synthetic phantom for sanity checks | A volume loads, a phantom round-trips through the loader | _TBD_ | S |
| **P1** | Static pipeline v1 | Render-and-compare fit on one volume; intensity-weighted initialization; basic densification | A single volume fits and renders; PSNR logged | _TBD_ | M |
| **P2** | Supervision decided (RQ1) | Implement direct voxel-query path + differentiable voxelizer; run integration-bias study (E1) | Decision recorded: render-and-compare vs. voxel-query as primary supervision | _TBD_ | M |
| **P3** | Static characterization (RQ2) | Initialization ablation (E2), quality–compactness frontier (E3), densification re-tuning (E4), baseline panel (E5) | Pareto plot produced; static method frozen | _TBD_ | L |
| **P4** | Export & viewer | `.ply`/`.splat` export with round-trip check (E6); static viewer working; 4D container design | A fitted volume is viewable in a browser; round-trip fidelity verified | _TBD_ | M |
| **P5** | Temporal representations (RQ3) | Implement per-frame/shared-identity, native-4D, deformation-based; storage scaling (E9); interpolation (E10); comparison (E7) | All three variants run on a short time series; comparison table filled | _TBD_ | L |
| **P6** | Stress test & validation (RQ4, RQ5) | Mitosis stress test (E8); downstream-task equivalence on annotated data (E11) | Division-event failure quantified; downstream metrics reported | _TBD_ | M |
| **P7** | Write-up & 4D renderer polish | Final report; 4D playback in browser; reproducibility pass | Report submitted; renderer scrubs a 4D sequence at interactive rates | _TBD_ | M |

*Effort key: S ≈ small, M ≈ medium, L ≈ large.*

---

## 5. Experiment matrix

| ID | Research question | Setup / what varies | Primary metrics | Success / decision criterion | Goal |
|----|-------------------|---------------------|-----------------|------------------------------|------|
| **E1** | RQ1 | Render-and-compare vs. direct voxel-query supervision, same volume & budget | Density-MAE, intensity bias (signed error in overlap regions), PSNR | Identify whether render-and-compare has systematic bias; pick primary supervision | 1 |
| **E2** | — | Initialization: random vs. intensity-weighted vs. blob/nuclei detection | Final PSNR, fit time, #Gaussians to reach target PSNR | Best init strategy chosen; sensitivity quantified | 1 |
| **E3** | RQ2 | Gaussian budget sweep (e.g. 1k → 500k) | PSNR, SSIM, file size, fit time, peak GPU memory | Quality–compactness frontier traced; "knee" identified | 1 |
| **E4** | — | Densification heuristics: gradient threshold, opacity-reset schedule, split/clone rules | PSNR vs. #Gaussians, redundancy (Gaussians in empty space) | Microscopy-tuned densification config; reported as a result | 1 |
| **E5** | RQ2 | 3DGS vs. compressed voxels, sparse point cloud, implicit baseline | Fidelity–storage Pareto; render FPS | Show where (if) 3DGS dominates | 1 |
| **E6** | — | Export → re-import round trip; density-channel mapping into `.splat` fields | Round-trip PSNR/MAE vs. in-memory representation | Export verified lossless (or loss quantified) | 2 |
| **E7** | RQ3 | Per-frame/shared-identity vs. native-4D vs. deformation-based, same time series | Total storage, temporal smoothness, frame-to-frame consistency, per-frame PSNR | Comparison table; recommended representation for embryo data | 3 |
| **E8** | RQ4 | Time window containing cell-division events | Localized PSNR drop at division, identity-switch count, artifact characterization | Each method's mitosis failure mode quantified | 3 |
| **E9** | RQ3 | Storage vs. number of frames, per representation | Total parameters / file size as a function of `t` | Confirm/measure O(t·N) vs. sublinear scaling; find crossover frame count | 3 |
| **E10** | — | Render at non-integer timestamps (e.g. t = 2.5) | Interpolation PSNR vs. held-out frame | Establish which representations support temporal interpolation | 3, 4 |
| **E11** | RQ5 | Downstream task (counting / segmentation / tracking) on raw volume vs. Gaussian representation | Count error, segmentation Dice/IoU, tracking MOTA / track purity | Show biological structure is preserved, not just intensity | 1, 3 |

---

## 6. Metric definitions

Precise definitions so results are comparable across team members.

**Reconstruction fidelity**
- **PSNR** — peak signal-to-noise ratio between reconstruction and ground-truth volume. Specify and fix the evaluation domain: either per-2D-slice averaged over all slices, or directly on the 3D voxel grid. Use the same choice everywhere.
- **SSIM** — structural similarity; use a volumetric (3D-window) variant for the volume domain, or per-slice SSIM averaged. State which.
- **Density-MAE** — mean absolute error of the queried Gaussian density field vs. the ground-truth voxel grid. The honest metric for density fitting (independent of any 2D projection).
- **Intensity bias** — *signed* mean error, separated for overlapping vs. isolated regions. The diagnostic for the R²-Gaussian integration bias (E1).

**Structure preservation** (operationalizes the brief's vague "structure preservation")
- **Nucleus-count error** — |detected − ground-truth| nucleus count on the reconstruction.
- **Segmentation Dice / IoU** — overlap between segmentation of the reconstruction and of the original volume, using a fixed off-the-shelf segmenter.
- **Detection F1** — precision/recall of nucleus/cell detection on the reconstruction vs. original.

**Compactness & cost**
- **File size** — actual `.ply`/`.splat` size on disk (not just parameter count).
- **Compression ratio** — file size relative to the (compressed) voxel grid.
- **Fit time** — wall-clock to reach a fixed target PSNR (fairer than fixed iterations).
- **Peak GPU memory** — max allocated during fitting.
- **Render FPS** — playback frame rate in the browser viewer at a fixed resolution.

**Temporal metrics**
- **Temporal smoothness** — jitter of Gaussian parameters across time; e.g. mean second difference (acceleration) of positions/scales, or rendered-frame optical-flow magnitude where no motion is expected.
- **Frame-to-frame consistency** — PSNR of frame `t` rendered using the neighbouring frame's model (or warp-consistency error). Lower drift = more consistent.
- **Temporal interpolation PSNR** — render at a held-out intermediate timestamp, compare to the true frame.
- **Tracking quality** — for the shared-identity representation, MOTA and track purity against Cell Tracking Challenge annotations.

---

## 7. Baseline panel

A "3DGS fits microscopy" result proves little without comparison. Minimum baseline set for E5:

- **Compressed dense voxels** — e.g. `zfp` / `blosc` on the raw volume. The "do nothing clever" baseline.
- **Sparse representation** — thresholded point cloud or sparse voxel octree. Tests whether anisotropy/optimization actually buys anything over naive sparsity.
- **Implicit field** — a small NeRF/SIREN-style fit. The implicit counterpart 3DGS is meant to replace.

Deliverable: a single fidelity-vs-storage Pareto plot with 3DGS as one curve among these. If 3DGS does not dominate, that is still a publishable finding — characterize *where* it wins and loses.

---

## 8. Datasets

- **Primary** — *Tribolium* and *Drosophila* light-sheet volumes (time-resolved).
- **Validation with ground truth** — Cell Tracking Challenge data (has segmentation + lineage annotations → enables E11 and tracking metrics). **In hand:** Fluo-N3DL-DRO (developing Drosophila, Keller lab, 50 frames, 30s cadence, 0.406×0.406×2.03 µm voxels). See `volsplat/ctc.py` and `scripts/inspect_ctc.py`.
- **References / optional** — IDR light-sheet studies.
- **Synthetic phantom** — a hand-built sparse volume with known structure and known division events; essential for debugging E1 and E8 before touching real data. *DRO has zero division events in the TRA GT — confirmed via inspect_ctc.py — so E8 will need synthetic phantoms with controlled divisions regardless.*

---

## 9. Risks & open decisions

- **Supervision fork (decide in P2).** Render-and-compare reuses `gsplat` almost unchanged but may inherit the integration bias; direct voxel-query is more faithful but requires building a differentiable voxelizer. E1 decides this — do not over-invest in P1 before P2 resolves it.
- **Initialization.** No SfM points exist; poor initialization tends to dominate final quality in sparse scenes. Mitigated by E2.
- **Densification heuristics** are tuned for photographic scenes and will likely misbehave on sparse volumes — budget time for E4.
- **Anisotropic PSF.** Light-sheet data has a strongly anisotropic point spread function; a PSF-aware projection operator may be needed (cf. slice-based microscopy GS work). Flag as a possible extension if quality plateaus. *Mitigated for init via per-axis `init_scale`; PSF-aware projection is still an open extension.*
- **No standard 4D `.splat` container.** Goal 4 includes a real format-design decision (keyframes + deltas vs. streamed deformation field); trade-off is storage vs. scrub latency.
- **Topology change at mitosis.** Native-4D Gaussians cannot bifurcate along time without temporal densification; fixed-identity per-frame breaks when cell count changes. This is a known hard case — it is the experiment (E8), not a bug to avoid.
- **Sparsity assumption may not hold.** Stage-5 Drosophila blastoderm is a dense shell of nuclei rather than sparse blobs in empty space. The RQ2 compression argument needs re-examination before P3 — possibly reframe around structure preservation (RQ5) or the temporal angle (RQ3) as the lead narrative.

---

## 10. Deliverables

1. `gsplat`-based static density-fitting pipeline (PyTorch).
2. Differentiable voxelizer + integration-bias study (E1).
3. Quality–compactness frontier and baseline Pareto plot.
4. `.ply`/`.splat` exporter with verified round-trip fidelity.
5. Three temporal representations with a head-to-head comparison table.
6. Mitosis stress-test analysis.
7. Downstream-task equivalence report.
8. Browser-based 4D time-series renderer.
9. Final project report.

---

## Suggested first two weeks

P0 in full, then start P1: get one *Tribolium* volume fitting end-to-end with render-and-compare on the synthetic phantom first, then real data. The goal of the first sprint is a *pipeline*, not quality — quality work starts once E1 has decided the supervision strategy.

---

## 11. P2 results — E1 integration-bias study

**Status: complete. Decision: voxel-query supervision for P3 onward.**

### Setup

- Implementation: pure-torch orthographic alpha-blending rasterizer in `volsplat/rasterize.py` (front-to-back depth-sort + cumprod(1-α), splat opacity clamped at 0.999), and a projection-supervised training loop `train_static_projection` cycling through the three axis-aligned views.
- Diagnostic: `volsplat/bias.py` reports MAE, RMSE, signed mean error on a uniform voxel subsample, stratified by per-voxel Gaussian-overlap count (number of Gaussians contributing >10% of their peak at that voxel).
- Runner: `scripts/run_e1.py` trains both supervisors with matched init / Adam / budget on the same phantom, then runs the diagnostic on each.

### Single-Gaussian sanity check (closed form)

The R²-Gaussian bias has a closed-form prediction for an isolated Gaussian rendered along axis z: the true z-sum-projection equals the alpha-blended render times √(2π)·σ_z. We verified this numerically with one Gaussian (α=1, σ=2): predicted ratio 5.013, measured 5.018. The alpha-blend model trained to match the true sum-projection must therefore inflate its amplitude by ~√(2π)·σ_z, producing a density field that overshoots ground truth at every Gaussian's center.

### E1 on the phantom (24×40×40, 15 blobs σ=2, 400 Gaussians, 800 iters)

| supervisor | density PSNR | signed mean | at-blob ratio |
|---|---|---|---|
| voxel-query | +31.7 dB | +0.010 | 0.96× |
| alpha-blend | −7.1 dB | +0.599 | 14.8× |

Signed error stratified by Gaussian overlap count (alpha-blend):

| overlap | 0 | 1 | 2–3 | 4–7 | 8–15 | 16+ |
|---|---|---|---|---|---|---|
| signed mean | +0.006 | +0.062 | +0.310 | +1.30 | +4.96 | +12.46 |

The bias is positive (overshoot), monotonic, and grows roughly geometrically with overlap count — exactly the R²-Gaussian prediction. Voxel-query at the same overlap bins stays under +0.04.

### Interpretation

The closed-form 5.0× minimum is for isolated Gaussians along a single axis. The phantom has heavy blob overlap (typical of microscopy) and the supervisor cycles three axes, so the trained model is forced to inflate further to satisfy all three projections against an overlap-summed GT. The measured 14.8× at-blob ratio is the consistent extension of the 5.0× floor, not a contradiction of it.

The bias is not a transient: signed-mean ≈ MAE for alpha-blend means the model converges *toward* the inflated solution rather than oscillating around an unbiased one. Longer training would tighten projection PSNR but not the density-domain bias.

### Decision

Voxel-query (equivalently, the full-grid differentiable voxelizer) is the primary supervision for P3 onward. Render-and-compare via the stock gsplat-style operator is unsuitable for density supervision on volumetric microscopy: the recovered density field is systematically too hot, with error proportional to local Gaussian crowding. A bias-free *analytic* projection (closed-form integral of the Gaussian along the viewing axis) remains a possible secondary path if projection-domain supervision is ever needed; it is unused for P3.

### Artifacts

- `runs/e1_v1/e1_report.json` — first E1 run, σ=2.
- `runs/e1_sigma1/`, `runs/e1_sigma2/`, `runs/e1_sigma3/` — σ-scaling robustness sweep.

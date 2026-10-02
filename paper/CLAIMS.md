# Claim matrix (Study A)

Every claim the paper makes, the evidence behind it, and its status. "Rule" refers to
paper/DECISION_RULES.md. All numbers come from runs/fidelity/results_all.csv via
scripts/fidelity/reproduce.sh. Frozen v1.0 = git tag `study-a-v1.0` (4 frames, 2 embryos,
LoG + 2D-stitched Cellpose, 1,095 labelled nuclei).

Status key: **solid** = holds under its rule with intervals excluding zero; **qualified** =
holds in a narrowed or hedged form; **pending** = a committed test has not run yet.

| # | Claim (paper wording) | Evidence in v1.0 | Rule / verdict | Still to test | Status |
|---|---|---|---|---|---|
| C1 | JPEG-XL is nearly lossless for detection up to its maximum ratio (82–118×; worst case 92%, Cellpose) | 4 frames × 2 detectors | 1c: QUALIFY (worst 0.925 < 0.97) | A1: 8 more frames, 95%/90% crossings per detector | qualified |
| C2 | JPEG2000 degrades beyond ~100× and keeps 32–83% at ~230× | 4 frames × 2 detectors | descriptive | A1 crossings | solid (descriptive) |
| C3 | Luxar's own budget K* (~28–30×) keeps ≥97% under both detectors, where JPEG-XL is equally safe; fitting costs 17–46 min on an A100 | E1 t150, t194 only | 3: SAFE (LoG 101%, Cellpose 97–98%) | embryo-2 calibration (Capella 4323549/50) | pending (embryo 2) |
| C4 | At ~230× Luxar keeps far more nuclei than JPEG2000 (LoG 96–103% vs 48–83%; Cellpose 82–88% vs 32–74%) | 4 frames × 2 detectors; bootstrap over nuclei | 1b: NARROWED to ≳200× (reversed at 110–135× under Cellpose, E2); CI > 0 on 4/4 (LoG), 3/4 (Cellpose; E1 t150 unresolved) | A1 adds J2K at Luxar's bytes on 8 frames | qualified |
| C5 | Luxar falls below 90% (LoG) only at ≤2.3 splats per nucleus; 66–67% at one per nucleus | 4 frames + E1 t100 | 2: SUPPORTED (t100 never drops) | A1: interpolated 90% crossing ≤3.5 splats/nucleus on ≥6 of 8 frames (v1.0 frames: 2.4–3.1) | pending (A1) |
| C6 | PSNR does not reveal the cliff: Luxar PSNR flat within 19.1–22.6 dB; PSNR orders 7–24% of pairs the wrong way for faint nuclei kept | 4 frames | descriptive | A1: PSNR range < 3 dB while nuclei kept moves ≥ 0.15, on ≥6 of 8 frames | pending (A1) |
| C7 | Splat reconstructions cost a learned segmenter 6–15 points of recall vs JPEG-XL at 45–94×, on every frame | 4 frames, Cellpose 2D stitched; bootstrap | 1a: REPLICATES; CI > 0 on 4/4 (E2 t180 narrowly) | A1: same gap (JPEG-XL ≤120× vs Luxar K=16000) ≥5 points on ≥6 of 8 frames; A2: Cellpose 3D mode (is it the stitching?) | qualified (moderate vs Cellpose's own 14-point wobble) |
| C8 | The blob detector is unaffected at 45–94× (98–104%) | 4 frames | descriptive | A2: classical 3D watershed — does a non-learned *segmenter* show the gap? | solid for LoG; generality pending |
| C9 | The blob detector needs ~3–5 splats per nucleus; Cellpose ~100 (the regime of K*) | cliff figure, 5 frames | descriptive | A1 adds 8 frames; Cellpose 90% crossing at 41–145 splats/nucleus in v1.0 | solid (descriptive) |
| C10 | Survival (label-free) tracks nuclei kept across Luxar fits (Spearman 0.68, n = 30); flags the cliff, not the segmenter gap | 5 frames | descriptive | A1 adds ~48 fits | solid (descriptive) |
| C11 | Mechanism: splats render smooth blobs on a black background, an image distribution the segmenter was not trained on | Fig. 1 only (qualitative) | none | A2 watershed and Cellpose 3D constrain it; fine-tuning is future work | hypothesis — keep worded as such |
| L1 | Limitation: run-to-run variance of Luxar fits | none yet | — | repeats of E1 t194 (Capella 4323551/52) | pending |

## Reading the gaps

- The paper's two headline claims are C4 (splats beat JPEG2000 at ≳200×) and C7 (learned
  segmenters pay a recall cost). Both are qualified, not solid. A1 decides whether they
  generalise across frames; A2 decides whether C7 says something about Gaussian
  representations or only about 2D-stitched Cellpose.
- C3 is the claim most exposed to reviewers ("you only checked one embryo"); the embryo-2
  calibration is already submitted.
- Nothing in the matrix depends on PSNR-optimised tuning, K-sweeps beyond the committed
  grid, or new compressors — those stay out of scope.

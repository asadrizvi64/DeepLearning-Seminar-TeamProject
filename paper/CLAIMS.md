# Claim matrix (Study A, final data 2026-10-06)

Every claim the paper makes, the evidence behind it, and its status. "Rule" refers to
paper/DECISION_RULES.md. All numbers come from the per-frame result files via
scripts/fidelity/reproduce.sh; worded claims are asserted in scripts/fidelity/paper_numbers.py
and stop the build if the data stop supporting them (9 assertions, all passing).

**Data.** 12 C. elegans frames from two embryos (2,958 labelled nuclei): 4 chosen first + 8
added under pre-registered rules (section 4); E1 t100 reported separately. 80 Luxar fits,
each paired with JPEG2000 at identical bytes; four detectors (LoG, 3D watershed, Cellpose
2D-stitched, Cellpose 3D mode), all on all 12 frames.

**History.** v1.0 (tag `study-a-v1.0`) used a broken JPEG2000 baseline; its headline (splats
beat JPEG2000 beyond ~200x) was an artefact. A second draft blamed a "learned-model" penalty;
Cellpose 3D mode refuted that (the penalty is from slice-wise processing).

| # | Claim (paper wording) | Evidence | Rule / verdict | Status |
|---|---|---|---|---|
| C1 | Frame level (12 frames, Wilcoxon, Holm over 8 tests), below ~300x: 3D Cellpose favours splats (+3.5 points, 10/12 frames, p=0.010); watershed (−4.0) and 2D Cellpose (−9.4) favour JPEG2000 on 12/12 (p=0.004); LoG NOT consistent (+1.3, 8/12, p=0.11; embryo 2 only) | frame_level.csv; 86/78 matched pairs | 5 + frame-level tests | solid (LoG claim withdrawn) |
| C2 | Watershed and 2D Cellpose never keep significantly fewer nuclei with JPEG2000 | JPEG2000 better in 63/86 pairs each, Luxar 0 | 5 | solid |
| C3 | Beyond ~300x every detector favours JPEG2000; splats lose nuclei abruptly | far-ratio means −17 to −20 points | 5 | solid |
| C4 | PSNR favours JPEG2000 on every frame (12/12, both regimes) and in every pair (80/80): right at extreme ratios and for slice-wise segmentation, wrong for 3D Cellpose at moderate ratios — insufficient, not useless | frame_level.csv | descriptive | solid (headline) |
| C5 | Budget cliff: LoG crosses 90% at 1.1–3.1 splats per nucleus on all 12 frames | a1_crossings | A1 cliff HOLDS 8/8 (+ 4 first frames) | solid |
| C6 | PSNR misses the cliff — narrowed: PSNR flat (≤2.6 dB span) but the pre-registered test held on only 4/8 added frames (cliff below K=1000 on small frames) | a1 rule | A1 PSNR DOES NOT HOLD (4/8) | qualified, reported as failed |
| C7 | 2D Cellpose penalty vs JPEG-XL at 45–94x | 4 first frames (1a) + 8 added | 1a REPLICATES; A1 HOLDS 8/8 | solid (for 2D-stitched) |
| C8 | The penalty is absent in Cellpose's 3D mode → slice-wise processing, not the learned model | 3D gap ≥5 on 0/12 frames; Luxar 0–10 points better | A2 stitching: penalty absent | solid |
| C9 | Watershed sits in between (gap < 5 points on 11/12) | a1 rule | A2 watershed | solid |
| C10 | Luxar's K* = 64,000 on all 4 calibrated frames (both embryos), ~25–30x; keeps 89–105% depending on detector; JPEG2000 at the same bytes 99–102%; fitting 17–46 min (A100) | 4 frames, full-data fits | 3 SAFE (+ embryo-2 extension) | solid |
| C11 | JPEG-XL nearly lossless to its maximum distance (82–120x; worst 91%) | 12 frames × 4 detectors | 1c QUALIFY | qualified |
| C12 | Luxar fits reproducible: ≤ 2.1 / 3.5 / 3.7 points across 3 fits × 3 budgets | E1 t194 repeats | descriptive | solid |
| C13 | Survival (label-free) tracks nuclei kept (ρ = 0.66, n = 80) | 12 frames | descriptive | solid |
| C14 | "Smooth splats denoise for volumetric detectors" | kept > 1 under LoG / 3D Cellpose | none | interpretation — worded as such |
| C15 | Pitfall: a common JPEG2000 binding encodes (z, y, x) along the wrong axes | regression test | — | solid |

## Dropped or void

- v1.0 "splats beat JPEG2000 at ≳200x" (rule 1b): void; on the corrected codec it does not hold.
- "Out-of-distribution for a learned segmenter" mechanism: refuted by Cellpose 3D mode.
- Downsampling and ZFP: naive configurations; removed from comparisons.

## Fixed 2026-10-06

- One corrupted Cellpose cache (E2 t130 K16000, all-zero centroids from the run interrupted on 2026-10-03) had scored a Luxar volume as 0% kept and inflated the 2D-Cellpose deficit (−13.0 → −9.4 after the fix). Caches are now written atomically and checked (`check_caches.py`) before every analysis.

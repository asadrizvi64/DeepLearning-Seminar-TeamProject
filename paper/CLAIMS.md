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
| C1 | Below ~300x the winner depends on the detector: LoG (+1.2) and 3D Cellpose (+3.7) favour splats, watershed (−4.2) and 2D Cellpose (−11.7) favour JPEG2000 (points, Luxar − JPEG2000) | 86 matched pairs (LoG/WS/CP2D), 78 (CP3D), 13 frames | 5 (paired CIs) | solid |
| C2 | Watershed and 2D Cellpose never keep significantly fewer nuclei with JPEG2000 | JPEG2000 better in 63/86 pairs each, Luxar 0 | 5 | solid |
| C3 | Beyond ~300x every detector favours JPEG2000; splats lose nuclei abruptly | far-ratio means −17 to −20 points | 5 | solid |
| C4 | PSNR favours JPEG2000 in every matched pair (80/80, by 1.0–6.8 dB) | run CSVs | descriptive | solid |
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

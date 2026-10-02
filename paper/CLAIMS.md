# Claim matrix (Study A)

Every claim the paper makes, the evidence behind it, and its status. "Rule" refers to
paper/DECISION_RULES.md. All numbers come from the per-frame result files via
scripts/fidelity/reproduce.sh; worded claims ("never", "every size") are asserted in
scripts/fidelity/paper_numbers.py and stop the build if the data stop supporting them.

**History.** v1.0 (tag `study-a-v1.0`) used a broken JPEG2000 baseline (wavelet never ran
along x). Its headline -- splats beat JPEG2000 beyond ~200x -- was an artefact and is gone.
This matrix describes the corrected study (4 frames, 2 embryos, 1,095 labelled nuclei;
LoG, 3D watershed, Cellpose 2D-stitched; JPEG2000 at every Luxar fit's exact bytes).

Status key: **solid** = holds under its rule with intervals excluding zero; **qualified** =
holds in a narrowed or hedged form; **pending** = a committed test has not run yet.

| # | Claim (paper wording) | Evidence | Rule / verdict | Still to test | Status |
|---|---|---|---|---|---|
| C1 | JPEG-XL is nearly lossless for detection up to its maximum distance (82–118×; worst 92%) | 4 frames × 3 detectors | 1c: QUALIFY (worst 0.925) | A1: 8 more frames | qualified |
| C2 | JPEG2000 keeps 100–102% (LoG), 97–100% (watershed), 84–101% (Cellpose) at 321–530× | 4 frames × 3 detectors, corrected codec | descriptive | A1 crossings | solid (descriptive) |
| C3 | At identical bytes, JPEG2000 keeps at least as many nuclei as Luxar for both segmenters: significantly more in 23/30 (watershed) and 19/30 (Cellpose) pairs, never fewer | matched_pairs.csv, paired bootstrap over nuclei | 5: holds for segmenters | A1 frames | solid |
| C4 | Under LoG, Luxar keeps 3–4 points more at 46–227× on embryo 2, and 8–11 points more on E1 t100 (larger nuclei); JPEG2000 leads beyond ~300× | matched_pairs.csv | 5: named exceptions (11 Luxar-better LoG pairs) | A1 frames | solid (as an exception) |
| C5 | JPEG2000 has higher PSNR at every matched size (30/30) | run CSVs | descriptive | A1 | solid |
| C6 | Luxar's own K* (64,000, ~28–30×) keeps 96–101% under all detectors; JPEG2000 at the same bytes 99–102%; fitting 17–46 min (A100) | E1 t150, t194; E2 calibration also gives K*=64,000 | 3: SAFE | E2 full-data K=64,000 fits (to submit) | qualified (scored on E1 only) |
| C7 | Budget cliff: Luxar < 90% (LoG) only at ≤ 2.3 splats/nucleus; 66–67% at one; PSNR flat within 19.1–22.6 dB | 4 frames + E1 t100 | 2: SUPPORTED | A1: 90% crossing ≤ 3.5 splats/nucleus on ≥ 6/8 frames | pending (A1) |
| C8 | Cellpose loses 6–15 points vs JPEG-XL at 45–94× on Luxar (CI > 0 on 4/4) | 4 frames | 1a: REPLICATES | A1 (2D-stitched); A2 Cellpose 3D mode (cluster) | solid; 3D-mode pending |
| C9 | The classical watershed loses only 2–4 points vs JPEG-XL there (CI > 0 on 4/4, < 5 points on 3/4) → the large penalty is specific to the learned segmenter | 4 frames | A2 watershed rule: "learned segmenters are the sensitive ones" (provisional, 4 of 12 frames) | A1 frames complete the rule | qualified (provisional) |
| C10 | Luxar fits are reproducible: 3 fits × 3 budgets differ by ≤ 2.1 / 3.5 / 3.7 points | E1 t194 repeats (Capella 4323551/52) | descriptive | — | solid |
| C11 | Survival (label-free) tracks nuclei kept across Luxar fits (ρ = 0.68, n = 30) | 4 frames | descriptive | A1 | solid (descriptive) |
| C12 | Mechanism: smooth blobs on a black background are out of distribution for a learned segmenter | Fig. 1, C8 vs C9 | none | fine-tuning is future work | hypothesis — worded as such |
| C13 | Pitfall: a common JPEG2000 binding encodes (z, y, x) along the wrong axes; it reversed our high-ratio conclusion | regression test (SIZ marker) | — | — | solid |

## Dropped or void

- v1.0 "splats beat JPEG2000 at ≳200×" (rule 1b): void, broken baseline; on the corrected
  codec rule 1b does not hold (Cellpose: JPEG2000 better in every embryo-2 pair).
- Downsampling and ZFP: naive configurations (8-bit data stored as 16-bit; isotropic factor;
  fixed-rate ZFP); removed from the comparisons rather than tuned after the fact.
- "PSNR orders X% of pairs wrong": no longer used in the text (now 3–55%, dominated by the
  blob detector preferring Luxar at lower PSNR).

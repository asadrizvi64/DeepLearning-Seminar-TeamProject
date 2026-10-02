# Decision rules, fixed before the pending results

Committed on 2026-09-30, while Capella jobs 4320426-4320430 were still running and before
any of their outputs had been seen. The paper's claims follow these rules; they are not
adjusted after the fact.

Measured with the same pipeline as the embryo-1 results (`rate_detectability.py`,
`render_luxar.py`, `cellpose_score.py`, `log_rescore.py`), match radius 0.6 D.

## 1. Replication on embryo 2 (Fluo-N3DH-CE sequence 02, frames 150 and 180)

- **Detector dependence.** Replicates if, with Cellpose, Luxar keeps at least 5 percentage
  points fewer nuclei than JPEG-XL at 55-94x on both frames. Otherwise the claim is
  downgraded from a headline to "observed in one embryo".
- **High-ratio advantage.** Replicates if, beyond 120x, Luxar keeps more nuclei than JPEG2000
  under both detectors on both frames. Otherwise it is removed.
- **JPEG-XL safe to its maximum.** Replicates if JPEG-XL keeps at least 97% of nuclei under
  both detectors at every size it can reach. Otherwise it is qualified.

## 2. Cliff scaling (embryo 1, frame 100, 93 labelled nuclei)

- Supported if nuclei kept falls below 90% (LoG) only at fewer than ~5 splats per labelled
  nucleus, as on frames 150 and 194. Otherwise the "splats per nucleus" rule is removed and
  the cliff is reported per frame only.

## 3. Luxar's own budget (self-calibrated K*, embryo 1, frames 150 and 194)

- If K* keeps at least 95% of nuclei under the blob detector on both frames, the paper states
  that Luxar's own calibration is safe on this data for blob detection, and the "PSNR
  cannot see the cliff" claim is limited to fixed or hand-chosen budgets.
- If K* keeps less than 95% on either frame, the paper reports that following Luxar's own
  recommendation loses nuclei.
- In both cases, K* is also scored with Cellpose and reported.

## 4. Robustness expansion (Study A, stages A1-A2) -- added 2026-10-02, before any A1/A2 data

New frames: embryo 1 t110, t130, t170, t185; embryo 2 t110, t130, t165, t185 (all with
roughly 100+ labelled nuclei; earlier stages are excluded because their nuclei are larger
than the fixed D). Luxar budgets K = 1000, 2000, 4000, 16000; codecs at 50-800 KiB.
A claim "holds" if it holds on at least 6 of the 8 new frames (75%).

- **A1 cliff.** For each frame, the smallest splats-per-nucleus at which LoG keeps >= 90%.
  The paper's "cliff at <= ~2-3 splats per nucleus" holds if that crossing is <= 3.5 splats
  per nucleus. Otherwise the paper reports the per-frame distribution and drops the number.
- **A1 PSNR.** Holds if Luxar's PSNR varies by < 3 dB across the four budgets while LoG
  nuclei kept varies by >= 0.15.
- **A1 detector dependence.** Holds if, with 2D-stitched Cellpose, JPEG-XL (at its largest
  reachable ratio <= 120x) keeps >= 5 points more nuclei than Luxar at K = 16000.
- **A2 stitching.** With Cellpose in 3D mode (do_3D), the same gap >= 5 points on >= 75% of
  frames rules out the 2D stitching as the cause. Otherwise the claim is restricted to
  2D-stitched segmentation.
- **A2 learned vs classical.** With a non-learned 3D watershed detector, a gap < 5 points on
  >= 75% of frames supports "learned segmenters are the sensitive ones"; a gap >= 5 points
  means the penalty is not specific to learned models, and the paper says so.

**Amendment (2026-10-02, still before any A1 data).** K=500 and K=1500 are added to the Luxar
budgets, because on the ~100-nucleus frames K=1000 already gives about 3 splats per nucleus
(E1 t100: 288 splats for 93 nuclei), so the 90% crossing would not be bracketed. The
crossing is interpolated linearly in log(splats per nucleus) between the two budgets that
bracket 90%. If even the smallest budget keeps >= 90%, the crossing is reported as an upper
bound: it counts as holding if that bound is <= 3.5, and as unresolved otherwise. Unresolved
frames count against the 6-of-8 threshold. The A1 PSNR rule still uses K=1000-16000 only.
Codecs: JPEG2000 and JPEG-XL only (the two the paper's claims rest on), at 50, 100, 200, 400
and 800 KiB, plus at each Luxar fit's byte size once those are known. The 95% and 90%
crossings are reported for every method and detector, as descriptive results.

## Baseline correction (2026-10-02, before any corrected detection result was seen)

The JPEG2000 baseline of v1.0 (tag `study-a-v1.0`) was mis-configured: the (z, y, x)
volume was passed to OpenJPEG as a z-y image with one component per x column, so the
wavelet never ran along x (E2 t150 at 50 KiB: 12.5 dB instead of 23.1 dB). It is replaced
by a 2D wavelet per xy slice, with all slices as components of one codestream, the 2D
counterpart of the per-slice JPEG-XL. Every rule that involves JPEG2000 (1b, and the
JPEG2000 crossings in section 4) is re-evaluated unchanged on the corrected baseline. JPEG2000
is also encoded at the exact byte size of every Luxar fit. The v1.0 verdicts that involved
JPEG2000 are void, not "replicated" or "failed". Only the PSNR of the corrected codec on
one frame was seen before this note was written.

**Baseline fairness audit (same day).** Downsampling stored the 8-bit data as uint16 and
reduced z by the same factor as xy although z is already 11x coarser; ZFP ran in fixed-rate
mode on float32 copies of 8-bit data. Both are naive configurations, so they are dropped from
the paper's comparisons rather than tuned after the fact; their rows remain in the results
table. JPEG-XL's ceiling (about 82-118x) is the encoder's maximum distance (25) without its
optional resampling, so the paper says "at its maximum distance", not "cannot exceed".

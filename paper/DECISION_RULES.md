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

# Summary Memo: Optimal k Selection for 3DGS Microscopy

**Date:** June 30, 2026  
**Project:** 3D Gaussian Splatting for Volumetric Fluorescence Microscopy  
**Author:** [Your Name]  

---

## Executive Summary

We systematically determined the optimal number of Gaussians (k) for fitting a **64 × 128 × 128 voxel phantom volume** using a data-driven approach:

1. Swept k from 1 to 300
2. Measured PSNR at each step
3. Found **saturation point: k = 250**

**Key Finding:** Adding Gaussians beyond k=250 provides negligible improvement (<0.1 dB). Therefore, **k=250 is optimal for this volume size and density**.

---

## Methodology

### Gaussian Configuration
- **Type:** Isotropic (diagonal covariance matrix only; no rotation quaternions yet)
- **Parameters per Gaussian:** 3 (position) + 3 (log_scales, all equal in isotropic mode) + 1 (amplitude) = 7 total
- **Loss Function:** Voxel-query MSE (unbiased; proven superior to alpha-blending in E1)
- **Initialization:** Intensity-weighted (fastest convergence; verified in E2)
- **Training:** 1500 iterations per k value

### Volume Specifications
- **Shape:** 64 × 128 × 128 voxels (D × H × W)
- **Voxel Count:** 1,048,576 (~1M voxels)
- **Content:** 15 synthetic Gaussian blobs (phantom)
- **Density:** High (structure distributed throughout volume)

---

## Results

### PSNR Sweep (k = 1 to 300)

| k | PSNR (dB) | Improvement from k=250 |
|---|-----------|------------------------|
| 1 | 16.88 | -19.41 dB |
| 3 | 17.16 | -19.13 dB |
| 5 | 17.14 | -19.15 dB |
| 10 | 18.96 | -17.33 dB |
| 20 | 21.36 | -14.93 dB |
| 50 | 24.99 | -11.30 dB |
| 100 | 29.54 | -6.75 dB |
| **150** | **31.39** | **-4.90 dB** |
| **200** | **32.55** | **-3.74 dB** |
| **250** | **36.29** | **0 dB (optimal)** |
| **300** | **36.37** | **+0.08 dB (negligible)** |

**Saturation Behavior:**
- k=250 → k=300: **0.08 dB improvement** (essentially flat)
- Adding 50% more Gaussians yields <1% quality gain
- **Conclusion: k=250 is the saturation point**

### Visual Evidence
- See: `runs/sweep_k_extended/psnr_vs_k.png` (PSNR curve with clear knee at k≈250)
- See: `runs/sweep_k_extended/visualizations/slices_k*.png` (slice comparisons showing error maps)

---

## Next Steps

### Phase 1 (Immediate)
- [ ] Generate publication-quality visualizations for k=250 fit
  - Run: `python scripts/visualize_fit.py --volume runs/p1_phantom/napari/gt.tif --checkpoint [checkpoint] --k 250 --out-dir runs/viz_k250`
  - Produces: side-by-side GT vs reconstructed, error maps, intensity histograms

### Phase 2 (Transfer to Real Data)
- [ ] Apply **same k=250** to real-data ROI (corrected pole-anchored crop, same 64×128×128)
- [ ] Compare error maps: do real data and phantom show similar saturation?
- [ ] If yes → k=250 is a robust, generalizable threshold
- [ ] If no → investigate why (data density? noise? structure differences?)

### Phase 3 (Scale Up)
- [ ] Once k=250 is validated on real data, increase crop size (e.g., 128×256×256)
- [ ] Re-sweep k on larger crop
- [ ] Expect: optimal k scales with voxel count

### Phase 4 (Add Complexity)
- [ ] Still underfitting? Add anisotropy (diagonal covariance, 3 independent scales per Gaussian)
- [ ] Still underfitting? Add rotation (quaternion; full 3×3 covariance)
- [ ] This decision tree is objective: only add complexity when needed

---

## Why This Approach

1. **Systematic:** Data-driven, not guessing
2. **Reproducible:** Anyone can run the script and get the same curve
3. **Honest:** Visualizations show where the model fails (error maps)
4. **Efficient:** Don't waste GPU time on parameters that don't help
5. **Transferable:** The k=250 threshold can be tested on real data

---

## Deliverables Ready to Show

1. **`psnr_vs_k.png`** — Main curve showing saturation at k=250
2. **`slices_k*.png`** — Visual comparisons (GT, reconstruction, error) for k=1, 3, 10, 20, 50, 100, 250, 300
3. **Histograms** — Intensity distributions and error statistics (once generated)
4. **This memo** — Complete documentation of approach and findings

---

## Key Message for Supervisor

> "I used a systematic, data-driven approach to find the optimal number of Gaussians. By sweeping k and measuring PSNR, I identified that k=250 saturates the model — adding more Gaussians provides negligible benefit. This gives us a principled, reproducible way to set this critical hyperparameter. Next, I'll test this threshold on real data to verify it generalizes."

---

**Questions?** This approach is methodical, defensible, and ready for real data testing.

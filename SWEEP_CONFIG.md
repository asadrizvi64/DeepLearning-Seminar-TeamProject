# k-Sweep Experiment Configuration & Results

## Phase 1: Find Optimal k (Synthetic Phantom)

### Crop Parameters (REMEMBER THIS)
- **Volume shape:** 64 × 128 × 128 voxels (D × H × W)
- **Source:** `runs/p1_phantom/napari/gt.tif` (phantom ground truth)
- **Voxel density:** Dense (15 blobs in 1M voxels)

### Gaussian Configuration
- **Type:** Isotropic ONLY (no quaternion rotation, no anisotropy yet)
  - Each Gaussian has: position (3) + log_scales (3, all must be equal in isotropic mode) + amplitude (1)
  - Total: 7 parameters per Gaussian
- **Initialization:** Intensity-weighted (from E2)
- **Loss:** Voxel-query MSE (unbiased, from E1)
- **Iterations:** 1500 per k value

### Experiments Done

#### Initial sweep (k=1 to 100):
| k | PSNR (dB) | Status |
|---|-----------|--------|
| 1 | 16.88 | Done |
| 2 | 17.10 | Done |
| 3 | 17.16 | Done |
| 5 | 17.14 | Done |
| 10 | 18.96 | Done |
| 20 | 21.36 | Done |
| 50 | 24.99 | Done |
| 100 | 29.54 | Done |

**Status:** Curve still climbing at k=100. Need to extend sweep.

#### Extended sweep (k=100 to 300):
To run:
```bash
python scripts/sweep_k_visual.py \
    --volume runs/p1_phantom/napari/gt.tif \
    --out-dir runs/sweep_k_extended \
    --iters 1500 \
    --k-values 100 150 200 250 300
```

Results: (pending)

---

## Phase 2: Apply to Real Data (Once k_opt is found)

### Real-Data Setup (to match synthetic)
- **Crop size:** Match the 64 × 128 × 128 structure if possible
  - For Fluo-N3DL-DRO ROI: aim for similar voxel count (~1M voxels)
  - Adjust if needed (e.g., 64 × 128 × 128 or 96 × 256 × 256)
- **Gaussian configuration:** SAME as phantom
  - Isotropic only
  - k = k_opt (to be determined)
  - 1500 iterations
- **Source:** Real embryo data with corrected ROI (pole-anchored, not interior)

### Real Data Experiments (to be done):
- [ ] Test k_opt on real-data ROI
- [ ] Compare PSNR curve to synthetic
- [ ] Visual inspection: do error maps look similar?
- [ ] If yes → confirm k_opt is robust across synthetic and real data

---

## Decision Tree: When to Add Complexity

```
Current state: Isotropic Gaussians, fixed k
↓
Once k_opt is found and verified on real data:
├─ Does the fit look good? (PSNR > threshold, visual inspection OK)
│  └─ YES → Move to next step
│     ├─ Still underfitting? Add anisotropy (diagonal covariance)
│     ├─ Still underfitting? Add rotation (quaternions)
│     └─ Fitting well? Done
│
└─ NO → Increase iterations or revisit loss function
```

---

## Important Notes

1. **Isotropic rationale:** Start simple, add complexity only when needed. Isotropic is stable and easier to debug.
2. **Crop size consistency:** The 64 × 128 × 128 must be remembered when we move to real data. Don't change crop size mid-analysis.
3. **k_opt is volume-density-dependent:** The optimal k depends on how much structure is in the volume. A sparser volume needs fewer Gaussians; a denser volume needs more.
4. **Transfer learning hypothesis:** If k_opt on synthetic is close to k_opt on real data, we've found a principled way to set k without guessing.

#!/bin/bash
# Extended sweep to find optimal k for the 64x128x128 phantom
# Crop size: 64x128x128 voxels
# Source: runs/p1_phantom/napari/gt.tif

echo "=== Extended k-sweep to find saturation point ==="
echo "Crop: 64x128x128 voxels"
echo "Sweeping k: [1, 2, 3, 5, 10, 20, 50, 100, 150, 200, 250, 300]"
echo ""

python scripts/sweep_k_visual.py \
    --volume runs/p1_phantom/napari/gt.tif \
    --out-dir runs/sweep_k_extended \
    --iters 1500 \
    --k-values 1 2 3 5 10 20 50 100 150 200 250 300

echo ""
echo "=== RESULTS ==="
echo "PSNR vs k curve: runs/sweep_k_extended/psnr_vs_k.png"
echo "Slice visualizations: runs/sweep_k_extended/visualizations/"
echo "Raw data: runs/sweep_k_extended/sweep_results.json"
echo ""
echo "=== CROP PARAMETERS TO REMEMBER ==="
echo "Volume shape: 64 x 128 x 128 (D x H x W)"
echo "Source: phantom from runs/p1_phantom/napari/gt.tif"
echo "Will use ISOTROPIC Gaussians (diagonal covariance only, no quaternion rotation)"
echo "Once optimal k is found, apply the SAME k to real data with similar crop size"

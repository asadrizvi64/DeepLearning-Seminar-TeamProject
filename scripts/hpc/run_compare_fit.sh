#!/bin/bash
# Run compare_fit.py with SSIM on the GPU fit checkpoint.
# Run this on the LOGIN NODE (no GPU needed - it's CPU inference + metrics).
#
# MUST load modules before activating venv (fixes libpython3.11.so error):
#   bash scripts/hpc/run_compare_fit.sh

set -eo pipefail
export LD_PRELOAD="${LD_PRELOAD:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
module purge
module load release/24.04 GCCcore/12.3.0 Python/3.11.3
set -u
source "${VENV_DIR:-$HOME/volsplat-venv}/bin/activate"

cd /projects/p_scads_mlhpc/DeepLearning-Seminar-TeamProject
git pull

WS="$(ws_find volsplat)"
if [ -z "$WS" ]; then echo "Workspace 'volsplat' not found." >&2; exit 1; fi

OUT_DIR="$WS/runs/ctc_t000_gpu/compare"
mkdir -p "$OUT_DIR"

echo "Running compare_fit on GPU checkpoint (40k Gaussians, 64x256x256 ROI)..."
python scripts/compare_fit.py \
    --volume "$WS/runs/ctc_t000_gpu/roi.npy" \
    --checkpoint "$WS/runs/ctc_t000_gpu/final.pt" \
    --out-dir "$OUT_DIR"

echo ""
echo "MIP comparison images: $OUT_DIR/compare_mip.png"
echo "Slice comparison:       $OUT_DIR/compare_slice.png"
echo ""
echo "Download to view:"
echo "  scp syas272h@login1.alpha.hpc.tu-dresden.de:$OUT_DIR/compare_mip.png ."

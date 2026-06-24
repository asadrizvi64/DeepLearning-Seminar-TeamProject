#!/bin/bash
# Static fit on the CORRECTED ROI (z anchored at blastoderm pole, not interior centre).
# Runs train_ctc.py then compare_fit.py so PSNR + SSIM are both produced.
#
# Submit from the repo root:
#   sbatch scripts/hpc/run_ctc_fixed.sbatch

#SBATCH --job-name=volsplat-ctc-fixed
#SBATCH --partition=alpha
#SBATCH --account=p_scads_mlhpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/%x_%j.out

set -eo pipefail
export LD_PRELOAD="${LD_PRELOAD:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

module purge
module load release/24.04 GCCcore/12.3.0 Python/3.11.3
set -u
source "${VENV_DIR:-$HOME/volsplat-venv}/bin/activate"

WS="$(ws_find volsplat)"
if [ -z "$WS" ]; then echo "Workspace 'volsplat' not found." >&2; exit 1; fi

REPO=/projects/p_scads_mlhpc/DeepLearning-Seminar-TeamProject
trap 'cp "/tmp/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$WS/logs/" 2>/dev/null || true' EXIT
mkdir -p "$WS/logs" "$WS/runs/ctc_t000_fixed/compare"

echo "=== Static fit: corrected ROI (pole-anchored z) ==="
python "$REPO/scripts/train_ctc.py" \
    --root "$WS/Fluo-N3DL-DRO" \
    --sequence 01 --frame 0 --crop \
    --target-shape 64 256 256 \
    --num-gaussians 40000 --iterations 5000 \
    --out-dir "$WS/runs/ctc_t000_fixed"

echo "=== compare_fit: PSNR + SSIM ==="
python "$REPO/scripts/compare_fit.py" \
    --volume "$WS/runs/ctc_t000_fixed/roi.npy" \
    --checkpoint "$WS/runs/ctc_t000_fixed/final.pt" \
    --out-dir "$WS/runs/ctc_t000_fixed/compare"

echo "Done. Download:"
echo "  scp syas272h@login1.alpha.hpc.tu-dresden.de:$WS/runs/ctc_t000_fixed/compare/compare_mip.png ."

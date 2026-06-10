#!/bin/bash
# Converged single-frame fit of a real Fluo-N3DL-DRO volume on one A100.
# This is the GPU version of the CPU-limited run we did locally (which capped at
# ~26 dB / 4000 Gaussians / 400 iters). On an A100 we can afford a real budget.
#
# Submit from the repo root on a login node:
#   sbatch scripts/hpc/fit_ctc_frame.sbatch
# Watch it:
#   squeue --me        # queue status
#   tail -f logs/volsplat-ctc_*.out
#
# Fill in the two PLACEHOLDERs below before submitting.

#SBATCH --job-name=volsplat-ctc
#SBATCH --partition=alpha
#SBATCH --account=p_scads_mlhpc            # TU Dresden project account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000                  # 8 cpus * 8 GB = 64 GB host RAM
#SBATCH --time=02:00:00                     # generous; the fit itself is minutes
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

# ---- environment ------------------------------------------------------------
VENV_DIR="${VENV_DIR:-$HOME/volsplat-venv}"     # must match setup_env.sh
module purge
module load release/24.04 GCCcore/12.3.0 Python/3.11.3
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---- data location on the cluster -------------------------------------------
# Copy the dataset to a workspace first (NOT $HOME). The DRO 01 sequence is ~GBs.
# Example transfer from your machine via the export/datamover nodes (see README).
DRO_ROOT="${DRO_ROOT:-PLACEHOLDER_PATH/Fluo-N3DL-DRO}"   # <-- holds 01/ and 01_GT/

# ---- the fit ----------------------------------------------------------------
# GPU-appropriate config: bigger ROI, far more Gaussians, enough iters to converge.
# Anisotropic init (xy=2.0, z=0.4 voxels) is on by default for DRO's 5x z-anisotropy.
srun python scripts/train_ctc.py \
    --root "$DRO_ROOT" \
    --sequence 01 \
    --frame 0 \
    --crop --target-shape 64 256 256 \
    --num-gaussians 40000 \
    --iterations 5000 \
    --batch-size 8192 \
    --out-dir runs/ctc_t000_gpu

echo "Done. Outputs in runs/ctc_t000_gpu/ (final.pt, history.json)."
echo "Next: run scripts/compare_fit.py for the MIP/slice/residual diagnostic + SSIM."

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
#SBATCH --nodes=1                          # alpha cli_filter requires explicit node count
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6                   # alpha limit: <=6 cores per requested GPU
#SBATCH --mem-per-cpu=8000                  # 6 cpus * 8 GB = 48 GB host RAM (ample)
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

# ---- workspace (compute + data + outputs live here, NOT in HOME/project) ----
# Resolve the workspace created with: ws_allocate -F horse volsplat 90
WS="$(ws_find volsplat)"
if [ -z "$WS" ]; then echo "Workspace 'volsplat' not found - run ws_allocate first." >&2; exit 1; fi

# Dataset location: copy the DRO data into the workspace first (see README).
# Override at submit time with:  sbatch --export=ALL,DRO_ROOT=/some/path fit_ctc_frame.sbatch
DRO_ROOT="${DRO_ROOT:-$WS/Fluo-N3DL-DRO}"               # must hold 01/ (and 01_GT/ later)
OUT_DIR="$WS/runs/ctc_t000_gpu"

# ---- the fit ----------------------------------------------------------------
# GPU-appropriate config: bigger ROI, far more Gaussians, enough iters to converge.
# Anisotropic init (xy=2.0, z=0.4 voxels) is on by default for DRO's 5x z-anisotropy.
# Call python directly (single-task job) - avoids the alpha srun cli_filter.
python scripts/train_ctc.py \
    --root "$DRO_ROOT" \
    --sequence 01 \
    --frame 0 \
    --crop --target-shape 64 256 256 \
    --num-gaussians 40000 \
    --iterations 5000 \
    --batch-size 8192 \
    --out-dir "$OUT_DIR"

echo "Done. Outputs in $OUT_DIR (final.pt, history.json, roi.npy)."
echo "Next: python scripts/compare_fit.py --volume $OUT_DIR/roi.npy \\"
echo "        --checkpoint $OUT_DIR/final.pt --out-dir $OUT_DIR/compare"

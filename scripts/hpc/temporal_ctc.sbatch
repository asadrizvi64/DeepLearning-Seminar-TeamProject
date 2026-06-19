#!/bin/bash
# Real-data shared-identity temporal reconstruction (P5) on 4 consecutive DRO frames.
# Exports per-frame .splat files for 3D+t scrubbing in viewer/index.html.
#
# Submit from the repo root:
#   sbatch scripts/hpc/temporal_ctc.sbatch
# Watch:
#   squeue --me
#   tail -f $WS/logs/volsplat-temporal_*.out   (after it starts)

#SBATCH --job-name=volsplat-temporal
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

mkdir -p logs
module purge
module load release/24.04 GCCcore/12.3.0 Python/3.11.3
set -u
source "${VENV_DIR:-$HOME/volsplat-venv}/bin/activate"

WS="$(ws_find volsplat)"
if [ -z "$WS" ]; then echo "Workspace 'volsplat' not found." >&2; exit 1; fi

trap 'cp "/tmp/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$WS/logs/" 2>/dev/null || true' EXIT
mkdir -p "$WS/logs" "$WS/runs/ctc_temporal"

DRO_ROOT="${DRO_ROOT:-$WS/Fluo-N3DL-DRO}"
REPO=/projects/p_scads_mlhpc/DeepLearning-Seminar-TeamProject

# Count available frames so the job works with 1 or more frames in the workspace
N_FRAMES=$(ls "$DRO_ROOT/01/t"*.tif 2>/dev/null | wc -l)
N_FRAMES=$(( N_FRAMES > 4 ? 4 : N_FRAMES ))
N_FRAMES=$(( N_FRAMES < 1 ? 1 : N_FRAMES ))
echo "Found $N_FRAMES frame(s) in $DRO_ROOT/01/"

python "$REPO/scripts/train_temporal_ctc.py" \
    --root "$DRO_ROOT" \
    --sequence 01 \
    --frames "$N_FRAMES" \
    --target-shape 24 64 64 \
    --num-gaussians 1500 \
    --iters-first 1500 \
    --iters-warm 400 \
    --out-dir "$WS/runs/ctc_temporal"

echo "Temporal reconstruction done. Splats in $WS/runs/ctc_temporal/splats/"
echo "Download and load all t*.splat into viewer/index.html for 3D+t scrubbing."

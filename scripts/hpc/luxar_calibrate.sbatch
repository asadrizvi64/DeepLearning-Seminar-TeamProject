#!/bin/bash
# Luxar's own budget calibration (blind-spot CV, K*) on a whole C. elegans frame,
# over the same K grid as the rate-detectability sweep.
#
# Submit from the code folder on a login node:
#     sbatch --export=ALL,SEQ=01,FRAME=150 scripts/hpc/luxar_calibrate.sbatch
#     sbatch --export=ALL,SEQ=01,FRAME=194 scripts/hpc/luxar_calibrate.sbatch

#SBATCH --job-name=luxar-cal
# no --partition: each cluster (Alpha, Capella) uses its default GPU partition;
# override with  sbatch -p <partition> ...
#SBATCH --account=p_scads_mlhpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --time=08:00:00
#SBATCH --output=/tmp/%x_%j.out

set -euo pipefail
WS="$(ws_find volsplat)"
mkdir -p "$WS/logs"
trap 'cp -f /tmp/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out "$WS/logs/" 2>/dev/null || true' EXIT

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
SEQ="${SEQ:-01}"
FRAME="${FRAME:-194}"
OUT="${OUT:-$WS/runs/fidelity/calibrate_ce_s${SEQ}_t$(printf %03d "$FRAME")}"

source "$WS/luxenv/bin/activate"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
nvidia-smi --query-gpu=name,memory.total --format=csv

python "$REPO/scripts/fidelity/luxar_calibrate.py" --dataset CE --seq "$SEQ" --frame "$FRAME" \
    --root "$WS/Fluo-N3DH-CE" --out "$OUT" --device cuda

echo "Results: $OUT/summary.json"

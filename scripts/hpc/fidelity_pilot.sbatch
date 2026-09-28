#!/bin/bash
# Go/no-go pilot for the "Beyond PSNR" audit: Luxar fits of a Drosophila crop at
# several splat budgets, each scored against codecs matched to the SAME bytes on disk.
#
# Submit from the repo root on a login node:
#     sbatch scripts/hpc/fidelity_pilot.sbatch
# Optional overrides:
#     sbatch --export=ALL,SEEDS="500 2000 8000",SHAPE="64 256 256" scripts/hpc/fidelity_pilot.sbatch
# Watch:  squeue --me      Log:  ls -t $(ws_find volsplat)/logs/ | head

#SBATCH --job-name=fidelity-pilot
#SBATCH --partition=alpha
#SBATCH --account=p_scads_mlhpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/%x_%j.out

set -euo pipefail
WS="$(ws_find volsplat)"
mkdir -p "$WS/logs"
trap 'cp -f /tmp/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out "$WS/logs/" 2>/dev/null || true' EXIT

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
DRO_ROOT="${DRO_ROOT:-$WS/Fluo-N3DL-DRO}"
SEEDS="${SEEDS:-250 500 1000 2000 4000 8000 16000}"
ORIGIN="${ORIGIN:-32 416 544}"
SHAPE="${SHAPE:-64 128 128}"
OUT="${OUT:-$WS/runs/fidelity/pilot_dro_$(echo $SHAPE | tr ' ' 'x')}"

source "$WS/luxenv/bin/activate"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python "$REPO/scripts/fidelity/rate_detectability.py" \
    --out "$OUT" --dro "$DRO_ROOT" --device cuda --n-iters 1000 \
    --origin $ORIGIN --shape $SHAPE --luxar-seeds $SEEDS

echo "Results: $OUT/rate_detectability.csv"

#!/bin/bash
# Luxar fits at several splat budgets, each scored against codecs matched to the SAME
# bytes on disk (scripts/fidelity/rate_detectability.py).
#
# Submit from the code folder on a login node:
#     sbatch scripts/hpc/fidelity_pilot.sbatch                     # DRO crop pilot
#     sbatch --export=ALL,DATASET=CE,FRAME=150 scripts/hpc/fidelity_pilot.sbatch
#     sbatch --export=ALL,DATASET=CE,FRAME=194 scripts/hpc/fidelity_pilot.sbatch
#     sbatch --export=ALL,DATASET=CE,SEQ=02,FRAME=180 scripts/hpc/fidelity_pilot.sbatch
# CE runs use the whole volume (every nucleus is labelled). Watch: squeue --me
# Log: ls -t $(ws_find volsplat)/logs/ | head

#SBATCH --job-name=fidelity-pilot
# no --partition: each cluster (Alpha, Capella) uses its default GPU partition;
# override with  sbatch -p <partition> ...
#SBATCH --account=p_scads_mlhpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --time=06:00:00
#SBATCH --output=/tmp/%x_%j.out

set -euo pipefail
WS="$(ws_find volsplat)"
mkdir -p "$WS/logs"
trap 'cp -f /tmp/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out "$WS/logs/" 2>/dev/null || true' EXIT

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
DATASET="${DATASET:-DRO}"
FRAME="${FRAME:-0}"
SEQ="${SEQ:-01}"
if [ "$DATASET" = "CE" ]; then
    ROOT="${ROOT:-$WS/Fluo-N3DH-CE}"
    # budgets around the cliff seen on seq 01 (1-7 splats per nucleus) plus a plateau check
    SEEDS="${SEEDS:-1000 1500 2000 3000 4000 8000}"
    WHERE="--full"
    TAG="ce_s${SEQ}_t$(printf %03d "$FRAME")"
else
    ROOT="${ROOT:-$WS/Fluo-N3DL-DRO}"
    SEEDS="${SEEDS:-250 500 1000 2000 4000 8000 16000}"
    ORIGIN="${ORIGIN:-32 416 544}"
    SHAPE="${SHAPE:-64 128 128}"
    WHERE="--origin $ORIGIN --shape $SHAPE"
    TAG="dro_$(echo $SHAPE | tr ' ' 'x')"
fi
OUT="${OUT:-$WS/runs/fidelity/pilot_$TAG}"
CODECS="${CODECS:-jpeg2k jpegxl zfp downsample}"

source "$WS/luxenv/bin/activate"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python "$REPO/scripts/fidelity/rate_detectability.py" \
    --out "$OUT" --dataset "$DATASET" --root "$ROOT" --frame "$FRAME" --seq "$SEQ" $WHERE \
    --device cuda --n-iters 1000 --luxar-seeds $SEEDS --codecs $CODECS

echo "Results: $OUT/rate_detectability.csv"

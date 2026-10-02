#!/bin/bash
# Study A, stage A2: Cellpose in its volumetric mode (do_3D) on the Luxar fits and the
# JPEG-XL reconstructions matched to their bytes (paper/DECISION_RULES.md, sec. 4).
# JPEG2000 is excluded: the cluster run folders still hold the pre-fix (broken) JPEG2000
# volumes, and the A2 rule compares JPEG-XL with Luxar only.
# One array task per frame; reads the pilot run folders' recon/ volumes on the workspace.
#
# One-time, on a login node:  source "$(ws_find volsplat)/luxenv/bin/activate" && uv pip install "cellpose<4"
# Submit from the code folder:  sbatch --array=0-3 scripts/hpc/cellpose3d.sbatch   # 4 paper frames
#                               sbatch --array=4-11 scripts/hpc/cellpose3d.sbatch  # A1 frames (after their fits)
# Pack:  cd "$(ws_find volsplat)/runs/fidelity" && tar -czf ~/cellpose3d.tar.gz cellpose3d_*

#SBATCH --job-name=cellpose3d
#SBATCH --account=p_scads_mlhpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --time=05:00:00
#SBATCH --output=/tmp/%x_%A_%a.out

set -euo pipefail
WS="$(ws_find volsplat)"
mkdir -p "$WS/logs"
trap 'cp -f /tmp/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out "$WS/logs/" 2>/dev/null || true' EXIT
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
F="$WS/runs/fidelity"

TAGS=(ce_t150 ce_t194 ce_s02_t150 ce_s02_t180
      ce_s01_t110 ce_s01_t130 ce_s01_t170 ce_s01_t185
      ce_s02_t110 ce_s02_t130 ce_s02_t165 ce_s02_t185)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
DIRS="$F/pilot_$TAG"
[ -d "$F/pilot_${TAG}_hi" ] && DIRS="$DIRS $F/pilot_${TAG}_hi"

source "$WS/luxenv/bin/activate"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python -c "import cellpose, torch; print('cellpose ok, cuda', torch.cuda.is_available())"
python "$REPO/scripts/fidelity/cellpose_score.py" $DIRS --out "$F/cellpose3d_$TAG.csv" \
    --root "$WS/Fluo-N3DH-CE" --do-3d --gpu --only 'luxar|jpegxl'

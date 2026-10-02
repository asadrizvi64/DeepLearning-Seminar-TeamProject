#!/usr/bin/env bash
# Study A, stage A1 (robustness frames): codec side, run locally (no GPU needed).
# Luxar fits for the same frames run on the cluster (scripts/hpc/fidelity_pilot.sbatch).
# Usage: bash scripts/fidelity/run_a1_local.sh [codecs|log|cellpose|all]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-C:/Users/HP/cpenv/Scripts/python.exe}
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
FRAMES="01:110 01:130 01:170 01:185 02:110 02:130 02:165 02:185"
STAGE=${1:-all}
F=runs/fidelity

for sf in $FRAMES; do
  seq=${sf%%:*}; t=${sf##*:}; tag=ce_s${seq}_t${t}
  if [[ $STAGE == codecs || $STAGE == all ]] && [[ ! -f $F/codecs_$tag/rate_detectability.csv ]]; then
    echo "=== codecs $tag $(date +%T)"
    $PY scripts/fidelity/rate_detectability.py --dataset CE --seq $seq --frame $t --full \
        --out $F/codecs_$tag --codecs jpeg2k jpegxl --target-kib 50 100 200 400 800
  fi
done
for sf in $FRAMES; do
  seq=${sf%%:*}; t=${sf##*:}; tag=ce_s${seq}_t${t}
  dirs=$F/codecs_$tag
  [[ -d $F/luxar_render_$tag ]] && dirs="$dirs $F/luxar_render_$tag"
  if [[ $STAGE == log || $STAGE == all ]]; then
    echo "=== log $tag $(date +%T)"
    $PY scripts/fidelity/log_rescore.py $dirs --out $F/log_$tag.csv
  fi
  if [[ $STAGE == cellpose || $STAGE == all ]]; then
    echo "=== cellpose $tag $(date +%T)"
    $PY scripts/fidelity/cellpose_score.py $dirs --out $F/cellpose_$tag.csv
  fi
done
echo "=== done $(date +%T)"

#!/usr/bin/env bash
# After the JPEG2000 axis fix: re-encode JPEG2000 on the v1.0 frames, then rescore every
# detector (cached detections are reused for all other files).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-C:/Users/HP/cpenv/Scripts/python.exe}
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
F=runs/fidelity
TAGS=${TAGS-"ce_s02_t180 ce_t194 ce_s02_t150 ce_t150 ce_s01_t100"}
for tag in ${REDO_TAGS-$TAGS}; do
  echo "=== redo jpeg2k $tag $(date +%T)"
  $PY scripts/fidelity/redo_jpeg2k.py $tag
done
for tag in $TAGS; do
  dirs=$F/codecs_$tag
  for d in $F/luxar_render_${tag} $F/luxar_render_${tag}_hi; do if [[ -d $d ]]; then dirs="$dirs $d"; fi; done
  echo "=== log $tag $(date +%T)";       $PY scripts/fidelity/log_rescore.py $dirs --out $F/log_$tag.csv
  echo "=== watershed $tag $(date +%T)"; $PY scripts/fidelity/watershed_score.py $dirs --out $F/watershed_$tag.csv
done
for tag in $TAGS; do
  dirs=$F/codecs_$tag
  for d in $F/luxar_render_${tag} $F/luxar_render_${tag}_hi; do if [[ -d $d ]]; then dirs="$dirs $d"; fi; done
  echo "=== cellpose $tag $(date +%T)";  $PY scripts/fidelity/cellpose_score.py $dirs --out $F/cellpose_$tag.csv
done
echo "=== done $(date +%T)"

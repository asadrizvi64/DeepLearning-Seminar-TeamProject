#!/usr/bin/env bash
# Import the Capella bundle fidelity_a1.tar.gz (A1 Luxar fits on 8 frames, embryo-2 full-data
# fits at K* = 64000, Cellpose 3D results), then render, add same-size JPEG2000, score all
# detectors and regenerate the analysis and paper.
#
#   bash scripts/fidelity/ingest_a1.sh [path/to/fidelity_a1.tar.gz]
#
# Every step is cached/resumable; re-running after a partial bundle only does the new work.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-C:/Users/HP/cpenv/Scripts/python.exe}
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
TAR=${1:-$HOME/Downloads/fidelity_a1.tar.gz}
F=runs/fidelity
C=runs/fidelity_a1/fidelity
A1="ce_s01_t110 ce_s01_t130 ce_s01_t170 ce_s01_t185 ce_s02_t110 ce_s02_t130 ce_s02_t165 ce_s02_t185"

echo "=== unpack $(date +%T)"
rm -rf runs/fidelity_a1 && mkdir -p runs/fidelity_a1 && tar --force-local -xzf "$TAR" -C runs/fidelity_a1
for d in $C/pilot_*; do
    echo "  $(basename $d): $(ls $d/luxar | wc -l) Luxar fits, $(($(wc -l < $d/rate_detectability.csv) - 1)) rows"
done

echo "=== Cellpose 3D results $(date +%T)"
for p in $C/cellpose3d_*; do
    if [[ -d $p ]]; then mkdir -p $F/$(basename $p) && cp $p/* $F/$(basename $p)/; else cp $p $F/; fi
done
ls $F | grep -c '^cellpose3d_.*\.csv$' || true

echo "=== render Luxar fits $(date +%T)"
for tag in $A1; do
    if [[ -d $C/pilot_$tag/luxar ]] && [[ -n "$(ls $C/pilot_$tag/luxar)" ]]; then
        $PY scripts/fidelity/render_luxar.py $C/pilot_$tag --out $F/luxar_render_$tag
    fi
done
for t in 150 180; do
    if [[ -d $C/pilot_ce_s02_t${t}_k64/luxar ]]; then
        $PY scripts/fidelity/render_luxar.py $C/pilot_ce_s02_t${t}_k64 --out $F/luxar_render_ce_s02_t${t}_k64
    fi
done

echo "=== JPEG2000 at the new Luxar sizes $(date +%T)"
$PY scripts/fidelity/redo_jpeg2k.py --add-only $A1 ce_s02_t150 ce_s02_t180

for stage in log watershed cellpose; do
    for tag in $A1 ce_s02_t150 ce_s02_t180; do
        dirs=$F/codecs_$tag
        for d in $F/luxar_render_${tag} $F/luxar_render_${tag}_hi $F/luxar_render_${tag}_k64; do
            if [[ -d $d ]]; then dirs="$dirs $d"; fi
        done
        echo "=== $stage $tag $(date +%T)"
        case $stage in
            log)       $PY scripts/fidelity/log_rescore.py $dirs --out $F/log_$tag.csv > /dev/null ;;
            watershed) $PY scripts/fidelity/watershed_score.py $dirs --out $F/watershed_$tag.csv > /dev/null ;;
            cellpose)  $PY scripts/fidelity/cellpose_score.py $dirs --out $F/cellpose_$tag.csv > /dev/null 2>&1 ;;
        esac
    done
done

echo "=== analysis + paper $(date +%T)"
bash scripts/fidelity/reproduce.sh
echo "=== INGEST DONE $(date +%T)"

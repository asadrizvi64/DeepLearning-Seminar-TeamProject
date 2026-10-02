#!/usr/bin/env bash
# Study A (v1.0 freeze): regenerate every table, figure, number and rule verdict of the paper
# from the per-frame result files, then build the PDF.  Nothing in the paper is typed by hand.
#
#   bash scripts/fidelity/reproduce.sh            # analysis + PDF (minutes, CPU only)
#
# The per-frame result files themselves come from the pipeline below (hours; Luxar fits need
# a GPU).  Each step is resumable and caches its detections, so re-running is cheap.
#   1. Luxar fits + codecs at the fits' bytes   scripts/hpc/fidelity_pilot.sbatch (cluster)
#      Luxar self-calibration (K*)               scripts/hpc/luxar_calibrate.sbatch (cluster)
#   2. Codecs at 50-800 KiB                      scripts/fidelity/rate_detectability.py --codecs ...
#   3. Re-render the fits locally                scripts/fidelity/render_luxar.py
#   4. Detectors on every reconstruction         scripts/fidelity/log_rescore.py      (LoG)
#                                                scripts/fidelity/cellpose_score.py   (Cellpose, 2D stitched)
#                                                scripts/fidelity/watershed_score.py  (classical 3D)
#      JPEG2000 after the 2026-10-02 axis fix:   scripts/fidelity/run_redo_jpeg2k.sh
#                                                scripts/hpc/cellpose3d.sbatch        (Cellpose 3D, GPU)
#   A1 frames: scripts/fidelity/run_a1_local.sh (local side) and scripts/hpc/submit_a1.sh (fits)
# Environments: C:/Users/HP/cpenv (Python 3.12, cellpose<4, luxar); LaTeX via conda env "tex"
# (tectonic).  Data: Cell Tracking Challenge Fluo-N3DH-CE training set (sequences 01, 02).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-C:/Users/HP/cpenv/Scripts/python.exe}
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MPLBACKEND=Agg

$PY scripts/test_metrics.py                 # matcher / scorer regression suite first
$PY scripts/fidelity/results_table.py       # runs/fidelity/results_all.csv (the single table)
$PY scripts/fidelity/matched_pairs.py       # runs/fidelity/matched_pairs.csv (Luxar vs same-size codecs, CIs)
$PY scripts/fidelity/check_rules.py         # runs/fidelity/rule_check.md  (rules 1-3, 5)
$PY scripts/fidelity/a1_crossings.py        # runs/fidelity/rule_check_a1.md (rules 4), a1_crossings.csv
$PY scripts/fidelity/bootstrap_ci.py        # paper/numbers_ci.tex, runs/fidelity/bootstrap_ci.csv
$PY scripts/fidelity/paper_numbers.py       # paper/numbers.tex, tables, figures
conda run -n tex tectonic -X compile paper/main.tex
echo "Done: paper/main.pdf"

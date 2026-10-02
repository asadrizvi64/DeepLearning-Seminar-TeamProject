#!/bin/bash
# Study A, stage A1: Luxar fits on the 8 robustness frames (paper/DECISION_RULES.md, sec. 4).
# Run from the code folder on a Capella login node:  bash scripts/hpc/submit_a1.sh
# Codecs at 50-800 KiB run locally (scripts/fidelity/run_a1_local.sh); here only the two
# codecs the claims rest on are matched to each fit's bytes.
set -euo pipefail
for sf in 01:110 01:130 01:170 01:185 02:110 02:130 02:165 02:185; do
    seq=${sf%%:*}; t=${sf##*:}
    sbatch --job-name=a1-s${seq}t${t} --time=03:00:00 \
        --export=ALL,DATASET=CE,SEQ=$seq,FRAME=$t,SEEDS="500 1000 1500 2000 4000 16000",CODECS="jpeg2k jpegxl" \
        scripts/hpc/fidelity_pilot.sbatch
done
squeue --me
# When all are done, pack (fits only, no recon volumes) and copy to the laptop:
#   cd "$(ws_find volsplat)/runs" && tar -czf ~/fidelity_a1.tar.gz --exclude='*/recon' fidelity/pilot_ce_s01_t{110,130,170,185} fidelity/pilot_ce_s02_t{110,130,165,185}
#   (laptop)  scp syas272h@login1.capella.hpc.tu-dresden.de:fidelity_a1.tar.gz Downloads/

#!/bin/bash
# One-time environment setup for volsplat on the TU Dresden Alpha cluster.
# Run this ONCE on a login node (login[1-2].alpha.hpc.tu-dresden.de) after you
# have copied the repo to a workspace. It builds a venv with CUDA PyTorch and
# installs volsplat in editable mode.
#
# Usage (on a login node):
#   bash scripts/hpc/setup_env.sh
#
# Notes
# -----
# * Alpha nodes have NVIDIA A100 (40 GiB). A100 requires CUDA 11+.
# * The compendium recommends virtualenv inside a workspace for isolation.
# * If you prefer the module system instead of pip-installed torch, run
#       module spider PyTorch
#   to find the exact hierarchical module name, load it, and skip the torch
#   pip install below (install the rest with --no-deps as needed).
set -euo pipefail

# ---- where the venv lives. Prefer a workspace, not $HOME (quota). ------------
# Create a workspace first, e.g.:
#   ws_allocate -F horse volsplat 90      # 90-day workspace named "volsplat"
# then point VENV_DIR at it.
VENV_DIR="${VENV_DIR:-$HOME/volsplat-venv}"   # <-- override to a workspace path

# ---- load a Python toolchain ------------------------------------------------
# Module names drift between cluster releases; if these fail, run
#   module spider Python
# and substitute the names it reports.
module purge
module load release/24.04 GCCcore/12.3.0 Python/3.11.3 || {
    echo "Module load failed - run 'module spider Python' and edit this script." >&2
    exit 1
}

# ---- build the venv ---------------------------------------------------------
python -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# ---- CUDA PyTorch (cu121 wheels work on A100) -------------------------------
pip install torch --index-url https://download.pytorch.org/whl/cu121

# ---- volsplat + deps (editable) ---------------------------------------------
# Run from the repo root so `pip install -e .` finds pyproject.toml.
pip install -e .

# ---- sanity check -----------------------------------------------------------
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY
echo
echo "Setup done. venv at: $VENV_DIR"
echo "Note: torch.cuda.is_available() prints False on the LOGIN node (no GPU there)."
echo "It will be True inside a SLURM job on the alpha partition."

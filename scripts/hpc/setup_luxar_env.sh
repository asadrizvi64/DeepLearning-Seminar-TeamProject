#!/bin/bash
# One-time setup of a Luxar (royerlab/luxar) environment on TU Dresden Alpha.
# Run on a LOGIN node (needs internet; no GPU needed to install):
#     bash scripts/hpc/setup_luxar_env.sh
#
# Luxar needs Python >= 3.12; the cluster's Python module is 3.11, so uv supplies its
# own managed CPython 3.12 (downloaded once into ~/.local/share/uv). Everything big
# goes in the workspace, not $HOME.
set -euo pipefail

WS="$(ws_find volsplat)"
ENV="$WS/luxenv"
LUXAR="$WS/luxar"

# ---- uv (single static binary into ~/.local/bin)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# ---- Luxar source
if [ ! -d "$LUXAR/.git" ]; then
    git clone --depth 1 https://github.com/royerlab/luxar.git "$LUXAR"
fi
git -C "$LUXAR" log -1 --format='luxar %h %cd'

# ---- Python 3.12 env with CUDA torch, Luxar (fitting extras) and the scorer's deps
uv venv --python 3.12 "$ENV"
uv pip install --python "$ENV/bin/python" torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$ENV/bin/python" -e "$LUXAR[gsplats]" \
    tifffile imagecodecs scikit-image pandas matplotlib tqdm

"$ENV/bin/python" - <<'PY'
import torch, numpy, zarr
from luxar.gsplats import fit_gaussian_splats  # noqa: F401
print('env ok | torch', torch.__version__, '| numpy', numpy.__version__, '| zarr', zarr.__version__)
print('cuda visible here (expected False on a login node):', torch.cuda.is_available())
PY
echo "Done. Activate with:  source $ENV/bin/activate"

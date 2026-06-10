# Running volsplat on the TU Dresden Alpha cluster

Quick guide to get a converged real-data fit on a GPU. The Alpha (Alpha Centauri)
partition has 8× NVIDIA A100 (40 GiB) per node and is SLURM-scheduled. You have
~1000 GPU-h/month — a single-frame fit costs minutes, so that's plenty for now.

Cluster docs: https://compendium.hpc.tu-dresden.de

## 0. Before you start — fill in these placeholders

| Placeholder | Where | What it is |
|---|---|---|
| `PLACEHOLDER_PROJECT` | `fit_ctc_frame.sbatch` `--account` | your HPC project/account name (the thing you were added to) |
| `PLACEHOLDER_PATH` | `fit_ctc_frame.sbatch` `DRO_ROOT` | workspace path holding `Fluo-N3DL-DRO/01/` |
| `VENV_DIR` | both scripts (optional) | where the venv lives; default `$HOME/volsplat-venv` |

Find your account name with: `sacctmgr show user $USER -s format=user,account`

## 1. Get the code + data onto the cluster

```bash
# From a login node: make a workspace (don't use $HOME for big data — it has quota)
ws_allocate -F horse volsplat 90        # 90-day workspace
ws_find volsplat                        # prints the path, e.g. /data/horse/ws/<user>-volsplat

# Copy the repo (run from your local machine; use the export nodes for big files)
#   git clone <your repo>      ... or scp/rsync the project dir
# Copy the dataset (~GBs) via the dedicated transfer nodes, NOT the login nodes:
#   https://compendium.hpc.tu-dresden.de/data_transfer/datamover/
```

The data only needs the `01/` frames (and `01_GT/` later for tracking/E11).

## 2. One-time environment setup (login node)

```bash
cd <repo root>
bash scripts/hpc/setup_env.sh
```

This builds a venv with CUDA PyTorch (cu121, A100-compatible) and installs
volsplat editable. `torch.cuda.is_available()` prints **False** on the login node
(no GPU there) — that's expected; it's True inside a job.

If the `module load` line fails, the module names have changed — run
`module spider Python` and `module spider PyTorch` and edit the scripts.

## 3. Submit the fit

```bash
sbatch scripts/hpc/fit_ctc_frame.sbatch
squeue --me                    # watch the queue
tail -f logs/volsplat-ctc_*.out
```

Default config: frame 0, 64×256×256 ROI, **40 000 Gaussians, 5000 iters** — the
real budget the CPU box couldn't afford. Outputs land in `runs/ctc_t000_gpu/`.

## 4. Interactive debugging (optional)

For a quick interactive GPU shell instead of batch:

```bash
srun --partition=alpha --account=PLACEHOLDER_PROJECT --gres=gpu:1 \
     --cpus-per-task=8 --mem-per-cpu=8000 --time=1:00:00 --pty bash -l
# then: source $VENV_DIR/bin/activate  and run python interactively
```

## 5. Diagnose the result

```bash
# residual / MIP / slice / SSIM diagnostic (same one we used locally)
python scripts/compare_fit.py \
    --checkpoint runs/ctc_t000_gpu/final.pt \
    --volume <the cropped ROI>     # see note below
```

**Note:** `compare_fit.py` currently re-loads a plain volume; for the CTC crop it
needs the same ROI the fit used. Two options:
1. (quick) reuse the inline re-crop diagnostic we ran locally, or
2. (better, recommended) have `train_ctc.py` save the cropped ROI as `.npy`
   next to `final.pt` so `compare_fit.py --volume that.npy` is exact. This is the
   small reproducibility fix discussed — worth doing before the GPU run so the
   diagnostic is reproducible.

## What changes vs the CPU runs

| | CPU (local) | A100 (here) |
|---|---|---|
| per-iter cost | ~11 s (8k Gaussians) | ~milliseconds |
| feasible budget | 4k Gaussians, 400 iters | 40k+ Gaussians, 5000+ iters |
| converged single frame | hours | minutes |
| all 50 frames (P5) | infeasible | feasible |

This unblocks: the converged single-frame fit, E3 (budget sweep) on **real** data,
E4 (densification re-tuning), and eventually the temporal phases.

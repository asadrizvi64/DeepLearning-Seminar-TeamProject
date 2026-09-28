"""Durable provenance for every experimental result row.

Checkpointing made results survive a killed process. This makes them survive TIME:
a durable number whose configuration is not recorded can still become ambiguous later.

This project has already had two conclusions reverse because of silent configuration
drift rather than any modelling error:
  * `baseline_fragility` ranked candidates by LoG instead of intensity, which reverses
    at large budgets -- recall read 0.543 instead of 0.868 and the persistence verdict
    flipped once it was found;
  * detection thresholds were absolute rather than offset-invariant, so a run carrying
    a background pedestal was scored against one without it.

Neither was visible in the saved numbers. Both would have been visible in a stamped
config. So every experiment attaches `run_metadata(...)` to each row it writes.

`metrics_fingerprint` hashes the SCORING code specifically -- a change to how cells are
detected or metrics computed silently invalidates cross-run comparison, and is exactly
the class of drift that has bitten this project.

Usage:
    from volsplat.provenance import run_metadata
    meta = run_metadata(experiment='scale_cap', k=250, iters=3000,
                        lr_position=0.0016, loss_mode='mse')
    rows.append({**result, **meta})
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import time
from pathlib import Path

_PKG = Path(__file__).resolve().parent

# Modules whose content defines how a result is SCORED. A change here means numbers
# from before and after are not directly comparable.
_SCORING_MODULES = ('cellmetrics.py', 'metrics.py', 'ablation.py')


def _git(*args, default=None):
    try:
        out = subprocess.run(('git',) + args, cwd=_PKG.parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else default
    except Exception:
        return default


def metrics_fingerprint() -> str:
    """Short content hash of the scoring code (detection + metrics + fitting)."""
    h = hashlib.sha256()
    for name in _SCORING_MODULES:
        p = _PKG / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def run_metadata(experiment: str = None, **config) -> dict:
    """Provenance stamp to merge into every result row.

    Anything passed as **config is recorded verbatim -- pass the knobs that define the
    run (seed, k, iterations, lr_position, loss_mode, background, max_scale, ...).
    """
    commit = _git('rev-parse', '--short', 'HEAD', default='nogit')
    dirty = _git('status', '--porcelain', default='')
    meta = {
        'experiment': experiment,
        'git_commit': commit,
        'git_dirty': bool(dirty) if dirty is not None else None,
        'metrics_fingerprint': metrics_fingerprint(),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'python': platform.python_version(),
    }
    meta.update({f'cfg_{k}': v for k, v in config.items()})
    return meta


def describe() -> str:
    m = run_metadata()
    return (f"commit {m['git_commit']}{'+dirty' if m['git_dirty'] else ''}  "
            f"metrics {m['metrics_fingerprint']}  {m['timestamp']}")


if __name__ == '__main__':
    print(describe())

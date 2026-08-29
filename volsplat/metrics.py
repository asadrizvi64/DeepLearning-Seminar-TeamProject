"""Reconstruction metrics for the ablation framework.

Every fit in the ablation grid is scored on the same battery of metrics so runs are
directly comparable across the (dataset x init x k x parameterization) axes.

Fidelity metrics (higher better unless noted):
    psnr        peak signal-to-noise ratio (dB)
    ssim        3D structural similarity  [-1, 1]
    mse         mean squared error                 (lower better)
    mae         mean absolute error                (lower better)
    max_error   worst-case absolute error          (lower better)
    corr        Pearson correlation of voxel intensities

Cost metrics:
    num_gaussians   splat count
    num_params      free parameters (depends on parameterization)
    bytes           num_params * 4 (float32)
    compression     num_voxels / num_params   (higher = more compression)

Efficiency:
    psnr_per_kparam   PSNR divided by (num_params / 1000)
"""
from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------- parameterization

# Free parameters PER GAUSSIAN for each parameterization.
#   position: 3   amplitude: 1
#   isotropic scale: 1        (single sigma, no rotation)
#   diagonal scale : 3        (3 axis scales, no rotation)
#   full           : 3 scales + 3 rotation DOF (quaternion has 3 effective)
PARAMS_PER_GAUSSIAN = {
    'isotropic': 3 + 1 + 1,     # 5
    'diagonal':  3 + 3 + 1,     # 7
    'full':      3 + 3 + 3 + 1, # 10
}


def param_count(num_gaussians: int, parameterization: str) -> int:
    if parameterization not in PARAMS_PER_GAUSSIAN:
        raise ValueError(
            f"Unknown parameterization {parameterization!r}. "
            f"Available: {sorted(PARAMS_PER_GAUSSIAN)}"
        )
    return int(num_gaussians) * PARAMS_PER_GAUSSIAN[parameterization]


# --------------------------------------------------------------- fidelity metrics

def _to_numpy(a):
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(a)


def mse(pred, target) -> float:
    pred, target = _to_numpy(pred), _to_numpy(target)
    return float(((pred - target) ** 2).mean())


def mae(pred, target) -> float:
    pred, target = _to_numpy(pred), _to_numpy(target)
    return float(np.abs(pred - target).mean())


def max_error(pred, target) -> float:
    pred, target = _to_numpy(pred), _to_numpy(target)
    return float(np.abs(pred - target).max())


def psnr(pred, target, max_val: float = 1.0) -> float:
    m = mse(pred, target)
    if m <= 0:
        return float('inf')
    return float(20.0 * math.log10(max_val / math.sqrt(m)))


def pearson_corr(pred, target) -> float:
    pred, target = _to_numpy(pred).ravel(), _to_numpy(target).ravel()
    if pred.std() < 1e-12 or target.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def ssim3d(pred, target, data_range: float = 1.0) -> float:
    """3D structural similarity over the whole volume."""
    pred, target = _to_numpy(pred), _to_numpy(target)
    from skimage.metrics import structural_similarity
    # win_size must be <= smallest dim and odd; default 7 fails on thin volumes.
    win = min(7, min(pred.shape))
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)
    return float(structural_similarity(
        target, pred, data_range=data_range, win_size=win,
    ))


# --------------------------------------------------------------- battery

def compute_all(
    pred,
    target,
    num_gaussians: int,
    parameterization: str,
    fit_seconds: float = None,
    data_range: float = 1.0,
) -> dict:
    """Score one reconstruction on the full metric battery. Returns a flat dict
    suitable for a pandas row."""
    pred, target = _to_numpy(pred), _to_numpy(target)
    n_params = param_count(num_gaussians, parameterization)
    n_voxels = int(target.size)
    p = psnr(pred, target, max_val=data_range)

    row = {
        'num_gaussians':  int(num_gaussians),
        'parameterization': parameterization,
        'num_params':     n_params,
        'bytes':          n_params * 4,
        'num_voxels':     n_voxels,
        'compression':    n_voxels / max(n_params, 1),
        'psnr':           p,
        'ssim':           ssim3d(pred, target, data_range=data_range),
        'mse':            mse(pred, target),
        'mae':            mae(pred, target),
        'max_error':      max_error(pred, target),
        'corr':           pearson_corr(pred, target),
        'psnr_per_kparam': p / max(n_params / 1000.0, 1e-9),
    }
    if fit_seconds is not None:
        row['fit_seconds'] = float(fit_seconds)
        row['psnr_per_second'] = p / max(fit_seconds, 1e-9)
    return row

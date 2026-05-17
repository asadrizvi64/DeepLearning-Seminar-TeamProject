"""Loss functions and metrics."""
import math
import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def psnr(pred, target, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB. Accepts torch tensors or numpy arrays."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    mse = ((pred - target) ** 2).mean()
    if mse <= 0:
        return float('inf')
    return float(20.0 * math.log10(max_val / math.sqrt(float(mse))))

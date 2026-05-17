"""Basic adaptive density control for a GaussianSet.

A stripped-down version of the 3DGS clone/split/prune heuristics. Phase 3 / E4 will
re-tune these for sparse microscopy data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussians import GaussianSet


def build_optimizer(
    gs: GaussianSet,
    lr_position: float = 0.0016,
    lr_scale: float = 0.005,
    lr_rotation: float = 0.001,
    lr_amp: float = 0.05,
) -> torch.optim.Adam:
    """Adam optimizer with per-parameter learning rates following 3DGS conventions."""
    return torch.optim.Adam([
        {'params': [gs.positions],   'lr': lr_position, 'name': 'positions'},
        {'params': [gs.log_scales],  'lr': lr_scale,    'name': 'log_scales'},
        {'params': [gs.quaternions], 'lr': lr_rotation, 'name': 'quaternions'},
        {'params': [gs.amp_logits],  'lr': lr_amp,      'name': 'amp_logits'},
    ])


@torch.no_grad()
def densify(
    gs: GaussianSet,
    position_grad: torch.Tensor,
    grad_threshold: float = 2e-4,
    scale_threshold: float = 8.0,
    amp_threshold: float = 5e-3,
    max_gaussians: int = 100_000,
):
    """Apply clone / split / prune. Returns a stats dict.

    Note: this rebuilds nn.Parameters in-place. Caller MUST rebuild the optimizer after
    calling, otherwise it will point to stale parameter tensors.
    """
    N = gs.num_gaussians
    grad_norm = position_grad.norm(dim=-1)
    amp = gs.amplitudes
    max_scale = gs.scales.max(dim=-1).values

    keep_mask = amp > amp_threshold
    high_grad = grad_norm > grad_threshold
    clone_mask = high_grad & (max_scale <= scale_threshold) & keep_mask
    split_mask = high_grad & (max_scale >  scale_threshold) & keep_mask

    # Originals to keep (alive and not being replaced by split offspring)
    survivor_mask = keep_mask & ~split_mask

    pos_chunks = [gs.positions[survivor_mask]]
    ls_chunks  = [gs.log_scales[survivor_mask]]
    q_chunks   = [gs.quaternions[survivor_mask]]
    al_chunks  = [gs.amp_logits[survivor_mask]]

    # Clones: same params, position perturbed by small jitter
    if clone_mask.any():
        c_pos = gs.positions[clone_mask]
        jitter = torch.randn_like(c_pos) * 0.1
        pos_chunks.append(c_pos + jitter)
        ls_chunks.append(gs.log_scales[clone_mask])
        q_chunks.append(gs.quaternions[clone_mask])
        al_chunks.append(gs.amp_logits[clone_mask])

    # Splits: two offspring, position offset along a random unit direction, scales halved
    if split_mask.any():
        s_pos = gs.positions[split_mask]
        s_scl = gs.scales[split_mask]
        direction = torch.randn_like(s_pos)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        offset = direction * s_scl.mean(-1, keepdim=True) * 0.5
        pos_chunks += [s_pos + offset, s_pos - offset]
        ls_chunks  += [gs.log_scales[split_mask] - 0.6931, gs.log_scales[split_mask] - 0.6931]
        q_chunks   += [gs.quaternions[split_mask], gs.quaternions[split_mask]]
        al_chunks  += [gs.amp_logits[split_mask], gs.amp_logits[split_mask]]

    new_pos = torch.cat(pos_chunks, dim=0)
    new_ls  = torch.cat(ls_chunks,  dim=0)
    new_q   = torch.cat(q_chunks,   dim=0)
    new_al  = torch.cat(al_chunks,  dim=0)

    if new_pos.shape[0] == 0:
        raise RuntimeError(
            "Densification pruned every Gaussian (all amplitudes below "
            f"amp_threshold={amp_threshold}). The model has collapsed. "
            "Lower amp_threshold, increase num_gaussians, or reduce init_scale."
        )

    if new_pos.shape[0] > max_gaussians:
        # Keep top-amplitude Gaussians up to the cap
        amp_all = F.softplus(new_al)
        _, idx = torch.topk(amp_all, max_gaussians)
        new_pos, new_ls, new_q, new_al = new_pos[idx], new_ls[idx], new_q[idx], new_al[idx]

    device = gs.positions.device
    gs.positions   = nn.Parameter(new_pos.to(device))
    gs.log_scales  = nn.Parameter(new_ls.to(device))
    gs.quaternions = nn.Parameter(new_q.to(device))
    gs.amp_logits  = nn.Parameter(new_al.to(device))

    return {
        'old_count': N,
        'new_count': gs.num_gaussians,
        'cloned':    int(clone_mask.sum()),
        'split':     int(split_mask.sum()),
        'pruned':    int((~keep_mask).sum()),
    }

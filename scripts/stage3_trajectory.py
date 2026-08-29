"""Stage 3: HOW do correctly-seeded faint nuclei die during optimization?

The bias is reproduced and robust: S_A ~ 0.84 vs S_B ~ 0.38. Every mechanistic
explanation offered so far has failed a control (scalar pedestal most recently). So stop
proposing fixes and OBSERVE the failure instead.

Every nucleus starts with exactly one Gaussian on it (coverage init, no densification,
so Gaussian index i tracks nucleus i for the whole run). We log that Gaussian's state
through training and ask which of four distinguishable deaths occurs:

    amplitude collapse   a_i -> 0; the nucleus is erased
    neighbour capture    the Gaussian drifts toward a brighter neighbour
    scale blow-up        s_i broadens until the peak is no longer a peak
    detector-only loss   position/amplitude stay fine, but the reconstruction no
                         longer produces a local maximum the detector can find

Logged per tracked Gaussian, every `--log-every` iterations:
    amplitude, mean scale, displacement from init, distance to nearest OTHER Gaussian,
    model value at the nucleus centre, residual there, and the EFFECTIVE parameter
    update since the previous log (Adam changes the gradient->update relation, so raw
    gradients are not the quantity of interest -- actual parameter movement is).

Groups A / B are defined exactly as in every earlier Stage 3 experiment:
    A  nuclei the OLD initializer (local_maxima, NMS=3) would also have seeded
    B  nuclei only covered once suppression was widened  (the faint population)

Usage:
    python scripts/stage3_trajectory.py --regions 2 --k 250 --iters 3000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from volsplat.init import init_gaussians
from volsplat.densify import build_optimizer, project_parameterization
from volsplat.losses import mse_loss
from volsplat.cellmetrics import detect_cells

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'runs/stage3_trajectory'
SHAPE = (64, 128, 128)
REGIONS = [(0, 528, 144), (0, 624, 144), (0, 240, 336), (0, 528, 240)]
SEED_R = 3.0
MATCH_R = 0.765 * 13


def fit_and_trace(roi, nuclei, k, iters, log_every, seed, lr_position, device='cpu'):
    """Standard voxel-MSE fit, but tracing the nucleus-seeded Gaussians throughout."""
    torch.manual_seed(seed)
    D, H, W = roi.shape
    vol_t = torch.from_numpy(roi).to(device)
    flat = vol_t.flatten()

    # coverage init places one Gaussian per detected nucleus FIRST, then support,
    # so indices 0..n_nuc-1 correspond to nuclei in detection order.
    gs = init_gaussians(roi, k, strategy='coverage', init_scale=2.0, seed=seed).to(device)
    n_nuc = len(nuclei)
    project_parameterization(gs, 'full')
    opt = build_optimizer(gs, parameterization='full', lr_position=lr_position)

    pos0 = gs.positions.detach().clone()
    prev = dict(pos=gs.positions.detach().clone(),
                amp=gs.amplitudes.detach().clone(),
                scl=gs.scales.detach().clone())

    # nucleus centres as (x, y, z) query points
    nuc_xyz = torch.from_numpy(nuclei[:, ::-1].copy().astype(np.float32)).to(device)
    nuc_target = vol_t[nuclei[:, 0].astype(int),
                       nuclei[:, 1].astype(int),
                       nuclei[:, 2].astype(int)]

    n_int = int(2048 * 0.7)
    w = (flat + 1e-3); w = w / w.sum()
    snaps = []

    for it in range(iters):
        sel_i = torch.multinomial(w, n_int, replacement=True)
        sel_u = torch.randint(0, D * H * W, (2048 - n_int,), device=device)
        sel = torch.cat([sel_i, sel_u])
        z = sel // (H * W); rem = sel % (H * W); y = rem // W; x = rem % W
        pts = torch.stack([x.float(), y.float(), z.float()], -1) + \
            (torch.rand(sel.numel(), 3, device=device) - 0.5)
        loss = mse_loss(gs.query_density(pts), flat[sel])
        opt.zero_grad(); loss.backward(); opt.step()
        project_parameterization(gs, 'full')

        if it % log_every == 0 or it == iters - 1:
            with torch.no_grad():
                p = gs.positions.detach(); a = gs.amplitudes.detach(); s = gs.scales.detach()
                model_at_nuc = gs.query_density(nuc_xyz)
                # distance from each tracked Gaussian to the nearest OTHER Gaussian
                dd = torch.cdist(p[:n_nuc], p)
                dd[torch.arange(n_nuc), torch.arange(n_nuc)] = float('inf')
                nn_dist = dd.min(dim=1).values
                snaps.append(dict(
                    iter=it,
                    amp=a[:n_nuc].cpu().numpy().copy(),
                    scale=s[:n_nuc].mean(1).cpu().numpy().copy(),
                    disp=torch.norm(p[:n_nuc] - pos0[:n_nuc], dim=1).cpu().numpy().copy(),
                    step_pos=torch.norm(p[:n_nuc] - prev['pos'][:n_nuc], dim=1).cpu().numpy().copy(),
                    step_amp=(a[:n_nuc] - prev['amp'][:n_nuc]).abs().cpu().numpy().copy(),
                    step_scl=torch.norm(s[:n_nuc] - prev['scl'][:n_nuc], dim=1).cpu().numpy().copy(),
                    nn_dist=nn_dist.cpu().numpy().copy(),
                    model_val=model_at_nuc.cpu().numpy().copy(),
                    residual=(nuc_target - model_at_nuc).cpu().numpy().copy()))
                prev = dict(pos=p.clone(), amp=a.clone(), scl=s.clone())

    recon = gs.query_volume(roi.shape).cpu().numpy().astype(np.float32)
    return gs, snaps, recon


def recovered(recon, nuclei):
    pred = detect_cells(recon, mode='3d')
    got = np.zeros(len(nuclei), dtype=bool)
    if len(pred) and len(nuclei):
        d = np.linalg.norm(pred[:, None, :] - nuclei[None, :, :], axis=-1)
        ri, ci = linear_sum_assignment(d)
        for i, j in zip(ri, ci):
            if d[i, j] <= MATCH_R:
                got[j] = True
    return got


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--regions', type=int, default=2)
    p.add_argument('--k', type=int, default=250)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr-position', type=float, default=0.0016)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    vol = np.load(REPO / 'runs/colleague_full_volume.npy').astype(np.float32)
    rows = []

    for (z, y, x) in REGIONS[:args.regions]:
        tag = f'z{z}y{y}x{x}'
        roi = np.ascontiguousarray(vol[z:z + SHAPE[0], y:y + SHAPE[1], x:x + SHAPE[2]])
        nuclei = detect_cells(roi, mode='3d')

        old = init_gaussians(roi, args.k, strategy='local_maxima', init_scale=2.0, seed=0)
        p_old = old.positions.detach().cpu().numpy()[:, ::-1]
        d_old = np.linalg.norm(nuclei[:, None, :] - p_old[None, :, :], axis=-1).min(axis=1)
        group = np.where(d_old <= SEED_R, 'A', 'B')

        print(f'\n=== {tag}: {len(nuclei)} nuclei '
              f'(A={int((group=="A").sum())}, B={int((group=="B").sum())}) ===')
        gs, snaps, recon = fit_and_trace(roi, nuclei, args.k, args.iters,
                                         args.log_every, args.seed, args.lr_position)
        got = recovered(recon, nuclei)
        print(f'  recovered: A={got[group=="A"].mean():.3f}  B={got[group=="B"].mean():.3f}')

        for j in range(len(nuclei)):
            for s in snaps:
                rows.append(dict(region=tag, nucleus=j, group=group[j],
                                 recovered=bool(got[j]), iter=s['iter'],
                                 amp=float(s['amp'][j]), scale=float(s['scale'][j]),
                                 disp=float(s['disp'][j]), nn_dist=float(s['nn_dist'][j]),
                                 model_val=float(s['model_val'][j]),
                                 residual=float(s['residual'][j]),
                                 step_pos=float(s['step_pos'][j]),
                                 step_amp=float(s['step_amp'][j]),
                                 step_scl=float(s['step_scl'][j])))

    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(OUT / 'trajectories.csv', index=False)

    fin = df[df['iter'] == df['iter'].max()]
    print('\n' + '=' * 70)
    print('FINAL STATE by group x outcome')
    print('=' * 70)
    print(f'{"grp":>4s} {"outcome":>10s} {"n":>4s} {"amp":>8s} {"scale":>8s} '
          f'{"disp":>7s} {"nn_dist":>8s} {"model":>8s} {"resid":>8s}')
    for g in ['A', 'B']:
        for outc in [True, False]:
            s = fin[(fin.group == g) & (fin.recovered == outc)]
            if not len(s):
                continue
            print(f'{g:>4s} {"kept" if outc else "LOST":>10s} {len(s):4d} '
                  f'{s.amp.median():8.3f} {s.scale.median():8.2f} {s.disp.median():7.2f} '
                  f'{s.nn_dist.median():8.2f} {s.model_val.median():8.3f} '
                  f'{s.residual.median():8.3f}')

    # effective update magnitudes, averaged over training, by group x outcome
    print('\n=== MEAN EFFECTIVE UPDATE PER LOG STEP (Adam, actual movement) ===')
    print(f'{"grp":>4s} {"outcome":>10s} {"|d pos|":>10s} {"|d amp|":>10s} {"|d scale|":>10s}')
    for g in ['A', 'B']:
        for outc in [True, False]:
            s = df[(df.group == g) & (df.recovered == outc) & (df['iter'] > 0)]
            if not len(s):
                continue
            print(f'{g:>4s} {"kept" if outc else "LOST":>10s} '
                  f'{s.step_pos.mean():10.4f} {s.step_amp.mean():10.4f} '
                  f'{s.step_scl.mean():10.4f}')

    # which death mode? compare each LOST nucleus at start vs end
    print('\n=== DEATH MODE (lost nuclei: how did their state change?) ===')
    first = df[df['iter'] == 0].set_index(['region', 'nucleus'])
    last = fin.set_index(['region', 'nucleus'])
    lost = last[~last.recovered]
    modes = {'amplitude collapse': 0, 'scale blow-up': 0,
             'neighbour capture': 0, 'detector-only loss': 0, 'other': 0}
    for idx in lost.index:
        f, l = first.loc[idx], last.loc[idx]
        if l.amp < 0.5 * f.amp:
            modes['amplitude collapse'] += 1
        elif l.scale > 1.5 * f.scale:
            modes['scale blow-up'] += 1
        elif l.nn_dist < 0.5 * f.nn_dist:
            modes['neighbour capture'] += 1
        elif abs(l.residual) < 0.15:
            modes['detector-only loss'] += 1
        else:
            modes['other'] += 1
    tot = max(sum(modes.values()), 1)
    for m, c in sorted(modes.items(), key=lambda kv: -kv[1]):
        print(f'  {m:22s} {c:3d}  ({100*c/tot:4.1f}%)')
    json.dump(modes, open(OUT / 'death_modes.json', 'w'), indent=2)

    # figure: median trajectory per group x outcome
    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    styles = {('A', True): ('#0E7C6B', '-'), ('A', False): ('#0E7C6B', '--'),
              ('B', True): ('#B4462A', '-'), ('B', False): ('#B4462A', '--')}
    for ax, (col, lab) in zip(axes, [('amp', 'amplitude'), ('scale', 'mean scale (vox)'),
                                     ('model_val', 'model value at nucleus'),
                                     ('nn_dist', 'distance to nearest Gaussian')]):
        for (g, outc), (c, ls) in styles.items():
            s = df[(df.group == g) & (df.recovered == outc)]
            if not len(s):
                continue
            m = s.groupby('iter')[col].median()
            ax.plot(m.index, m.values, ls, color=c, lw=2,
                    label=f'{g} {"kept" if outc else "LOST"}')
        ax.set_xlabel('iteration', fontweight='bold')
        ax.set_ylabel(lab, fontweight='bold')
        ax.set_title(lab, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('Stage 3: how do correctly-seeded faint nuclei die?',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'trajectories.png', dpi=150, bbox_inches='tight')
    print(f'\nOutputs -> {OUT}')


if __name__ == '__main__':
    main()

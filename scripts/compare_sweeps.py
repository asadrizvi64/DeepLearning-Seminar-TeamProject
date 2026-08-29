"""Overlay the phantom and real-data k-sweep curves on one figure."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent


def load(path):
    data = json.load(open(path))
    return [d['k'] for d in data], [d['psnr'] for d in data]


def main():
    # Phantom: stitch the two sweeps (1..100 and 100..300)
    pk1, pp1 = load(REPO / 'runs/sweep_k_synthetic/sweep_results.json')
    pk2, pp2 = load(REPO / 'runs/sweep_k_extended/sweep_results.json')
    phantom = {}
    for k, p in zip(pk1 + pk2, pp1 + pp2):
        phantom[k] = p
    pk = sorted(phantom); pp = [phantom[k] for k in pk]

    rk, rp = load(REPO / 'runs/sweep_k_real/sweep_results.json')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pk, pp, 'o-', lw=2, ms=7, color='#1f77b4', label='Synthetic phantom')
    ax.plot(rk, rp, 's-', lw=2, ms=7, color='#d62728', label='Real embryo (Fluo-N3DL-DRO)')

    ax.axvline(50, color='grey', ls=':', alpha=0.6)
    ax.text(52, 17, 'real saturates ~k=50', color='#d62728', fontsize=9)
    ax.axvline(250, color='grey', ls=':', alpha=0.6)
    ax.text(180, 34, 'phantom saturates ~k=250', color='#1f77b4', fontsize=9)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Gaussians (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('k-sweep: synthetic phantom vs real embryo (same 64x128x128 crop)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='center right')

    out = REPO / 'runs/sweep_k_real/phantom_vs_real.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')

    # quick numeric summary
    print('\n         k :  phantom | real')
    for k in [1, 10, 50, 100, 150, 200, 250, 300]:
        pv = phantom.get(k, float('nan'))
        rv = dict(zip(rk, rp)).get(k, float('nan'))
        print(f'  {k:8d} :  {pv:6.2f}  | {rv:6.2f}')


if __name__ == '__main__':
    main()

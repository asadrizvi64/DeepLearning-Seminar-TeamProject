"""Figures for the seminar deck (presentation/build_deck.js). Real data and our own results
only -- no generated illustrations. Minimal text on every figure; the slide and the speaker
notes carry the explanation.

    C:/Users/HP/cpenv/Scripts/python.exe presentation/make_figures.py
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
import tifffile
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'presentation' / 'figs'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'Calibri', 'font.size': 13})


def norm(img, lo=1, hi=99.7):
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img - a) / (b - a + 1e-9), 0, 1)


def scalebar(ax, um, px_um, img_w, label):
    w = um / px_um
    ax.plot([img_w * 0.95 - w, img_w * 0.95], [img_w * 0.0 + 0, 0], alpha=0)  # keep limits
    y = ax.get_ylim()[0] * 0.93
    ax.plot([img_w * 0.95 - w, img_w * 0.95], [y, y], color='white', lw=4, solid_capstyle='butt')
    ax.text(img_w * 0.95 - w / 2, y * 0.965, label, color='white', ha='center', va='bottom', fontsize=12)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches='tight', pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    print('wrote', name)


# ---- 1. the three real datasets (maximum projections, grey, scale bars)
spec = importlib.util.spec_from_file_location('rd', REPO / 'scripts/fidelity/rate_detectability.py')
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)
trib = np.load(REPO / 'runs' / 'colleague_full_volume.npy', mmap_mode='r')
dro = tifffile.imread(r'D:/Download/train/Fluo-N3DL-DRO (1)/Fluo-N3DL-DRO/01/t000.tif')
ds = rd.DATASETS['CE']
ce, _, _ = rd.load_crop(ds['root'], 150, [0, 0, 0], None, '01')
panels = [(np.asarray(trib).max(0), 0.69, 50, 'Tribolium'), (dro.max(0), 0.406, 50, 'Drosophila'),
          (ce.max(0), 0.09, 10, 'C. elegans')]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor='black')
for ax, (img, px, bar, name) in zip(axes, panels):
    ax.imshow(norm(img.astype(np.float32)), cmap='gray')
    ax.set_axis_off()
    h, w = img.shape
    ax.plot([w * 0.93 - bar / px, w * 0.93], [h * 0.93, h * 0.93], color='white', lw=4, solid_capstyle='butt')
    ax.text(w * 0.93 - bar / px / 2, h * 0.89, f'{bar} µm', color='white', ha='center', fontsize=13)
    ax.set_title(name, color='white', fontsize=17)
save(fig, 'data_three.png')

# ---- 2. phantom: ground truth vs fit (crop our own comparison figure, XY view only)
im = Image.open(REPO / 'runs' / 'p1_phantom' / 'compare' / 'compare_mip.png').convert('RGB')
W, H = im.size
sx, sy = W / 1418, H / 1186                     # coordinates measured on the 1418x1186 render
gt = im.crop((int(72 * sx), int(101 * sy), int(408 * sx), int(437 * sy)))
rc = im.crop((int(535 * sx), int(101 * sy), int(871 * sx), int(437 * sy)))
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, (img, t) in zip(axes, ((gt, 'Phantom (known answer)'), (rc, 'Fitted blobs'))):
    ax.imshow(img)
    ax.set_title(t, fontsize=26)
    ax.set_axis_off()
save(fig, 'phantom.png')

# ---- 3. real Tribolium crop vs Gaussian fit, and the size cap (found = green, missed = red)
tgt = np.load(REPO / 'runs/real_proof/target.npy')
rec = np.load(REPO / 'runs/real_proof/recon.npy')
recc = np.load(REPO / 'runs/real_proof/recon_cap.npy')
nuc = np.load(REPO / 'runs/real_proof/target_nuclei.npy')
fnd = np.load(REPO / 'runs/real_proof/found.npy')
fndc = np.load(REPO / 'runs/real_proof/found_cap.npy')


def panel(ax, vol, title, found=None):
    ax.imshow(norm(vol.max(0), 1, 99.5), cmap='magma')
    ax.set_title(title, fontsize=26)
    ax.set_axis_off()
    if found is not None:
        for (z, y, x), f in zip(nuc, found):
            ax.add_patch(plt.Circle((x, y), 6, fill=False, lw=2.2, color='#3ddc84' if f else '#ff4d4d'))


fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
panel(axes[0], tgt, 'Real data')
panel(axes[1], rec, '250 fitted blobs')
save(fig, 'real_vs_fit.png')

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
panel(axes[0], rec, f'No cap: {int(fnd.sum())}/{len(fnd)} found', fnd)
panel(axes[1], recc, f'Size cap: {int(fndc.sum())}/{len(fndc)} found', fndc)
save(fig, 'size_cap.png')

# ---- 4. the budget cliff, deck wording: LoG nuclei kept vs blobs per hand-marked nucleus
import pandas as pd
res = pd.read_csv(REPO / 'runs' / 'fidelity' / 'results_all.csv')
lg = res[(res.detector == 'log') & (res.family == 'luxar') & res.role.isin(['main', 'a1', 'cliff'])]
fig, ax = plt.subplots(figsize=(8, 4.6))
for frame, g in lg.groupby('frame'):
    g = g.sort_values('splats_per_nucleus')
    col = '#2A78D6' if frame.startswith('E1') else '#D64545'
    ax.plot(g.splats_per_nucleus, g['kept@r0.6'], color=col, lw=1.8, marker='o', ms=4, alpha=0.85,
            ls='--' if frame == 'E1 t100' else '-')
ax.axhline(1.0, color='#6B7280', lw=1, ls=':')
ax.axhline(0.9, color='#6B7280', lw=1)
ax.set_xscale('log')
ax.set_xticks([1, 3, 10, 30, 100]); ax.set_xticklabels(['1', '3', '10', '30', '100'])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylim(0.5, 1.12)
ax.set_xlabel('blobs per hand-marked nucleus', fontsize=18)
ax.set_ylabel('nuclei kept (LoG)', fontsize=18)
ax.tick_params(labelsize=15)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.plot([], [], color='#2A78D6', lw=2, label='embryo 1'); ax.plot([], [], color='#D64545', lw=2, label='embryo 2')
ax.legend(frameon=False, fontsize=15, loc='lower right')
ax.text(0.34, 0.905, '90%', color='#6B7280', fontsize=13, va='bottom')
save(fig, 'cliff_deck.png')

# ---- 4b. same region, four formats at about the same size (text added on the slide)
spec2 = importlib.util.spec_from_file_location('rd2', REPO / 'scripts/fidelity/rate_detectability.py')
F = REPO / 'runs' / 'fidelity'
raw194, lo194, hi194 = rd.load_crop(ds['root'], 194, [0, 0, 0], None, '01')
gt194 = rd.gt_from_tra(ds['root'], 194, [0, 0, 0], None, '01')
cp = res[(res.detector == 'cellpose') & (res.tag == 'ce_t194')]
lux = cp[cp.method == 'luxar_K16000'].iloc[0]
j2 = cp[cp.family == 'jpeg2k'].iloc[(cp[cp.family == 'jpeg2k'].bytes - lux.bytes).abs().argsort().iloc[0]]
jx = cp[cp.family == 'jpegxl'].iloc[(cp[cp.family == 'jpegxl'].ratio - 85).abs().argsort().iloc[0]]
vols = [('raw', rd.normalise(raw194, lo194, hi194), None),
        ('jpegxl', np.load(F / 'codecs_ce_t194' / 'recon' / jx.file)['rec'].astype(np.float32), jx),
        ('jpeg2k', np.load(F / 'codecs_ce_t194' / 'recon' / j2.file)['rec'].astype(np.float32), j2),
        ('luxar', np.load(F / 'luxar_render_ce_t194' / 'recon' / lux.file)['rec'].astype(np.float32), lux)]
z = int(np.median(np.round(gt194[:, 0])))
y0, x0, h, w = 120, 180, 260, 340
labels = {}
for name, vol, row in vols:
    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    ax.imshow(vol[max(0, z - 1):z + 2].max(axis=0)[y0:y0 + h, x0:x0 + w], cmap='gray', vmin=0, vmax=1)
    ax.set_axis_off()
    save(fig, f'four_{name}.png')
    if row is not None:
        labels[name] = (round(float(row.ratio)), round(100 * float(row['kept@r0.6'])))
import json
json.dump(labels, open(OUT / 'four_labels.json', 'w'))
print('four-way labels', labels)

# ---- 5. title image: the Tribolium embryo alone, no text
fig, ax = plt.subplots(figsize=(4, 7.6), facecolor='black')
ax.imshow(norm(np.asarray(trib).max(0).astype(np.float32)), cmap='gray')
ax.set_axis_off()
save(fig, 'title_tribolium.png')

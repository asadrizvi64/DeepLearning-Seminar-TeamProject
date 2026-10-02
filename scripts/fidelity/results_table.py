"""Study A results in ONE long table: runs/fidelity/results_all.csv.

One row per (frame, detector, reconstruction). Every number in the paper can be read off
this table; paper_numbers.py, check_rules.py and a1_crossings.py are views of the same
per-frame files it is built from.

Columns
  tag, frame (E1 t150 ...), embryo, seq, t, role (main | cliff | a1), n_labels
  detector  log | cellpose | watershed | cellpose3d
  method, family (luxar | jpegxl | jpeg2k | zfp | downsample | raw), file, bytes, ratio
  psnr      vs raw, from the run that made the reconstruction
  seeds, splats, splats_per_nucleus   (Luxar rows)
  found@r{f}, kept@r{f}   f = 0.4, 0.6, 0.8 x nucleus diameter; kept = found / found(raw)
                          with the same detector on the same frame
  kept_faint@r0.6, kept_bright@r0.6, precision@r0.6 (segmenters only)

Frames and run folders are listed in FRAMES; a frame or detector whose files are missing
is skipped, so the table grows as results arrive.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
F = REPO / 'runs' / 'fidelity'
C1 = REPO / 'runs' / 'fidelity_cluster' / 'fidelity'
C2 = REPO / 'runs' / 'fidelity_cluster2' / 'fidelity'
CA1 = REPO / 'runs' / 'fidelity_a1' / 'fidelity'          # A1 Luxar fits, when unpacked
DETECTORS = ('log', 'cellpose', 'watershed', 'cellpose3d')
RADII = (0.4, 0.6, 0.8)

# tag -> (seq, t, role, Luxar run folders)
FRAMES = {
    'ce_t150': ('01', 150, 'main', [C1 / 'pilot_ce_t150']),
    'ce_t194': ('01', 194, 'main', [C1 / 'pilot_ce_t194']),
    'ce_s02_t150': ('02', 150, 'main', [C2 / 'pilot_ce_s02_t150', C2 / 'pilot_ce_s02_t150_hi']),
    'ce_s02_t180': ('02', 180, 'main', [C2 / 'pilot_ce_s02_t180', C2 / 'pilot_ce_s02_t180_hi']),
    'ce_s01_t100': ('01', 100, 'cliff', [C2 / 'pilot_ce_s01_t100']),
}
for _seq, _ts in (('01', (110, 130, 170, 185)), ('02', (110, 130, 165, 185))):
    for _t in _ts:
        FRAMES[f'ce_s{_seq}_t{_t}'] = (_seq, _t, 'a1', [CA1 / f'pilot_ce_s{_seq}_t{_t}'])


def run_rows(tag):
    """PSNR of every reconstruction and the Luxar fit facts, from the runs that made them."""
    seq, t, role, lux_runs = FRAMES[tag]
    ps, lux, n_labels = [], [], None
    for d in [F / f'codecs_{tag}'] + lux_runs:
        csv = d / 'rate_detectability.csv'
        if not csv.exists():
            continue
        n_labels = json.load(open(d / 'meta.json'))['n_manual']
        r = pd.read_csv(csv)
        ps.append(r[['method', 'bytes', 'psnr']])
        for _, x in r[r.method.str.startswith('luxar')].iterrows():
            p = json.loads(x.params)
            lux.append(dict(method=x.method, seeds=p['seeds'], splats=p['n_splats'], psnr_lux=x.psnr))
    psnr = pd.concat(ps).drop_duplicates(['method', 'bytes']) if ps else pd.DataFrame(columns=['method', 'bytes', 'psnr'])
    lux = pd.DataFrame(lux).drop_duplicates('method') if lux else pd.DataFrame(columns=['method', 'seeds', 'splats', 'psnr_lux'])
    return n_labels, psnr, lux


def frame_table(tag):
    seq, t, role, _ = FRAMES[tag]
    n_labels, psnr, lux = run_rows(tag)
    out = []
    for det in DETECTORS:
        csv = F / f'{det}_{tag}.csv'
        if not csv.exists():
            continue
        d = pd.read_csv(csv)
        raw = d[d.method == 'raw'].iloc[0]
        row = pd.DataFrame(dict(tag=tag, frame=f"E{int(seq)} t{t}", embryo=int(seq), seq=seq, t=t,
                                role=role, n_labels=n_labels, detector=det, method=d.method,
                                family=d.method.str.replace(r'_K\d+$', '', regex=True),
                                file=d.file, bytes=d.bytes, ratio=d.ratio))
        for f in RADII:
            row[f'found@r{f}'] = d[f'found@r{f}']
            row[f'kept@r{f}'] = d[f'found@r{f}'] / raw[f'found@r{f}']
        row['kept_faint@r0.6'] = d['found_faint@r0.6'] / raw['found_faint@r0.6']
        row['kept_bright@r0.6'] = d['found_bright@r0.6'] / raw['found_bright@r0.6']
        row['precision@r0.6'] = d['precision@r0.6'] if 'precision@r0.6' in d else np.nan
        out.append(row)
    if not out:
        return None
    df = pd.concat(out, ignore_index=True)
    df = df.merge(psnr, on=['method', 'bytes'], how='left')
    df = df.merge(lux, on='method', how='left')
    # Luxar renders are re-measured locally, so their bytes can differ by a few from the
    # cluster's; take the fit's own PSNR for those rows.
    df['psnr'] = df.psnr.fillna(df.pop('psnr_lux'))
    df['splats_per_nucleus'] = df.splats / df.n_labels
    return df


def build():
    parts = [frame_table(tag) for tag in FRAMES]
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    cols = ['tag', 'frame', 'embryo', 'seq', 't', 'role', 'n_labels', 'detector', 'method', 'family',
            'file', 'bytes', 'ratio', 'psnr', 'seeds', 'splats', 'splats_per_nucleus']
    df = df[cols + [c for c in df.columns if c not in cols]]
    return df.sort_values(['role', 'embryo', 't', 'detector', 'family', 'ratio']).reset_index(drop=True)


def main():
    df = build()
    df.to_csv(F / 'results_all.csv', index=False, float_format='%.6g')
    print(f'{len(df)} rows -> {F / "results_all.csv"}')
    print(df.groupby(['role', 'frame', 'detector']).size().unstack(fill_value=0).to_string())


if __name__ == '__main__':
    main()

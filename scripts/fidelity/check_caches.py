"""Fail if any cached detection file is unreadable, non-finite or mostly all-zero.

An interrupted run on 2026-10-03 left one Cellpose centroid file of 129 rows of (0, 0, 0),
which scored a Luxar reconstruction as 0% of nuclei kept. The scorers now write caches
atomically; this check runs before every analysis (scripts/fidelity/reproduce.sh).
"""
import glob
import sys

import numpy as np

files = [f for f in glob.glob('runs/fidelity/*_centroids/*.npy') + glob.glob('runs/fidelity/*_candidates/*.npy')
         if 'superseded' not in f and not f.endswith('.tmp.npy')]
bad = []
for f in files:
    try:
        a = np.load(f)
        if a.size and (not np.isfinite(a).all() or (np.abs(a).sum(axis=1) == 0).mean() > 0.5):
            bad.append(f'{f}: all-zero or non-finite rows')
    except Exception as e:                       # noqa: BLE001 -- any load failure is corruption
        bad.append(f'{f}: unreadable ({e})')
print(f'{len(files)} cache files checked, {len(bad)} bad')
if bad:
    print('\n'.join(bad))
    sys.exit('corrupt detection caches -- delete them and rerun the scorer')

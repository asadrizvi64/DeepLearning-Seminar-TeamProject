"""Turn apr_encode.py outputs into a fidelity-pipeline directory: meta.json copied from the
codec directory of the same frame, and recon/<name>_<bytes>.npz normalised exactly like the
codec reconstructions (rate_detectability.normalise with the raw volume's lo/hi). The
directory can then be scored by log_rescore.py, watershed_score.py and cellpose_score.py.

    python scripts/fidelity/apr_to_recon.py apr_out runs/fidelity/codecs_ce_t194 \
        runs/fidelity/apr_ce_t194
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rd', HERE / 'rate_detectability.py')
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)

src, codec_dir, out = (Path(a) for a in sys.argv[1:4])
meta = json.load(open(codec_dir / 'meta.json'))
ds = rd.DATASETS[meta['dataset']]
rd.configure(ds['voxel'], ds['nucleus_um'])
shape = None if meta['origin'] == [0, 0, 0] else meta['shape']
crop, lo, hi = rd.load_crop(ds['root'], meta['frame'], meta['origin'], shape, meta.get('seq', '01'))

(out / 'recon').mkdir(parents=True, exist_ok=True)
shutil.copy(codec_dir / 'meta.json', out / 'meta.json')
shutil.copy(src / 'apr_runs.json', out / f'apr_runs_{src.name}.json')
for f in sorted(src.glob('*_*.npy')):
    rec = np.load(f)
    assert rec.shape == crop.shape, (f, rec.shape)
    rn = rd.normalise(rec.astype(np.float32), lo, hi)
    np.savez_compressed(out / 'recon' / f'{f.stem}.npz', rec=rn.astype(np.float16))
    print(f'{f.stem}: ratio {crop.nbytes / int(f.stem.rsplit("_", 1)[1]):.1f}x')

"""Lund Tribolium dataset utilities (Zenodo 5837363).

Vorkel, Haase, Myers (2022). "Tribolium castaneum embryo development as seen with
lightsheet microscopy ('Lund' sample, first 36h)."
doi:10.5281/zenodo.5837363

Sample line provided by Akanksha Jain (Tomančák lab, MPI-CBG Dresden). Acquired with
ClearControl on a light-sheet microscope, exported with the beetlesafari library.

Properties
----------
    Voxel size : 0.6934 x 0.6934 x 3.0 microns (xy x z; ~4.33x anisotropic)
    Frame delay: ~5 minutes between consecutive `i` indices (some are duplicates)
    Duration   : 36 h (~177 timepoints)
    Bit depth  : 16-bit (uint16)
    Layout     : flat directory of single-volume TIFFs; no scaffold, no GT

File naming
-----------
    lund_i{export_idx}_oi_{original_idx}.tif

    export_idx   - 0-based sequential index in the published timelapse
    original_idx - microscope acquisition number (not consecutive; duplicates exist
                   where the microscope produced one stack per 10-min interval)
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


# Voxel size in microns (x, y, z), per the Zenodo metadata.
VOXEL_SIZE_LUND_UM = (0.6934, 0.6934, 3.0)

# Nominal frame delay between consecutive export indices (seconds).
FRAME_DELAY_LUND_SEC = 300

_FILENAME_RE = re.compile(r'lund_i(\d{6})_oi_(\d{6})\.tif')


@dataclass
class LundFrame:
    """Metadata for one timepoint of the Lund Tribolium timelapse."""
    export_index: int          # 0..N-1, sequential
    original_index: int        # microscope acquisition number
    path: Path

    @classmethod
    def from_path(cls, p) -> 'LundFrame':
        p = Path(p)
        m = _FILENAME_RE.match(p.name)
        if not m:
            raise ValueError(f"Not a Lund frame filename: {p.name}")
        return cls(
            export_index=int(m.group(1)),
            original_index=int(m.group(2)),
            path=p,
        )

    @property
    def acquisition_time_sec(self) -> float:
        """Approximate seconds since start of acquisition."""
        return self.export_index * FRAME_DELAY_LUND_SEC


def list_lund_frames(root) -> list[LundFrame]:
    """List all Lund frames in a directory, sorted by export index."""
    root = Path(root)
    frames = []
    for p in root.glob('lund_i*_oi_*.tif'):
        try:
            frames.append(LundFrame.from_path(p))
        except ValueError:
            continue
    return sorted(frames, key=lambda f: f.export_index)


def load_lund_volume(
    path,
    normalize: bool = True,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
) -> np.ndarray:
    """Load one Lund timepoint as (D, H, W) float32.

    Optionally percentile-normalizes to [0, 1] (default 0.5 / 99.5, robust to outliers).
    Reuses `volsplat.ctc.percentile_normalize`.
    """
    vol = tifffile.imread(str(path))
    # Some ClearControl exports have a leading singleton axis (channels, time, etc).
    while vol.ndim > 3 and vol.shape[0] == 1:
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    vol = vol.astype(np.float32)
    if normalize:
        from .ctc import percentile_normalize
        vol = percentile_normalize(vol, low_percentile, high_percentile)
    return vol

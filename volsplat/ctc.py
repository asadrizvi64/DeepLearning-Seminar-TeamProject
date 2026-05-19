"""Cell Tracking Challenge dataset utilities.

Specifically targets the Fluo-N3DL-DRO (developing Drosophila) dataset from the Keller
lab (Janelia), but the loader is generic across CTC 3D datasets.

DRO acquisition parameters
--------------------------
    Voxel size : 0.406 x 0.406 x 2.03 microns  (xy x z; ~5x anisotropic along z)
    Time step  : 30 seconds
    Microscope : SIMView light-sheet
    Reference  : Tomer et al., Nature Methods 2014 (doi:10.1038/nmeth.3036)

Standard CTC layout
-------------------
    root/
    |-- 01/                   sequence 01 (training)
    |   |-- t000.tif          3D TIFF stacks, one per timepoint (16-bit)
    |   |-- t001.tif
    |   `-- ...
    |-- 01_GT/
    |   |-- TRA/              tracking ground truth (one label map per frame)
    |   |   |-- man_track000.tif    label = track ID per voxel
    |   |   |-- ...
    |   |   `-- man_track.txt       track metadata: "L B E P" per line
    |   `-- SEG/              segmentation GT (sparse - not every frame)
    |       `-- man_seg*.tif
    `-- 02/, 02_GT/           sequence 02
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile


# Voxel size in microns (x, y, z) for Fluo-N3DL-DRO.
VOXEL_SIZE_DRO_UM = (0.406, 0.406, 2.03)


# ---------------------------------------------------------------- track metadata

@dataclass
class TrackInfo:
    """One row of a CTC man_track.txt file."""
    track_id: int
    begin_frame: int
    end_frame: int
    parent_id: int  # 0 if no parent (i.e. not the product of a division)


def parse_man_track(path) -> dict[int, TrackInfo]:
    """Parse a CTC man_track.txt file. Each line is `L B E P`."""
    tracks = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            L, B, E, P = map(int, parts)
            tracks[L] = TrackInfo(L, B, E, P)
    return tracks


# --------------------------------------------------------------- volume loading

def _load_3d_tiff(path: Path) -> np.ndarray:
    """Read a TIFF and coerce to a 3D (D, H, W) array, squeezing singleton axes."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected CTC file not found: {path}\n"
            f"Check --root points at the directory containing the sequence folders "
            f"(e.g. the one holding '01/', '01_GT/')."
        )
    vol = tifffile.imread(str(path))
    vol = np.squeeze(vol)  # drop singleton channel/time axes (OME-TIFF)
    if vol.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume from {path.name} but got shape {vol.shape} after "
            f"squeezing singleton axes. Multi-channel CTC data is unexpected here; "
            f"select a channel before loading if so."
        )
    return vol


def load_ctc_frame(
    root,
    sequence: str = '01',
    frame: int = 0,
    normalize: bool = True,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
) -> np.ndarray:
    """Load one 3D frame as (D, H, W) float32. Optionally percentile-normalize to [0, 1]."""
    path = Path(root) / sequence / f't{frame:03d}.tif'
    vol = _load_3d_tiff(path).astype(np.float32)
    if normalize:
        vol = percentile_normalize(vol, low_percentile, high_percentile)
    return vol


def load_tracking_labels(root, sequence: str = '01', frame: int = 0) -> Optional[np.ndarray]:
    """Load the per-frame tracking label map (track ID per voxel). Returns None if missing."""
    path = Path(root) / f'{sequence}_GT' / 'TRA' / f'man_track{frame:03d}.tif'
    if not path.exists():
        return None
    return np.squeeze(tifffile.imread(str(path)))


def load_segmentation(root, sequence: str = '01', frame: int = 0) -> Optional[np.ndarray]:
    """Load segmentation GT for a frame. Returns None if missing.

    Fluo-N3DL-DRO ships *sparse* SEG annotations named `man_seg_TTT_SSS.tif`
    (TTT = timepoint, SSS = annotated slice), as well as the full-volume form
    `man_segTTT.tif` used by other CTC sets. We try the full-volume name first,
    then fall back to the first sparse file for this timepoint.
    """
    seg_dir = Path(root) / f'{sequence}_GT' / 'SEG'
    full = seg_dir / f'man_seg{frame:03d}.tif'
    if full.exists():
        return np.squeeze(tifffile.imread(str(full)))
    sparse = sorted(seg_dir.glob(f'man_seg_{frame:03d}_*.tif'))
    if sparse:
        return np.squeeze(tifffile.imread(str(sparse[0])))
    return None


def list_frames(root, sequence: str = '01') -> list[int]:
    """Return sorted frame indices available in a sequence directory."""
    seq_dir = Path(root) / sequence
    frames = []
    for p in seq_dir.glob('t*.tif'):
        try:
            frames.append(int(p.stem[1:]))
        except ValueError:
            continue
    return sorted(frames)


# ---------------------------------------------------------- normalization / crop

def percentile_normalize(
    volume: np.ndarray,
    low_p: float = 0.5,
    high_p: float = 99.5,
    clip: bool = True,
) -> np.ndarray:
    """Normalize a volume to [0, 1] using percentile-based clipping.

    Robust to outliers and the long-tailed intensity distributions typical of
    fluorescence microscopy. Min-max normalization would map a handful of hot pixels
    to 1.0 and squash everything else into a narrow band near zero.
    """
    lo = float(np.percentile(volume, low_p))
    hi = float(np.percentile(volume, high_p))
    if hi > lo:
        out = (volume.astype(np.float32) - lo) / (hi - lo)
    else:
        out = np.zeros_like(volume, dtype=np.float32)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def find_intensity_bbox(
    volume: np.ndarray,
    threshold_percentile: float = 95.0,
    padding: int = 16,
) -> tuple[slice, slice, slice]:
    """Axis-aligned bounding box around the high-intensity content of a volume."""
    D, H, W = volume.shape
    thresh = float(np.percentile(volume, threshold_percentile))
    mask = volume > thresh
    if not mask.any():
        return slice(0, D), slice(0, H), slice(0, W)

    z_mask = mask.any(axis=(1, 2))
    y_mask = mask.any(axis=(0, 2))
    x_mask = mask.any(axis=(0, 1))
    z_idx = np.where(z_mask)[0]; y_idx = np.where(y_mask)[0]; x_idx = np.where(x_mask)[0]
    z_lo, z_hi = int(z_idx[0]), int(z_idx[-1])
    y_lo, y_hi = int(y_idx[0]), int(y_idx[-1])
    x_lo, x_hi = int(x_idx[0]), int(x_idx[-1])

    z_lo, z_hi = max(0, z_lo - padding), min(D, z_hi + padding + 1)
    y_lo, y_hi = max(0, y_lo - padding), min(H, y_hi + padding + 1)
    x_lo, x_hi = max(0, x_lo - padding), min(W, x_hi + padding + 1)
    return slice(z_lo, z_hi), slice(y_lo, y_hi), slice(x_lo, x_hi)


def center_roi(
    bbox: tuple[slice, slice, slice],
    target_shape: tuple[int, int, int],
    volume_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    """Recenter a bbox to a target shape, clipped to volume bounds."""
    def _one(s: slice, t: int, max_v: int) -> slice:
        center = (s.start + s.stop) // 2
        lo = max(0, center - t // 2)
        hi = min(max_v, lo + t)
        lo = max(0, hi - t)  # adjust if we hit the upper bound
        return slice(lo, hi)

    D, H, W = volume_shape
    z, y, x = bbox
    tz, ty, tx = target_shape
    return _one(z, tz, D), _one(y, ty, H), _one(x, tx, W)


# -------------------------------------------------------------- track centroids

def cell_centroids_from_labels(label_map: np.ndarray) -> dict[int, np.ndarray]:
    """Centroid (in voxel coords, (x, y, z) order) of each labeled cell.

    Useful for initializing Gaussians at known cell positions in the temporal
    variants (shared-identity per-frame fitting).
    """
    from scipy.ndimage import center_of_mass
    labels = np.unique(label_map)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return {}
    coms = center_of_mass(label_map > 0, labels=label_map, index=labels)
    # scipy returns (z, y, x); we standardize to (x, y, z)
    out = {}
    for lid, com in zip(labels, coms):
        z, y, x = com
        out[int(lid)] = np.array([x, y, z], dtype=np.float32)
    return out

"""Export a fitted GaussianSet to browser-viewable splat formats (P4 / E6).

Two formats, both readable by the common web viewers:
  * .ply   - the INRIA 3DGS PLY layout (SuperSplat, antimatter15/splat, gsplat.js).
  * .splat - the compact antimatter15 binary (32 bytes/Gaussian).

Microscopy has no colour, so we map the (scalar) density amplitude to BOTH a
grayscale colour and the opacity, so brighter nuclei show up brighter and more
opaque in the viewer. Geometry (position, anisotropic scale, rotation) is exact.

Conventions: positions are (x, y, z) voxel coords; quaternions are (w, x, y, z);
log_scales are natural-log voxel scales; amplitude = softplus(amp_logit).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Spherical-harmonics DC factor used by 3DGS (C0 = 0.5/sqrt(pi)).
_SH_C0 = 0.28209479177387814


def _checkpoint_arrays(ckpt: dict):
    """Pull (positions, log_scales, quats, amplitudes) as numpy from a checkpoint."""
    pos = ckpt['positions'].detach().cpu().float().numpy()
    log_scales = ckpt['log_scales'].detach().cpu().float().numpy()
    quats = ckpt['quaternions'].detach().cpu().float()
    quats = F.normalize(quats, dim=-1).numpy()                  # (G, 4) w,x,y,z
    amp = F.softplus(ckpt['amp_logits'].detach().cpu().float()).numpy()  # (G,)
    return pos, log_scales, quats, amp


def _amp_to_opacity(amp: np.ndarray, percentile: float = 99.0):
    """Normalize amplitude to a display value in (0, 1) robust to a few hot blobs."""
    hi = float(np.percentile(amp, percentile))
    if hi <= 0:
        hi = float(amp.max()) or 1.0
    return np.clip(amp / hi, 1e-3, 1.0)


def _inverse_sigmoid(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1.0 - x))


def to_ply(ckpt: dict, path, opacity_gamma: float = 1.0) -> int:
    """Write an INRIA-3DGS binary PLY. Returns the number of Gaussians written."""
    pos, log_scales, quats, amp = _checkpoint_arrays(ckpt)
    G = pos.shape[0]

    disp = _amp_to_opacity(amp) ** opacity_gamma                # (G,) in (0,1]
    # Grayscale colour from the same display value, stored as SH DC term.
    f_dc = (disp[:, None] - 0.5) / _SH_C0                       # (G, 3) after broadcast
    f_dc = np.repeat(f_dc, 3, axis=1)
    opacity = _inverse_sigmoid(disp)[:, None]                  # viewer applies sigmoid
    normals = np.zeros((G, 3), dtype=np.float32)

    # Column order must match the header below exactly.
    verts = np.concatenate([
        pos.astype(np.float32), normals,
        f_dc.astype(np.float32), opacity.astype(np.float32),
        log_scales.astype(np.float32), quats.astype(np.float32),
    ], axis=1).astype('<f4')                                    # (G, 17), little-endian

    props = (['x', 'y', 'z', 'nx', 'ny', 'nz',
              'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
              'scale_0', 'scale_1', 'scale_2',
              'rot_0', 'rot_1', 'rot_2', 'rot_3'])
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {G}\n"
    header += "".join(f"property float {p}\n" for p in props)
    header += "end_header\n"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(verts.tobytes())
    return G


def to_splat(ckpt: dict, path, opacity_gamma: float = 1.0) -> int:
    """Write the antimatter15 .splat binary (32 bytes/Gaussian). Returns count.

    Layout per splat: position[3] f32, scale[3] f32, rgba[4] u8, rot[4] u8.
    Sorted by descending display value so the viewer's front splats are the bright
    ones (the format has no depth sort of its own at load).
    """
    pos, log_scales, quats, amp = _checkpoint_arrays(ckpt)
    scales = np.exp(log_scales)                                 # actual voxel scales
    disp = _amp_to_opacity(amp) ** opacity_gamma

    order = np.argsort(-disp)
    pos, scales, quats, disp = pos[order], scales[order], quats[order], disp[order]

    rgba = np.empty((pos.shape[0], 4), dtype=np.uint8)
    rgba[:, :3] = np.clip(disp[:, None] * 255, 0, 255).astype(np.uint8)  # gray
    rgba[:, 3] = np.clip(disp * 255, 0, 255).astype(np.uint8)
    # Quaternion bytes: (q*128 + 128), order w,x,y,z (matches the PLY rot_0..3).
    rot_b = np.clip(quats * 128 + 128, 0, 255).astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        for i in range(pos.shape[0]):
            f.write(struct.pack('<3f', *pos[i]))
            f.write(struct.pack('<3f', *scales[i]))
            f.write(rgba[i].tobytes())
            f.write(rot_b[i].tobytes())
    return pos.shape[0]


def read_ply(path):
    """Read back an INRIA-3DGS PLY into a dict of arrays (for round-trip checks)."""
    path = Path(path)
    with open(path, 'rb') as f:
        # Parse ASCII header.
        props = []
        n = 0
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property float'):
                props.append(line.split()[-1])
            elif line == 'end_header':
                break
        data = np.frombuffer(f.read(n * len(props) * 4), dtype='<f4').reshape(n, len(props))
    cols = {p: data[:, i] for i, p in enumerate(props)}
    return cols, n

from __future__ import annotations

import numpy as np


def finite_xyz_array(points: np.ndarray) -> np.ndarray:
    """Return a dense float XYZ matrix after rejecting non-finite observations."""
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Point cloud must be an N x 3 XYZ array")
    return xyz[np.isfinite(xyz).all(axis=1)]


def crop_local_patch(points: np.ndarray, center_xyz: np.ndarray, side_m: float) -> np.ndarray:
    """Extract the selected square support volume used for local PCA fitting."""
    if side_m <= 0.0:
        raise ValueError("side_m must be positive")
    center = np.asarray(center_xyz, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("Patch center must be a 3-vector")

    half_side = side_m / 2.0
    offset = np.abs(points - center)
    patch = points[np.all(offset <= half_side, axis=1)]
    if patch.shape[0] < 3:
        raise ValueError("Selected patch does not contain enough points for PCA")
    return patch

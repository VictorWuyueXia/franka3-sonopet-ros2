from __future__ import annotations

import numpy as np

Point3 = tuple[float, float, float]


def crop_by_distance(points: list[Point3], max_distance_m: float) -> list[Point3]:
    """Keep points inside a sphere around the local origin."""
    if max_distance_m <= 0.0:
        raise ValueError("max_distance_m must be positive")
    max_dist_sq = max_distance_m * max_distance_m
    return [point for point in points if sum(value * value for value in point) <= max_dist_sq]


def trim_outliers(points: list[Point3], trim_fraction: float) -> list[Point3]:
    """Trim farthest points by distance; placeholder for the v1 cloud policy."""
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0.0, 0.5)")
    keep_count = int(round(len(points) * (1.0 - trim_fraction)))
    ordered = sorted(points, key=lambda point: sum(value * value for value in point))
    return ordered[:keep_count]


def finite_xyz_array(points: np.ndarray) -> np.ndarray:
    """Return a dense float XYZ matrix after rejecting non-finite observations."""
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Point cloud must be an N x 3 XYZ array")
    return xyz[np.isfinite(xyz).all(axis=1)]


def filter_planning_cloud(
    points: np.ndarray,
    max_distance_m: float = 0.5,
    trim_fraction: float = 0.2,
) -> np.ndarray:
    """Apply the frozen D405 local-distance and farthest-outlier cloud policy."""
    if max_distance_m <= 0.0:
        raise ValueError("max_distance_m must be positive")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0.0, 0.5)")

    xyz = finite_xyz_array(points)
    squared_radius = np.einsum("ij,ij->i", xyz, xyz)
    cropped = xyz[squared_radius <= max_distance_m * max_distance_m]
    if cropped.size == 0:
        raise ValueError("Point cloud has no points inside the planning radius")

    keep_count = int(round(cropped.shape[0] * (1.0 - trim_fraction)))
    if keep_count <= 0:
        raise ValueError("Point cloud trimming removed all planning points")

    cropped_radius = np.einsum("ij,ij->i", cropped, cropped)
    keep_indices = np.argpartition(cropped_radius, keep_count - 1)[:keep_count]
    return cropped[keep_indices]


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

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from fr3_sonopet_trajectory.cloud_io import (
    crop_local_patch,
    filter_planning_cloud,
    finite_xyz_array,
)
from fr3_sonopet_trajectory.raster_pattern import RasterSpec, build_surface_raster
from fr3_sonopet_trajectory.surface_geometry import (
    SurfaceFrame,
    estimate_local_normals,
    fit_surface_frame,
)


@dataclass(frozen=True)
class RasterBuild:
    center: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    quaternions_xyzw: np.ndarray
    frame: SurfaceFrame
    segment_names: list[str]
    config_hash: str


def build_raster_from_cloud(
    cloud_points: np.ndarray,
    selected_center_base: np.ndarray,
    spec: RasterSpec,
    *,
    pre_filtered: bool = False,
    dig_depth_m: float = 0.0,
) -> RasterBuild:
    """Build a base-frame Cartesian raster patch from a selected D405 cloud point."""
    if dig_depth_m < 0.0:
        raise ValueError("dig_depth_m must be non-negative")
    planning_points = cloud_points if pre_filtered else filter_planning_cloud(cloud_points)
    patch_points = crop_local_patch(planning_points, selected_center_base, spec.square_side_m)
    frame = fit_surface_frame(patch_points, selected_center_base)

    # Raster points follow interpolated surface height; normals remain reference metadata.
    raster = build_surface_raster(patch_points, frame, spec)
    raster_points = raster.points.copy()
    raster_points[:, 2] -= dig_depth_m
    center = frame.center.copy()
    center[2] -= dig_depth_m
    normals = estimate_local_normals(
        patch_points,
        raster.normal_reference_points,
        search_radius_m=spec.square_side_m,
    )
    # Motion planning owns EE attitude; trajectory poses carry positions and reference normals.
    quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    quaternions = np.repeat(quaternion[None, :], raster.points.shape[0], axis=0)
    return RasterBuild(
        center=center,
        points=raster_points,
        normals=normals,
        quaternions_xyzw=quaternions,
        frame=frame,
        segment_names=raster.segment_names,
        config_hash=config_hash(spec, dig_depth_m),
    )


def _resample_raster(
    cloud_points: np.ndarray,
    selected_center_base: np.ndarray,
    spec: RasterSpec,
    *,
    pre_filtered: bool = False,
    dig_depth_m: float = 0.0,
) -> RasterBuild:
    """Reuse selected X-Y, estimate new Z, then rebuild waypoints from the latest cloud."""
    planning_points = cloud_points if pre_filtered else filter_planning_cloud(cloud_points)
    updated_center = np.asarray(selected_center_base, dtype=np.float64).copy()
    search_radius_m = spec.square_side_m / 2.0

    finite_points = finite_xyz_array(planning_points)

    distances = np.linalg.norm(finite_points[:, :2] - updated_center[None, :2], axis=1)
    support_mask = distances <= search_radius_m
    if not bool(np.any(support_mask)):
        raise ValueError("No point-cloud points found near selected center XY")

    # Find the nearest 8 points and use their weighted average to update the center Z
    support_distances = distances[support_mask]
    support_z = finite_points[support_mask, 2]
    nearest_count = min(8, support_z.shape[0])
    nearest = np.argpartition(support_distances, nearest_count - 1)[:nearest_count]
    nearest_distances = support_distances[nearest]
    weights = 1.0 / np.maximum(nearest_distances, 1e-12) ** 2
    updated_center[2] = float(np.sum(weights * support_z[nearest]) / np.sum(weights))

    return build_raster_from_cloud(
        finite_points,
        updated_center,
        spec,
        pre_filtered=True,
        dig_depth_m=dig_depth_m,
    )


def config_hash(spec: RasterSpec, dig_depth_m: float = 0.0) -> str:
    """Hash the numerical raster policy into a compact plan identity string."""
    payload = (
        f"{spec.square_side_m:.9f}|{spec.line_spacing_m:.9f}|"
        f"{spec.downsample_rate}|{spec.pattern}|{dig_depth_m:.9f}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

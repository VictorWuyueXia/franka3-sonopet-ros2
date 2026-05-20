from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from fr3_sonopet_trajectory.cloud_io import crop_local_patch, filter_planning_cloud
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
) -> RasterBuild:
    """Build a base-frame Cartesian raster patch from a selected D405 cloud point."""
    planning_points = cloud_points if pre_filtered else filter_planning_cloud(cloud_points)
    patch_points = crop_local_patch(planning_points, selected_center_base, spec.square_side_m)
    frame = fit_surface_frame(patch_points, selected_center_base)

    # Raster points follow interpolated surface height; normals remain reference metadata.
    raster = build_surface_raster(patch_points, frame, spec)
    normals = estimate_local_normals(
        patch_points,
        raster.normal_reference_points,
        search_radius_m=spec.square_side_m,
    )
    # Motion planning owns EE attitude; trajectory poses carry positions and reference normals.
    quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    quaternions = np.repeat(quaternion[None, :], raster.points.shape[0], axis=0)
    return RasterBuild(
        center=frame.center,
        points=raster.points,
        normals=normals,
        quaternions_xyzw=quaternions,
        frame=frame,
        segment_names=raster.segment_names,
        config_hash=config_hash(spec),
    )


def config_hash(spec: RasterSpec) -> str:
    """Hash the numerical raster policy into a compact plan identity string."""
    payload = (
        f"{spec.square_side_m:.9f}|{spec.line_spacing_m:.9f}|"
        f"{spec.downsample_rate}|{spec.pattern}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

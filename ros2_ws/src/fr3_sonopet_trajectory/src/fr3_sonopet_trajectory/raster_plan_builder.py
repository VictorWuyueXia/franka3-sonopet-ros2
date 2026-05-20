from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from fr3_sonopet_trajectory.cloud_io import crop_local_patch, filter_planning_cloud
from fr3_sonopet_trajectory.raster_pattern import RasterSpec, build_raster_uv
from fr3_sonopet_trajectory.surface_geometry import (
    SurfaceFrame,
    fit_surface_frame,
    quaternion_xyzw_from_axes,
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
    selected_center_cloud: np.ndarray,
    spec: RasterSpec,
    *,
    pre_filtered: bool = False,
) -> RasterBuild:
    """Build a Cartesian raster patch from a selected in-hand D405 cloud point."""
    planning_points = cloud_points if pre_filtered else filter_planning_cloud(cloud_points)
    patch_points = crop_local_patch(planning_points, selected_center_cloud, spec.square_side_m)
    frame = fit_surface_frame(
        patch_points,
        selected_center_cloud,
        camera_origin_xyz=np.zeros(3, dtype=np.float64),
    )

    # The UV raster is lifted into the PCA tangent frame without changing surface height.
    uv = build_raster_uv(spec)
    points = (
        frame.center[None, :]
        + uv[:, 0, None] * frame.tangent[None, :]
        + uv[:, 1, None] * frame.bitangent[None, :]
    )
    normals = np.repeat(frame.normal[None, :], points.shape[0], axis=0)
    quaternion = quaternion_xyzw_from_axes(frame.tangent, frame.normal)
    quaternions = np.repeat(quaternion[None, :], points.shape[0], axis=0)
    return RasterBuild(
        center=frame.center,
        points=points,
        normals=normals,
        quaternions_xyzw=quaternions,
        frame=frame,
        segment_names=["raster"] * points.shape[0],
        config_hash=config_hash(spec),
    )


def config_hash(spec: RasterSpec) -> str:
    """Hash the numerical raster policy into a compact plan identity string."""
    payload = (
        f"{spec.square_side_m:.9f}|{spec.line_spacing_m:.9f}|"
        f"{spec.downsample_rate}|{spec.pattern}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

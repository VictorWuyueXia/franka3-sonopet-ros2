"""Raster pattern builders from selected square point-cloud points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


RASTER_PATTERN_BOUSTROPHEDON = "boustrophedon"
RASTER_PATTERN_UNIDIRECTIONAL_RETRACT = "unidirectional_retract"
SUPPORTED_RASTER_PATTERNS = {
    RASTER_PATTERN_BOUSTROPHEDON,
    RASTER_PATTERN_UNIDIRECTIONAL_RETRACT,
}
INTER_LINE_RETRACT_LIFT_M = 0.005


@dataclass(frozen=True)
class RasterPatternSpec:
    """Parameters shared by all selected-point raster patterns."""

    square_center_xyz: Tuple[float, float, float]
    square_side_len_m: float
    line_spacing_m: float
    pointcloud_downsample_rate: int
    raster_pattern: str


@dataclass(frozen=True)
class LocalSurfaceBasis:
    """Surface tangent basis centered at the selected square center."""

    center_xyz: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    normal_axis: np.ndarray


@dataclass(frozen=True)
class LocalSurfaceSamples:
    """Point-cloud samples represented in local surface coordinates."""

    xyz_points: np.ndarray
    uv_points: np.ndarray
    w_values: np.ndarray
    basis: LocalSurfaceBasis


def validate_raster_pattern_spec(spec: RasterPatternSpec) -> None:
    """Validate raster parameters before touching point-cloud data."""

    if str(spec.raster_pattern) not in SUPPORTED_RASTER_PATTERNS:
        raise ValueError(f"Unsupported raster pattern: {spec.raster_pattern}")
    if float(spec.square_side_len_m) <= 0.0:
        raise ValueError("square_side_len_m must be > 0")
    if float(spec.line_spacing_m) <= 0.0:
        raise ValueError("line_spacing_m must be > 0")
    if float(spec.line_spacing_m) >= float(spec.square_side_len_m) / 2.0:
        raise ValueError("line_spacing_m must be < half of square_side_len_m")
    if int(spec.pointcloud_downsample_rate) < 1:
        raise ValueError("pointcloud_downsample_rate must be >= 1")


def build_raster_points_from_selected_square(
    selected_points_xyz: np.ndarray,
    spec: RasterPatternSpec,
) -> np.ndarray:
    """Build final raster points by concatenating per-line sub-trajectories."""

    validate_raster_pattern_spec(spec)
    points_xyz = _as_points_array(selected_points_xyz)
    basis = _estimate_local_surface_basis(points_xyz, spec)
    surface_samples = _build_local_surface_samples(points_xyz, basis)
    square_samples = _select_square_uv_samples(surface_samples, spec)
    line_subtrajectories = _build_line_subtrajectories(square_samples, spec)
    if str(spec.raster_pattern) == RASTER_PATTERN_BOUSTROPHEDON:
        return _concat_boustrophedon_lines(line_subtrajectories)
    return _concat_unidirectional_retract_lines(line_subtrajectories)


def build_local_tangent_square_corners(
    selected_points_xyz: np.ndarray,
    spec: RasterPatternSpec,
) -> np.ndarray:
    """Build selected-square corners on the local PCA tangent plane."""

    validate_raster_pattern_spec(spec)
    points_xyz = _as_points_array(selected_points_xyz)
    basis = _estimate_local_surface_basis(points_xyz, spec)
    half_side = float(spec.square_side_len_m) / 2.0
    uv_corners = np.asarray(
        [
            [-half_side, -half_side],
            [half_side, -half_side],
            [half_side, half_side],
            [-half_side, half_side],
        ],
        dtype=float,
    )
    corners = []
    for uv in uv_corners:
        corner = basis.center_xyz + float(uv[0]) * basis.u_axis + float(uv[1]) * basis.v_axis
        corners.append(corner)
    return np.asarray(corners, dtype=float)


def downsample_line_points(line_points_xyz: np.ndarray, pointcloud_downsample_rate: int) -> np.ndarray:
    """Downsample one line while preserving both endpoints."""

    points_xyz = _as_points_array(line_points_xyz)
    rate = int(pointcloud_downsample_rate)
    if rate < 1:
        raise ValueError("pointcloud_downsample_rate must be >= 1")
    if rate == 1 or points_xyz.shape[0] <= 2:
        return points_xyz
    kept_indices = list(range(0, points_xyz.shape[0], rate))
    last_idx = int(points_xyz.shape[0] - 1)
    if kept_indices[-1] != last_idx:
        kept_indices.append(last_idx)
    return points_xyz[np.asarray(kept_indices, dtype=int)]


def _as_points_array(points_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    if points.shape[0] == 0:
        raise ValueError("points_xyz must not be empty")
    return points


def _estimate_local_surface_basis(points_xyz: np.ndarray, spec: RasterPatternSpec) -> LocalSurfaceBasis:
    center = np.asarray(spec.square_center_xyz, dtype=float)
    nearby = _select_nearby_points_for_pca(points_xyz, center, float(spec.square_side_len_m))
    normal = _estimate_pca_normal(nearby)
    u_axis = _build_reference_tangent(normal)
    v_axis = _normalize(np.cross(normal, u_axis))
    return LocalSurfaceBasis(
        center_xyz=center,
        u_axis=u_axis,
        v_axis=v_axis,
        normal_axis=normal,
    )


def _select_nearby_points_for_pca(points_xyz: np.ndarray, center_xyz: np.ndarray, side_len_m: float) -> np.ndarray:
    radius = max(float(side_len_m), 1e-6)
    deltas = points_xyz - center_xyz
    distances = np.linalg.norm(deltas, axis=1)
    nearby = points_xyz[distances <= radius]
    if nearby.shape[0] >= 3:
        return nearby
    if points_xyz.shape[0] < 3:
        raise ValueError("At least 3 points are required to estimate local surface normal")
    return points_xyz


def _estimate_pca_normal(points_xyz: np.ndarray) -> np.ndarray:
    centered = points_xyz - np.mean(points_xyz, axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal = np.asarray(eigenvectors[:, int(np.argmin(eigenvalues))], dtype=float)
    if normal[2] < 0.0:
        normal = -normal
    return _normalize(normal)


def _build_reference_tangent(normal_axis: np.ndarray) -> np.ndarray:
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(reference, normal_axis))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    tangent = reference - float(np.dot(reference, normal_axis)) * normal_axis
    return _normalize(tangent)


def _project_points_to_basis(points_xyz: np.ndarray, basis: LocalSurfaceBasis) -> np.ndarray:
    deltas = points_xyz - basis.center_xyz
    u_values = deltas @ basis.u_axis
    v_values = deltas @ basis.v_axis
    return np.column_stack((u_values, v_values))


def _project_normal_offsets(points_xyz: np.ndarray, basis: LocalSurfaceBasis) -> np.ndarray:
    deltas = points_xyz - basis.center_xyz
    return deltas @ basis.normal_axis


def _build_local_surface_samples(points_xyz: np.ndarray, basis: LocalSurfaceBasis) -> LocalSurfaceSamples:
    return LocalSurfaceSamples(
        xyz_points=points_xyz,
        uv_points=_project_points_to_basis(points_xyz, basis),
        w_values=_project_normal_offsets(points_xyz, basis),
        basis=basis,
    )


def _select_square_uv_samples(
    samples: LocalSurfaceSamples,
    spec: RasterPatternSpec,
) -> LocalSurfaceSamples:
    half_side = float(spec.square_side_len_m) / 2.0
    mask = (
        (samples.uv_points[:, 0] >= -half_side)
        & (samples.uv_points[:, 0] <= half_side)
        & (samples.uv_points[:, 1] >= -half_side)
        & (samples.uv_points[:, 1] <= half_side)
    )
    if not bool(np.any(mask)):
        raise ValueError("No point-cloud points found inside local tangent square")
    return LocalSurfaceSamples(
        xyz_points=samples.xyz_points[mask],
        uv_points=samples.uv_points[mask],
        w_values=samples.w_values[mask],
        basis=samples.basis,
    )


def _build_line_subtrajectories(
    samples: LocalSurfaceSamples,
    spec: RasterPatternSpec,
) -> List[np.ndarray]:
    centers_y = _line_centers_y(spec)
    half_band = float(spec.line_spacing_m) / 2.0
    lines: List[np.ndarray] = []
    for center_y in centers_y:
        nearby_count = _count_points_near_line(samples.uv_points, center_y, half_band)
        if nearby_count == 0:
            continue
        line_points = _interpolate_regular_line_points(samples, spec, center_y, nearby_count)
        lines.append(line_points)
    if not lines:
        raise ValueError("No line sub_trajectory points were generated")
    return lines


def _interpolate_regular_line_points(
    samples: LocalSurfaceSamples,
    spec: RasterPatternSpec,
    center_y: float,
    nearby_count: int,
) -> np.ndarray:
    half_side = float(spec.square_side_len_m) / 2.0
    n_points = _line_sample_count(int(nearby_count), int(spec.pointcloud_downsample_rate))
    u_values = np.linspace(-half_side, half_side, n_points)
    target_uv = np.column_stack((u_values, np.full(n_points, float(center_y))))
    interpolated_w = _interpolate_w_values(target_uv, samples.uv_points, samples.w_values)
    points = []
    for idx in range(n_points):
        u = float(target_uv[idx, 0])
        v = float(target_uv[idx, 1])
        w = float(interpolated_w[idx])
        point = (
            samples.basis.center_xyz
            + u * samples.basis.u_axis
            + v * samples.basis.v_axis
            + w * samples.basis.normal_axis
        )
        points.append(point)
    return np.asarray(points, dtype=float)


def _line_sample_count(nearby_count: int, pointcloud_downsample_rate: int) -> int:
    rate = max(1, int(pointcloud_downsample_rate))
    return max(2, int(np.ceil(float(nearby_count) / float(rate))))


def _interpolate_w_values(
    target_uv: np.ndarray,
    source_uv: np.ndarray,
    source_w: np.ndarray,
) -> np.ndarray:
    interpolated = []
    for uv in target_uv:
        distances = np.linalg.norm(source_uv - uv, axis=1)
        nearest_count = min(8, int(source_uv.shape[0]))
        nearest_indices = np.argsort(distances)[:nearest_count]
        nearest_distances = distances[nearest_indices]
        nearest_w = source_w[nearest_indices]
        if float(nearest_distances[0]) <= 1e-12:
            interpolated.append(float(nearest_w[0]))
            continue
        weights = 1.0 / np.maximum(nearest_distances, 1e-12) ** 2
        interpolated.append(float(np.sum(weights * nearest_w) / np.sum(weights)))
    return np.asarray(interpolated, dtype=float)


def _line_centers_y(spec: RasterPatternSpec) -> List[float]:
    half_side = float(spec.square_side_len_m) / 2.0
    spacing = float(spec.line_spacing_m)
    y_min = -half_side
    y_max = half_side
    centers: List[float] = []
    y = y_min
    while y <= y_max + 1e-12:
        centers.append(float(min(y, y_max)))
        y += spacing
    if abs(float(centers[-1]) - y_max) > 1e-12:
        centers.append(float(y_max))
    return centers


def _count_points_near_line(
    uv_points: np.ndarray,
    center_y: float,
    half_band_m: float,
) -> int:
    mask = np.abs(uv_points[:, 1] - float(center_y)) <= float(half_band_m) + 1e-12
    return int(np.count_nonzero(mask))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return vector / norm


def _point_lifted_in_z(point_xyz: np.ndarray, lift_m: float) -> np.ndarray:
    lifted = np.asarray(point_xyz, dtype=float).copy()
    lifted[2] += float(lift_m)
    return lifted


def _append_line_with_lifted_transition(
    chunks: List[np.ndarray],
    current_line: np.ndarray,
    next_line: np.ndarray | None,
) -> None:
    chunks.append(current_line)
    if next_line is None:
        return
    current_end = current_line[-1]
    next_start = next_line[0]
    transition = np.asarray(
        [
            _point_lifted_in_z(current_end, INTER_LINE_RETRACT_LIFT_M),
            _point_lifted_in_z(next_start, INTER_LINE_RETRACT_LIFT_M),
            next_start,
        ],
        dtype=float,
    )
    chunks.append(transition)


def _concat_boustrophedon_lines(line_subtrajectories: Sequence[np.ndarray]) -> np.ndarray:
    oriented_lines: List[np.ndarray] = []
    for idx, line_points in enumerate(line_subtrajectories):
        if idx % 2 == 0:
            oriented_lines.append(line_points)
        else:
            oriented_lines.append(line_points[::-1])
    chunks: List[np.ndarray] = []
    for idx, line_points in enumerate(oriented_lines):
        next_line = oriented_lines[idx + 1] if idx + 1 < len(oriented_lines) else None
        _append_line_with_lifted_transition(chunks, line_points, next_line)
    return np.vstack(chunks)


def _concat_unidirectional_retract_lines(line_subtrajectories: Sequence[np.ndarray]) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for idx, line_points in enumerate(line_subtrajectories):
        next_line = line_subtrajectories[idx + 1] if idx + 1 < len(line_subtrajectories) else None
        _append_line_with_lifted_transition(chunks, line_points, next_line)
    return np.vstack(chunks)

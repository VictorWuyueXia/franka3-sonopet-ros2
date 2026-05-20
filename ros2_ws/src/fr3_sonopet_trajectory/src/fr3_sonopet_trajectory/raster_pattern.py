from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fr3_sonopet_trajectory.surface_geometry import SurfaceFrame

INTER_LINE_RETRACT_LIFT_M = 0.005
SUPPORTED_PATTERNS = {"boustrophedon", "unidirectional_retract"}


@dataclass(frozen=True)
class RasterSpec:
    square_side_m: float = 0.02
    line_spacing_m: float = 0.002
    downsample_rate: int = 1
    pattern: str = "unidirectional_retract"


@dataclass(frozen=True)
class SurfaceRaster:
    points: np.ndarray
    normal_reference_points: np.ndarray
    segment_names: list[str]


def build_raster_uv(spec: RasterSpec) -> np.ndarray:
    """Build reference-style UV points with explicit lifted line transitions."""
    if spec.square_side_m <= 0.0:
        raise ValueError("square_side_m must be positive")
    if spec.line_spacing_m <= 0.0:
        raise ValueError("line_spacing_m must be positive")
    if spec.downsample_rate <= 0:
        raise ValueError("downsample_rate must be positive")
    if spec.pattern not in SUPPORTED_PATTERNS:
        raise ValueError(f"Unsupported raster pattern: {spec.pattern}")

    half = spec.square_side_m / 2.0
    line_count = max(2, int(round(spec.square_side_m / spec.line_spacing_m)) + 1)
    line_values = np.linspace(-half, half, line_count, dtype=np.float64)

    # Endpoint-preserving decimation keeps each raster line geometrically closed.
    column_indices = np.arange(0, line_count, spec.downsample_rate, dtype=np.int64)
    if column_indices[-1] != line_count - 1:
        column_indices = np.append(column_indices, line_count - 1)

    lines: list[np.ndarray] = []
    u_values = line_values[column_indices]
    for v in line_values:
        row_u = u_values
        row_v = np.full(row_u.shape, v, dtype=np.float64)
        lines.append(np.column_stack((row_u, row_v)))

    return _concat_lines_with_retract(lines)


def build_surface_raster(
    patch_points: np.ndarray,
    frame: SurfaceFrame,
    spec: RasterSpec,
) -> SurfaceRaster:
    """Interpolate selected pointcloud samples into a reference-style surface raster."""
    if spec.square_side_m <= 0.0:
        raise ValueError("square_side_m must be positive")
    if spec.line_spacing_m <= 0.0:
        raise ValueError("line_spacing_m must be positive")
    if spec.downsample_rate <= 0:
        raise ValueError("downsample_rate must be positive")
    if spec.pattern not in SUPPORTED_PATTERNS:
        raise ValueError(f"Unsupported raster pattern: {spec.pattern}")

    points = np.asarray(patch_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Patch points must be an N x 3 matrix")
    half = spec.square_side_m / 2.0
    deltas = points - frame.center
    uv = np.column_stack((deltas @ frame.tangent, deltas @ frame.bitangent))
    w_values = deltas @ frame.normal
    square_mask = (
        (uv[:, 0] >= -half)
        & (uv[:, 0] <= half)
        & (uv[:, 1] >= -half)
        & (uv[:, 1] <= half)
    )
    if not bool(np.any(square_mask)):
        raise ValueError("No point-cloud points found inside local tangent square")

    square_uv = uv[square_mask]
    square_w = w_values[square_mask]
    line_centers = np.arange(-half, half + 0.5 * spec.line_spacing_m, spec.line_spacing_m)
    line_centers[-1] = half
    lines: list[np.ndarray] = []
    normal_lines: list[np.ndarray] = []
    for center_v in line_centers:
        near_line = np.abs(square_uv[:, 1] - float(center_v)) <= 0.5 * spec.line_spacing_m
        nearby_count = int(np.count_nonzero(near_line))
        if nearby_count == 0:
            continue
        sample_count = max(2, int(np.ceil(float(nearby_count) / float(spec.downsample_rate))))
        line_uv = np.column_stack(
            (
                np.linspace(-half, half, sample_count, dtype=np.float64),
                np.full(sample_count, float(center_v), dtype=np.float64),
            )
        )
        distances = np.linalg.norm(square_uv[None, :, :] - line_uv[:, None, :], axis=2)
        nearest_count = min(8, square_uv.shape[0])
        nearest = np.argsort(distances, axis=1)[:, :nearest_count]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        nearest_w = square_w[nearest]
        weights = 1.0 / np.maximum(nearest_distances, 1e-12) ** 2
        exact = nearest_distances[:, 0] <= 1e-12
        interpolated_w = np.sum(weights * nearest_w, axis=1) / np.sum(weights, axis=1)
        interpolated_w[exact] = nearest_w[exact, 0]
        line = (
            frame.center[None, :]
            + line_uv[:, 0, None] * frame.tangent[None, :]
            + line_uv[:, 1, None] * frame.bitangent[None, :]
            + interpolated_w[:, None] * frame.normal[None, :]
        )
        lines.append(line)
        normal_lines.append(line.copy())

    if not lines:
        raise ValueError("No line sub-trajectory points were generated")
    points, normal_points, segment_names = _concat_lines_with_retract(
        lines,
        normal_lines=normal_lines,
    )
    return SurfaceRaster(points=points, normal_reference_points=normal_points, segment_names=segment_names)


def _concat_lines_with_retract(
    lines: list[np.ndarray],
    *,
    normal_lines: list[np.ndarray] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, list[str]]:
    chunks: list[np.ndarray] = []
    normal_chunks: list[np.ndarray] = []
    segment_names: list[str] = []
    for idx, line in enumerate(lines):
        full_line = np.vstack((line, line[-2::-1]))
        chunks.append(full_line)
        segment_names.extend(["raster"] * full_line.shape[0])
        if normal_lines is not None:
            full_normal_line = np.vstack((normal_lines[idx], normal_lines[idx][-2::-1]))
            normal_chunks.append(full_normal_line)
        if idx + 1 == len(lines):
            continue
        next_line = lines[idx + 1]
        lift = np.zeros(line.shape[1], dtype=np.float64)
        lift[-1] = INTER_LINE_RETRACT_LIFT_M
        transition = np.asarray([full_line[-1] + lift, next_line[0] + lift, next_line[0]])
        chunks.append(transition)
        segment_names.extend(["retract", "retract", "raster"])
        if normal_lines is not None:
            normal_transition = np.asarray(
                [full_normal_line[-1], normal_lines[idx + 1][0], normal_lines[idx + 1][0]]
            )
            normal_chunks.append(normal_transition)
    points = np.vstack(chunks)
    if normal_lines is None:
        return points
    return points, np.vstack(normal_chunks), segment_names

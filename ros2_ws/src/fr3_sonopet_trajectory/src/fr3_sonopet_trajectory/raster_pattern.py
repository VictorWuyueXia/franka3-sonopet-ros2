from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RasterSpec:
    square_side_m: float = 0.02
    line_spacing_m: float = 0.002
    downsample_rate: int = 1
    pattern: str = "unidirectional_retract"


def build_raster_uv(spec: RasterSpec) -> np.ndarray:
    """Build a deterministic UV raster over a square patch."""
    if spec.square_side_m <= 0.0:
        raise ValueError("square_side_m must be positive")
    if spec.line_spacing_m <= 0.0:
        raise ValueError("line_spacing_m must be positive")
    if spec.downsample_rate <= 0:
        raise ValueError("downsample_rate must be positive")

    half = spec.square_side_m / 2.0
    line_count = max(2, int(round(spec.square_side_m / spec.line_spacing_m)) + 1)
    line_values = np.linspace(-half, half, line_count, dtype=np.float64)

    # Endpoint-preserving decimation keeps each raster line geometrically closed.
    column_indices = np.arange(0, line_count, spec.downsample_rate, dtype=np.int64)
    if column_indices[-1] != line_count - 1:
        column_indices = np.append(column_indices, line_count - 1)

    rows: list[np.ndarray] = []
    u_values = line_values[column_indices]
    for row_index, v in enumerate(line_values):
        if spec.pattern == "boustrophedon" and row_index % 2 == 1:
            row_u = u_values[::-1]
        elif spec.pattern in {"boustrophedon", "unidirectional_retract"}:
            row_u = u_values
        else:
            raise ValueError(f"Unsupported raster pattern: {spec.pattern}")
        row_v = np.full(row_u.shape, v, dtype=np.float64)
        rows.append(np.column_stack((row_u, row_v)))

    return np.vstack(rows)

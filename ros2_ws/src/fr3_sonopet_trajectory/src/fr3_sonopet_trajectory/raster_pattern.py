from __future__ import annotations

from dataclasses import dataclass


Point2 = tuple[float, float]


@dataclass(frozen=True)
class RasterSpec:
    square_side_m: float = 0.02
    line_spacing_m: float = 0.002
    pattern: str = "unidirectional_retract"


def build_raster_uv(spec: RasterSpec) -> list[Point2]:
    """Build a deterministic UV raster over a square patch."""
    if spec.square_side_m <= 0.0:
        raise ValueError("square_side_m must be positive")
    if spec.line_spacing_m <= 0.0:
        raise ValueError("line_spacing_m must be positive")

    half = spec.square_side_m / 2.0
    line_count = max(2, int(round(spec.square_side_m / spec.line_spacing_m)) + 1)
    points: list[Point2] = []

    for index in range(line_count):
        v = -half + index * (spec.square_side_m / (line_count - 1))
        left = (-half, v)
        right = (half, v)
        if spec.pattern == "boustrophedon" and index % 2 == 1:
            points.extend([right, left])
        elif spec.pattern in {"boustrophedon", "unidirectional_retract"}:
            points.extend([left, right])
        else:
            raise ValueError(f"Unsupported raster pattern: {spec.pattern}")

    return points


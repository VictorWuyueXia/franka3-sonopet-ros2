from __future__ import annotations

import math


Vector3 = tuple[float, float, float]


def normalize(vector: Vector3) -> Vector3:
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def stabilize_normal(normal: Vector3, reference: Vector3 = (0.0, 0.0, 1.0)) -> Vector3:
    """Keep a surface normal pointing consistently with a reference axis."""
    unit = normalize(normal)
    dot = sum(a * b for a, b in zip(unit, reference))
    if dot < 0.0:
        return tuple(-value for value in unit)  # type: ignore[return-value]
    return unit


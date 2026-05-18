from __future__ import annotations


def require_complete_fraction(fraction: float, minimum: float = 1.0) -> None:
    if fraction < minimum:
        raise ValueError(f"Planned trajectory fraction {fraction:.3f} is below {minimum:.3f}")


def require_nonempty_points(point_count: int) -> None:
    if point_count <= 0:
        raise ValueError("Planned trajectory must contain at least one point")


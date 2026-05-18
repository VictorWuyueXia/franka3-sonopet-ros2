from __future__ import annotations

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


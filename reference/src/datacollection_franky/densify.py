"""densify.py

Distance based densification:
- Enforce max adjacent translation length in meters.
- Use quaternion slerp for orientation.

Waypoints are represented as simple dict-like objects with attributes:
  .xyz -> (x, y, z)
  .quat_xyzw -> (qx, qy, qz, qw)

We keep it generic so it works with cloudpoint_wrapper.PoseWaypoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Tuple, Type, TypeVar

from .math_utils import l2_dist, quat_slerp


class PoseLike(Protocol):
    xyz: Tuple[float, float, float]
    quat_xyzw: Tuple[float, float, float, float]


T = TypeVar("T", bound=PoseLike)


@dataclass(frozen=True)
class SimplePose:
    xyz: Tuple[float, float, float]
    quat_xyzw: Tuple[float, float, float, float]


def densify_by_max_step(
    poses: Iterable[T],
    max_step_m: float,
    waypoint_cls: Type[T] | Type[SimplePose] = SimplePose,
) -> List[T]:
    """Insert intermediate waypoints so adjacent translation distance <= max_step_m."""

    pts = list(poses)
    if len(pts) < 2:
        return list(pts)

    if max_step_m <= 0.0:
        raise ValueError("max_step_m must be > 0")

    dense: List[T] = []
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        if i == 0:
            dense.append(a)

        d = l2_dist(a.xyz, b.xyz)
        if d <= float(max_step_m) + 1e-12:
            dense.append(b)
            continue

        # Number of segments = ceil(d / max_step)
        n_seg = int(math.ceil(d / float(max_step_m)))
        # We already have endpoints, so insert n_seg-1 interior points.
        for k in range(1, n_seg):
            t = float(k) / float(n_seg)
            x = float(a.xyz[0]) + t * (float(b.xyz[0]) - float(a.xyz[0]))
            y = float(a.xyz[1]) + t * (float(b.xyz[1]) - float(a.xyz[1]))
            z = float(a.xyz[2]) + t * (float(b.xyz[2]) - float(a.xyz[2]))
            q = quat_slerp(a.quat_xyzw, b.quat_xyzw, t)
            dense.append(waypoint_cls(xyz=(x, y, z), quat_xyzw=q))
        dense.append(b)

    return dense

"""math_utils.py

Quaternion and interpolation helpers.

Quaternion convention: (x, y, z, w).
"""

from __future__ import annotations

import math
from typing import Tuple


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


def l2_dist(a: Vec3, b: Vec3) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def quat_normalize(q: Quat) -> Quat:
    n = math.sqrt(float(q[0]) * float(q[0]) + float(q[1]) * float(q[1]) + float(q[2]) * float(q[2]) + float(q[3]) * float(q[3]))
    if n <= 1e-12:
        raise ValueError("Quaternion normalization failed: near-zero norm")
    return (float(q[0]) / n, float(q[1]) / n, float(q[2]) / n, float(q[3]) / n)


def quat_abs_dot(q0: Quat, q1: Quat) -> float:
    return abs(float(q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]))


def quat_slerp(q0: Quat, q1: Quat, alpha: float) -> Quat:
    """Spherical linear interpolation between q0 and q1.

    Args:
        q0, q1: unit or non-unit quaternions.
        alpha: in [0, 1].
    """

    qa = quat_normalize(q0)
    qb = quat_normalize(q1)

    dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]
    if dot < 0.0:
        qb = (-qb[0], -qb[1], -qb[2], -qb[3])
        dot = -dot

    if dot > 0.9995:
        # Near linear.
        q = (
            qa[0] + alpha * (qb[0] - qa[0]),
            qa[1] + alpha * (qb[1] - qa[1]),
            qa[2] + alpha * (qb[2] - qa[2]),
            qa[3] + alpha * (qb[3] - qa[3]),
        )
        return quat_normalize(q)

    theta_0 = math.acos(float(max(-1.0, min(1.0, dot))))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * float(alpha)

    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0

    return (
        float(s0 * qa[0] + s1 * qb[0]),
        float(s0 * qa[1] + s1 * qb[1]),
        float(s0 * qa[2] + s1 * qb[2]),
        float(s0 * qa[3] + s1 * qb[3]),
    )

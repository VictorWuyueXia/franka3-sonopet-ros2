from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from fr3_sonopet_interfaces.msg import RasterPlan
from geometry_msgs.msg import Quaternion
from trajectory_msgs.msg import JointTrajectoryPoint

BASE_FRAME = "fr3_link0"
TOOL_FRAME = "sonopet_tcp"
IK_LINK_FRAME = "fr3_link8"
PLANNING_GROUP = "fr3_arm"
RASTER_PLAN_TOPIC = "/sonopet/raster_plan"
IK_SERVICE = "/compute_ik"
CONTROLLER_ACTION = "/fr3_arm_controller/follow_joint_trajectory"
PARKING_LIFT_M = 0.05
RETRACT_LIFT_M = 0.05
MAX_CARTESIAN_STEP_M = 0.01
APPROACH_SPEED_M_S = 0.03
RASTER_SPEED_M_S = 0.005
POINT_TIME_FLOOR_S = 0.05
IK_TIMEOUT_S = 1.0
JOINT_NAMES = (
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
)


@dataclass(frozen=True)
class CartesianSegment:
    name: str
    tcp_matrices: list[np.ndarray]
    speed_m_s: float


def build_cartesian_segments(plan: RasterPlan, current_tcp_matrix: np.ndarray) -> list[CartesianSegment]:
    """Re-anchor raster offsets at the current TCP pose and wrap them with staged motion."""
    plan_points = np.array(
        [[pose.position.x, pose.position.y, pose.position.z] for pose in plan.poses.poses],
        dtype=np.float64,
    )
    raster = np.repeat(current_tcp_matrix[None, :, :], plan_points.shape[0], axis=0)
    raster[:, :3, 3] = current_tcp_matrix[:3, 3] + plan_points - plan_points[0]
    first = raster[0]
    last = raster[-1]
    parking = np.array(first, dtype=np.float64, copy=True)
    retract = np.array(last, dtype=np.float64, copy=True)
    parking[2, 3] += PARKING_LIFT_M
    retract[2, 3] += RETRACT_LIFT_M
    return [
        CartesianSegment(
            "approach",
            densified_matrices([current_tcp_matrix, parking, first]),
            APPROACH_SPEED_M_S,
        ),
        CartesianSegment(
            "raster",
            densified_matrices(list(raster)),
            RASTER_SPEED_M_S,
        ),
        CartesianSegment(
            "retract",
            densified_matrices([last, retract]),
            APPROACH_SPEED_M_S,
        ),
        CartesianSegment(
            "return_to_start",
            densified_matrices([retract, current_tcp_matrix]),
            APPROACH_SPEED_M_S,
        ),
    ]


def densified_matrices(matrices: list[np.ndarray]) -> list[np.ndarray]:
    """Limit Cartesian spacing before sequential IK, matching the reference compiler."""
    dense = [matrices[0]]
    for start, end in zip(matrices[:-1], matrices[1:], strict=True):
        delta = end[:3, 3] - start[:3, 3]
        steps = max(1, int(math.ceil(float(np.linalg.norm(delta)) / MAX_CARTESIAN_STEP_M)))
        for index in range(1, steps + 1):
            t = float(index) / float(steps)
            matrix = np.array(end, dtype=np.float64, copy=True)
            matrix[:3, 3] = start[:3, 3] + t * delta
            dense.append(matrix)
    return dense


def joint_trajectory_points(
    joints: list[dict[str, float]],
    tcp_matrices: list[np.ndarray],
    speed_m_s: float,
) -> list[JointTrajectoryPoint]:
    points: list[JointTrajectoryPoint] = []
    elapsed_s = POINT_TIME_FLOOR_S
    for index, joint_dict in enumerate(joints):
        if index > 0:
            distance_m = float(
                np.linalg.norm(tcp_matrices[index][:3, 3] - tcp_matrices[index - 1][:3, 3])
            )
            elapsed_s += max(distance_m / float(speed_m_s), POINT_TIME_FLOOR_S)
        sec = int(math.floor(elapsed_s))
        point = JointTrajectoryPoint()
        point.positions = [float(joint_dict[name]) for name in JOINT_NAMES]
        point.time_from_start.sec = sec
        point.time_from_start.nanosec = int(round((elapsed_s - float(sec)) * 1_000_000_000.0))
        points.append(point)
    return points


def quaternion_from_matrix(matrix: np.ndarray) -> Quaternion:
    rot = matrix[:3, :3]
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        x = (rot[2, 1] - rot[1, 2]) / scale
        y = (rot[0, 2] - rot[2, 0]) / scale
        z = (rot[1, 0] - rot[0, 1]) / scale
        w = 0.25 * scale
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        scale = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        x = 0.25 * scale
        y = (rot[0, 1] + rot[1, 0]) / scale
        z = (rot[0, 2] + rot[2, 0]) / scale
        w = (rot[2, 1] - rot[1, 2]) / scale
    elif rot[1, 1] > rot[2, 2]:
        scale = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        x = (rot[0, 1] + rot[1, 0]) / scale
        y = 0.25 * scale
        z = (rot[1, 2] + rot[2, 1]) / scale
        w = (rot[0, 2] - rot[2, 0]) / scale
    else:
        scale = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        x = (rot[0, 2] + rot[2, 0]) / scale
        y = (rot[1, 2] + rot[2, 1]) / scale
        z = 0.25 * scale
        w = (rot[1, 0] - rot[0, 1]) / scale
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def rotation_matrix_from_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

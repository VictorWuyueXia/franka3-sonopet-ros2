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
PREVIEW_JOINT_STATES_TOPIC = "/sonopet/preview/joint_states"
IK_SERVICE = "/fr3/compute_ik"
FK_SERVICE = "/compute_fk"
CONTROLLER_ACTION = "/fr3_arm_controller/follow_joint_trajectory"
MAX_CARTESIAN_STEP_M = 0.01
POINT_TIME_FLOOR_S = 0.05
SEGMENT_SETTLING_TIME_S = 5.0
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


def build_cartesian_segments(plan: RasterPlan, current_tcp_matrix: np.ndarray) -> list[np.ndarray]:
    """Use raster plan positions with the latched TCP orientation."""
    plan_points = np.array(
        [[pose.position.x, pose.position.y, pose.position.z] for pose in plan.poses.poses],
        dtype=np.float64,
    )
    raster = np.repeat(current_tcp_matrix[None, :, :], plan_points.shape[0], axis=0)
    raster[:, :3, 3] = plan_points
    return list(raster)


def staged_cartesian_segments(
    idle_tcp_matrix: np.ndarray,
    raster: list[np.ndarray],
    motion_speed_m_s: float,
    raster_speed_m_s: float,
    parking_lift_m: float,
) -> list[CartesianSegment]:
    """Build the reference-style Cartesian stages around the relative raster path."""
    first = raster[0]
    last = raster[-1]
    parking = np.array(first, dtype=np.float64, copy=True)
    retract = np.array(last, dtype=np.float64, copy=True)
    parking[2, 3] += parking_lift_m
    retract[2, 3] += parking_lift_m
    return [
        CartesianSegment("idle_to_parking", densified_matrices([idle_tcp_matrix, parking]), motion_speed_m_s),
        CartesianSegment("parking_to_first", densified_matrices([parking, first]), raster_speed_m_s),
        CartesianSegment("raster", densified_matrices(raster), raster_speed_m_s),
        CartesianSegment("retract", densified_matrices([last, retract]), motion_speed_m_s),
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
            rotation_u, _, rotation_vt = np.linalg.svd((1.0 - t) * start[:3, :3] + t * end[:3, :3])
            matrix[:3, :3] = rotation_u @ rotation_vt
            if np.linalg.det(matrix[:3, :3]) < 0.0:
                rotation_u[:, -1] *= -1.0
                matrix[:3, :3] = rotation_u @ rotation_vt
            dense.append(matrix)
    return dense


def joint_trajectory_points(
    joints: list[dict[str, float]],
    tcp_matrices: list[np.ndarray],
    speed_m_s: float,
) -> list[JointTrajectoryPoint]:
    points: list[JointTrajectoryPoint] = []
    elapsed_s = SEGMENT_SETTLING_TIME_S
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


def joint_interpolation_points(
    start: dict[str, float],
    goal: dict[str, float],
    duration_s: float,
) -> list[JointTrajectoryPoint]:
    max_delta = max(abs(float(goal[name]) - float(start[name])) for name in JOINT_NAMES)
    steps = max(1, int(math.ceil(max_delta / 0.05)))
    points: list[JointTrajectoryPoint] = []
    for index in range(steps + 1):
        point = JointTrajectoryPoint()
        ratio = float(index) / float(steps)
        point.positions = [
            (1.0 - ratio) * float(start[name]) + ratio * float(goal[name]) for name in JOINT_NAMES
        ]
        elapsed_s = SEGMENT_SETTLING_TIME_S + ratio * duration_s
        sec = int(math.floor(elapsed_s))
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

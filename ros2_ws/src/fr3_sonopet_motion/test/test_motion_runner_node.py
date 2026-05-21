import math

import numpy as np
from fr3_sonopet_interfaces.msg import RasterPlan
from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import Header

from fr3_sonopet_motion.motion_geometry import (
    APPROACH_SPEED_M_S,
    RASTER_SPEED_M_S,
    build_cartesian_segments,
    joint_trajectory_points,
    rotation_matrix_from_quaternion,
)


def _plan_from_positions(positions: list[tuple[float, float, float]]) -> RasterPlan:
    plan = RasterPlan()
    plan.header = Header(frame_id="fr3_link0")
    plan.poses.header = plan.header
    for index, xyz in enumerate(positions):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = xyz
        pose.orientation = Quaternion(x=0.0, y=0.0, z=float(index), w=1.0)
        plan.poses.poses.append(pose)
    return plan


def _duration_seconds(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def test_raster_offsets_start_at_current_tcp_and_ignore_plan_orientation():
    current = np.eye(4, dtype=np.float64)
    current[:3, :3] = rotation_matrix_from_quaternion(
        np.array([0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0)])
    )
    current[:3, 3] = np.array([0.45, -0.10, 0.22])
    plan = _plan_from_positions([(1.0, 2.0, 3.0), (1.01, 2.0, 3.005)])

    raster = build_cartesian_segments(plan, current)[1].tcp_matrices

    assert np.allclose(raster[0][:3, 3], current[:3, 3])
    assert np.allclose(raster[-1][:3, 3], current[:3, 3] + np.array([0.01, 0.0, 0.005]))
    assert all(np.allclose(matrix[:3, :3], current[:3, :3]) for matrix in raster)


def test_cartesian_segments_follow_approach_raster_retract_return_order():
    current = np.eye(4, dtype=np.float64)
    current[:3, 3] = np.array([0.4, 0.0, 0.3])
    plan = _plan_from_positions([(0.0, 0.0, 0.0), (0.02, 0.0, 0.0)])

    segments = build_cartesian_segments(plan, current)

    assert [segment.name for segment in segments] == [
        "approach",
        "raster",
        "retract",
        "return_to_start",
    ]
    assert np.allclose(segments[0].tcp_matrices[-1][:3, 3], current[:3, 3])
    assert segments[1].speed_m_s == RASTER_SPEED_M_S
    assert segments[0].speed_m_s == APPROACH_SPEED_M_S
    assert segments[2].tcp_matrices[-1][2, 3] > segments[1].tcp_matrices[-1][2, 3]


def test_raster_timing_is_slower_than_approach_for_equal_distance():
    joints = [
        {
            "fr3_joint1": 0.0,
            "fr3_joint2": 0.0,
            "fr3_joint3": 0.0,
            "fr3_joint4": 0.0,
            "fr3_joint5": 0.0,
            "fr3_joint6": 0.0,
            "fr3_joint7": 0.0,
        }
        for _ in range(2)
    ]
    start = np.eye(4, dtype=np.float64)
    end = np.eye(4, dtype=np.float64)
    end[0, 3] = 0.01

    approach = joint_trajectory_points(joints, [start, end], APPROACH_SPEED_M_S)
    raster = joint_trajectory_points(joints, [start, end], RASTER_SPEED_M_S)

    approach_dt = _duration_seconds(approach[-1].time_from_start) - _duration_seconds(
        approach[0].time_from_start
    )
    raster_dt = _duration_seconds(raster[-1].time_from_start) - _duration_seconds(
        raster[0].time_from_start
    )
    assert raster_dt > approach_dt

import math
from pathlib import Path

import numpy as np
from fr3_sonopet_interfaces.msg import RasterPlan
from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import Header

from fr3_sonopet_motion.operator_policy import EXECUTE_TOKEN
from fr3_sonopet_motion.motion_geometry import (
    SEGMENT_SETTLING_TIME_S,
    build_cartesian_segments,
    joint_interpolation_points,
    joint_trajectory_points,
    rotation_matrix_from_quaternion,
    staged_cartesian_segments,
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


def test_raster_positions_land_on_plan_points_and_ignore_plan_orientation():
    current = np.eye(4, dtype=np.float64)
    current[:3, :3] = rotation_matrix_from_quaternion(
        np.array([0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0)])
    )
    current[:3, 3] = np.array([0.45, -0.10, 0.22])
    plan = _plan_from_positions([(1.0, 2.0, 3.0), (1.01, 2.0, 3.005)])

    raster = build_cartesian_segments(plan, current)

    assert np.allclose(raster[0][:3, 3], (1.0, 2.0, 3.0))
    assert np.allclose(raster[-1][:3, 3], (1.01, 2.0, 3.005))
    assert all(np.allclose(matrix[:3, :3], current[:3, :3]) for matrix in raster)


def test_cartesian_segments_follow_approach_raster_retract_return_order():
    current = np.eye(4, dtype=np.float64)
    current[:3, 3] = np.array([0.4, 0.0, 0.3])
    plan = _plan_from_positions([(0.0, 0.0, 0.0), (0.02, 0.0, 0.0)])

    raster = build_cartesian_segments(plan, current)
    idle = np.array(current, dtype=np.float64, copy=True)
    idle[0, 3] -= 0.05
    segments = staged_cartesian_segments(idle, raster, 0.03, 0.005, 0.05)

    assert [segment.name for segment in segments] == [
        "idle_to_parking",
        "parking_to_first",
        "raster",
        "retract",
    ]
    assert np.allclose(segments[1].tcp_matrices[-1][:3, 3], (0.0, 0.0, 0.0))
    assert segments[1].speed_m_s == 0.005
    assert segments[2].speed_m_s == 0.005
    assert segments[0].speed_m_s == 0.03
    assert segments[3].tcp_matrices[-1][2, 3] > segments[2].tcp_matrices[-1][2, 3]
    assert np.allclose(segments[0].tcp_matrices[0], idle)
    for segment in segments:
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
            for _ in segment.tcp_matrices
        ]
        points = joint_trajectory_points(joints, segment.tcp_matrices, segment.speed_m_s)
        distance_m = sum(
            float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
            for start, end in zip(segment.tcp_matrices[:-1], segment.tcp_matrices[1:], strict=True)
        )
        segment_dt = _duration_seconds(points[-1].time_from_start) - _duration_seconds(
            points[0].time_from_start
        )
        assert segment_dt + 1e-9 >= distance_m / segment.speed_m_s


def test_idle_to_parking_orientation_changes_gradually_after_idle_pose():
    idle = np.eye(4, dtype=np.float64)
    raster_start = np.eye(4, dtype=np.float64)
    raster_start[:3, :3] = rotation_matrix_from_quaternion(
        np.array([0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)])
    )
    raster_start[:3, 3] = np.array([0.08, 0.0, 0.0])
    segment = staged_cartesian_segments(idle, [raster_start], 0.03, 0.005, 0.05)[0]

    assert np.allclose(segment.tcp_matrices[0], idle)
    assert not np.allclose(segment.tcp_matrices[1][:3, :3], idle[:3, :3])
    assert not np.allclose(segment.tcp_matrices[1][:3, :3], raster_start[:3, :3])


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

    approach = joint_trajectory_points(joints, [start, end], 0.03)
    raster = joint_trajectory_points(joints, [start, end], 0.005)

    approach_dt = _duration_seconds(approach[-1].time_from_start) - _duration_seconds(
        approach[0].time_from_start
    )
    raster_dt = _duration_seconds(raster[-1].time_from_start) - _duration_seconds(
        raster[0].time_from_start
    )
    assert _duration_seconds(approach[0].time_from_start) == SEGMENT_SETTLING_TIME_S
    assert _duration_seconds(raster[0].time_from_start) == SEGMENT_SETTLING_TIME_S
    assert raster_dt > approach_dt


def test_joint_segments_use_configured_cartesian_speed_duration():
    start = {
        "fr3_joint1": 0.0,
        "fr3_joint2": 0.0,
        "fr3_joint3": 0.0,
        "fr3_joint4": 0.0,
        "fr3_joint5": 0.0,
        "fr3_joint6": 0.0,
        "fr3_joint7": 0.0,
    }
    goal = {
        "fr3_joint1": 0.1,
        "fr3_joint2": -0.1,
        "fr3_joint3": 0.0,
        "fr3_joint4": -0.2,
        "fr3_joint5": 0.0,
        "fr3_joint6": 0.2,
        "fr3_joint7": 0.0,
    }
    points = joint_interpolation_points(start, goal, 0.2 / 0.03)

    assert _duration_seconds(points[-1].time_from_start) >= 0.2 / 0.03


def test_execute_token_and_beginning_pose_latch_are_encoded():
    package_root = Path(__file__).resolve().parents[1]
    runner = (
        package_root / "src" / "fr3_sonopet_motion" / "motion_runner_node.py"
    ).read_text(encoding="utf-8")

    assert EXECUTE_TOKEN == "E"
    assert 'setattr(self, "_beginning_seed", None)' in runner
    assert "if self._beginning_seed is None:" in runner
    assert "idle_joint_positions" in runner
    assert 'segment.name == "idle_to_parking" and index == 0' in runner
    assert "solved_points.append(dict(seed))" in runner
    assert "def _run_stop_recovery(self, goal_handle):" in runner
    assert "def _compute_recovery_trajectory(self) -> list[JointTrajectory]:" in runner
    assert 'trajectories.append(("current_to_idle", current_to_idle))' in runner
    assert 'trajectories.append(("return_to_start", return_to_start))' in runner
    assert "/sonopet/stop_motion" in runner
    assert "cancel_goal_async" in runner
    assert "retract[2, 3] +=" in runner

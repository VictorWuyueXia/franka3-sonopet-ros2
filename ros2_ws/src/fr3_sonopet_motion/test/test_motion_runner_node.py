import math
from pathlib import Path

import numpy as np
from fr3_sonopet_interfaces.msg import RasterPlan
from fr3_sonopet_motion.motion_geometry import (
    JOINT_NAMES,
    PREVIEW_JOINT_STATES_TOPIC,
    SEGMENT_SETTLING_TIME_S,
    build_cartesian_segments,
    joint_interpolation_points,
    joint_trajectory_points,
    rotation_matrix_from_quaternion,
    staged_cartesian_segments,
)
from fr3_sonopet_motion.operator_policy import EXECUTE_TOKEN
from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import Header


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
    assert SEGMENT_SETTLING_TIME_S >= 1.5
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
    point_times = [_duration_seconds(point.time_from_start) for point in points]

    assert point_times[0] == SEGMENT_SETTLING_TIME_S
    assert all(later > earlier for earlier, later in zip(point_times[:-1], point_times[1:], strict=True))
    assert point_times[-1] >= SEGMENT_SETTLING_TIME_S + 0.2 / 0.03


def test_execute_token_and_beginning_pose_latch_are_encoded():
    package_root = Path(__file__).resolve().parents[1]
    runner = (
        package_root / "src" / "fr3_sonopet_motion" / "motion_runner_node.py"
    ).read_text(encoding="utf-8")

    assert EXECUTE_TOKEN == "E"
    assert "@dataclass(frozen=True)" in runner
    assert "class BeginningPose:" in runner
    assert "self._beginning_pose: BeginningPose | None = None" in runner
    assert 'lambda plan: setattr(self, "_latest_plan", plan)' in runner
    assert "self._set_beginning_pose_service = self.create_service(" in runner
    assert "Trigger" in runner
    assert 'SET_BEGINNING_POSE_SERVICE = "/fr3/set_beginning_pose"' in runner
    assert "def _set_beginning_pose(self, _request, response):" in runner
    assert "def _cache_beginning_pose(self, joint_state: JointState) -> None:" in runner
    assert "def _require_beginning_pose(self) -> BeginningPose:" in runner
    assert "beginning_pose = self._require_beginning_pose()" in runner
    assert "idle_joint_positions" in runner
    assert 'segment.name == "idle_to_parking" and index == 0' in runner
    assert "solved_points.append(dict(seed))" in runner
    assert "seed = dict(beginning_pose.joint_seed)" in runner
    assert "base_from_tcp = beginning_pose.base_from_tcp" in runner
    assert "link_from_tcp = beginning_pose.link_from_tcp" in runner
    assert "raster = build_cartesian_segments(self._latest_plan, base_from_tcp)" in runner
    assert "def _run_stop_recovery(self, goal_handle):" in runner
    assert "def _compute_recovery_trajectory(self) -> list[JointTrajectory]:" in runner
    assert 'trajectories.append(("current_to_idle", current_to_idle))' in runner
    assert 'trajectories.append(("return_to_start", return_to_start))' in runner
    assert "beginning_pose.joint_seed" in runner
    assert "/fr3/stop_motion" in runner
    assert "cancel_goal_async" in runner
    assert "retract[2, 3] +=" in runner


def test_preview_playback_joint_state_path_is_encoded():
    package_root = Path(__file__).resolve().parents[1]
    runner = (
        package_root / "src" / "fr3_sonopet_motion" / "motion_runner_node.py"
    ).read_text(encoding="utf-8")

    assert PREVIEW_JOINT_STATES_TOPIC == "/sonopet/preview/joint_states"
    assert "self._preview_joint_pub = self.create_publisher(" in runner
    assert "PREVIEW_JOINT_STATES_TOPIC" in runner
    assert "self._on_joint_state" in runner
    assert "if not preview_active:" in runner
    assert "if not execute:" in runner
    assert "self._publish_preview_playback(goal_handle, trajectories)" in runner
    assert "controller_goal = FollowJointTrajectory.Goal()" in runner
    assert "self._active_preview = False" in runner
    assert "self._active_preview = True" in runner
    assert "if self._stop_requested.is_set():" in runner
    assert "preview_active = self._active_preview" in runner
    assert "Preview stopped by operator." in runner


def test_vendor_joint_state_health_monitor_is_encoded():
    package_root = Path(__file__).resolve().parents[1]
    runner = (
        package_root / "src" / "fr3_sonopet_motion" / "motion_runner_node.py"
    ).read_text(encoding="utf-8")

    assert "JOINT_STATE_SILENCE_S = 0.5" in runner
    assert "JOINT_STATE_MONITOR_PERIOD_S = 0.1" in runner
    assert 'WAITING_FOR_JOINT_STATES = "waiting_for_joint_states"' in runner
    assert "self._joint_state_count = 0" in runner
    assert "self._joint_state_received_s: float | None = None" in runner
    assert "self._joint_states_available = False" in runner
    assert "self._joint_state_condition = Condition(self._state_lock)" in runner
    assert "self._recovery_active = False" in runner
    assert 'VENDOR_JOINT_STATE_STALE_TOPIC = "/sonopet/vendor_joint_state_stale"' in runner
    assert "self._vendor_stale_pub = self.create_publisher(" in runner
    assert "self._joint_state_monitor = self.create_timer(" in runner
    assert "def _monitor_joint_states(self) -> None:" in runner
    assert "monotonic() - self._joint_state_received_s <= JOINT_STATE_SILENCE_S" in runner
    assert "self._start_vendor_recovery()" in runner


def test_startup_tcp_tf_gap_keeps_vendor_joint_state_stale():
    package_root = Path(__file__).resolve().parents[1]
    runner = (
        package_root / "src" / "fr3_sonopet_motion" / "motion_runner_node.py"
    ).read_text(encoding="utf-8")

    assert "from tf2_ros import Buffer, TransformException, TransformListener" in runner
    assert "except TransformException as exc:" in runner
    assert "Waiting for Sonopet TCP TF before enabling motion" in runner
    assert "self._vendor_stale_pub.publish(Bool(data=True))" in runner
    assert "self._vendor_stale_pub.publish(Bool(data=False))" in runner
    stale_branch = runner.split("except TransformException as exc:", 1)[1].split(
        "with self._state_lock:\n            self._latest_joint_state = joint_state", 1
    )[0]
    assert "return" in stale_branch
    live_branch = runner.split("if beginning_pose_missing:", 1)[1].split(
        "if recovered:", 1
    )[0]
    assert "self._cache_beginning_pose(joint_state)" in live_branch
    assert "self._joint_states_available = True" in live_branch


def test_vendor_joint_state_silence_invalidates_only_live_state():
    package_root = Path(__file__).resolve().parents[1]
    runner = (
        package_root / "src" / "fr3_sonopet_motion" / "motion_runner_node.py"
    ).read_text(encoding="utf-8")

    assert "self._latest_joint_state = None" in runner
    monitor_body = runner.split("def _monitor_joint_states(self) -> None:", 1)[1].split(
        "def _start_vendor_recovery(self) -> None:", 1
    )[0]
    assert "self._beginning_pose = None" not in monitor_body
    assert "self._joint_states_available = False" in runner
    assert "self._joint_states_lost = True" in runner
    assert "self._vendor_stale_pub.publish(Bool(data=True))" in runner
    assert "self._vendor_stale_pub.publish(Bool(data=False))" in runner
    assert "Vendor joint states unavailable; waiting for robot mode recovery." in runner
    assert "Vendor joint states recovered." in runner


def test_active_franka_recovery_clients_and_constants_are_encoded():
    package_root = Path(__file__).resolve().parents[1]
    runner = (
        package_root / "src" / "fr3_sonopet_motion" / "motion_runner_node.py"
    ).read_text(encoding="utf-8")

    assert "from action_msgs.msg import GoalStatus" in runner
    assert (
        "from controller_manager_msgs.srv import SetHardwareComponentState, SwitchController"
        in runner
    )
    assert "from franka_msgs.action import ErrorRecovery" in runner
    assert "from lifecycle_msgs.msg import State" in runner
    assert 'FRANKA_ERROR_RECOVERY_ACTION = "/action_server/error_recovery"' in runner
    assert 'HARDWARE_STATE_SERVICE = "/controller_manager/set_hardware_component_state"' in runner
    assert 'SWITCH_CONTROLLER_SERVICE = "/controller_manager/switch_controller"' in runner
    assert 'FRANKA_HARDWARE_COMPONENT = "FrankaHardwareInterface"' in runner
    assert '"franka_robot_state_broadcaster"' in runner
    assert '"joint_state_broadcaster"' in runner
    assert '"fr3_arm_controller"' in runner
    assert "self._franka_recovery_client = ActionClient(" in runner
    assert "self._hardware_state_client = self.create_client(" in runner
    assert "self._switch_controller_client = self.create_client(" in runner


def test_active_franka_recovery_sequence_is_encoded():
    package_root = Path(__file__).resolve().parents[1]
    runner = (
        package_root / "src" / "fr3_sonopet_motion" / "motion_runner_node.py"
    ).read_text(encoding="utf-8")

    assert "def _start_vendor_recovery(self) -> None:" in runner
    assert "Thread(target=self._run_vendor_recovery, daemon=True).start()" in runner
    assert "def _run_vendor_recovery(self) -> None:" in runner
    assert "self._recover_vendor_stack_once()" in runner
    assert "def _recover_vendor_stack_once(self) -> None:" in runner
    assert "self._send_franka_error_recovery()" in runner
    assert "self._activate_franka_hardware()" in runner
    assert "self._activate_franka_controllers()" in runner
    assert "ErrorRecovery.Goal()" in runner
    assert "GoalStatus.STATUS_SUCCEEDED" in runner
    assert "State(id=State.PRIMARY_STATE_ACTIVE, label=\"active\")" in runner
    assert "request.activate_controllers = list(FRANKA_RECOVERY_CONTROLLERS)" in runner
    assert "request.strictness = SwitchController.Request.BEST_EFFORT" in runner
    assert "request.activate_asap = True" in runner


def test_motion_package_declares_active_recovery_dependencies():
    package_root = Path(__file__).resolve().parents[1]
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "<exec_depend>action_msgs</exec_depend>" in package_xml
    assert "<exec_depend>controller_manager_msgs</exec_depend>" in package_xml
    assert "<exec_depend>franka_msgs</exec_depend>" in package_xml
    assert "<exec_depend>lifecycle_msgs</exec_depend>" in package_xml
    assert "<exec_depend>std_srvs</exec_depend>" in package_xml


def test_stop_recovery_return_timing_uses_beginning_pose_distance():
    package_root = Path(__file__).resolve().parents[1]
    runner = (
        package_root / "src" / "fr3_sonopet_motion" / "motion_runner_node.py"
    ).read_text(encoding="utf-8")

    assert "or self._beginning_pose is None" in runner
    assert 'motion_speed_m_s = float(self.get_parameter("motion_speed_m_s").value)' in runner
    assert "return_distance_m = float(" in runner
    assert "retract[:3, 3] - beginning_pose.base_from_tcp[:3, 3]" in runner
    assert "max(return_distance_m / motion_speed_m_s, 0.05)" in runner


def test_motion_actions_wait_for_fresh_vendor_joint_states():
    package_root = Path(__file__).resolve().parents[1]
    runner = (
        package_root / "src" / "fr3_sonopet_motion" / "motion_runner_node.py"
    ).read_text(encoding="utf-8")

    assert "def _wait_for_vendor_joint_state(self, goal_handle, feedback_kind: str)" in runner
    assert "PreviewMotion.Feedback(phase=WAITING_FOR_JOINT_STATES)" in runner
    assert "ExecuteMotion.Feedback(active_segment=WAITING_FOR_JOINT_STATES)" in runner
    assert "StopMotion.Feedback(phase=WAITING_FOR_JOINT_STATES)" in runner
    assert 'self._wait_for_vendor_joint_state(goal_handle, "stop")' in runner
    assert '"execute" if execute else "preview"' in runner
    assert "Motion stopped while waiting for vendor joint states." in runner


def test_controller_retry_waits_for_vendor_joint_state_recovery():
    package_root = Path(__file__).resolve().parents[1]
    runner = (
        package_root / "src" / "fr3_sonopet_motion" / "motion_runner_node.py"
    ).read_text(encoding="utf-8")

    assert "while True:" in runner
    assert 'self._wait_for_vendor_joint_state(goal_handle, "execute")' in runner
    assert "controller_result = self._send_controller_trajectory(trajectory)" in runner
    assert "observed_count = self._joint_state_count" in runner
    assert "if not self._controller_failure_matches_joint_state_loss(observed_count):" in runner
    assert "raise RuntimeError(MOTION_VENDOR_ERROR)" in runner
    assert "def _send_controller_trajectory(self, trajectory: JointTrajectory) -> int:" in runner
    assert "return controller_result.error_code" in runner
    assert (
        "def _controller_failure_matches_joint_state_loss(self, observed_count: int) -> bool:"
        in runner
    )
    assert "return not self._joint_states_available" in runner


def test_joint_names_remain_fr3_vendor_joint_state_names():
    assert JOINT_NAMES == (
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    )


def test_cutting_trigger_is_encoded_for_raster_execution_only():
    package_root = Path(__file__).resolve().parents[1]
    runner = (
        package_root / "src" / "fr3_sonopet_motion" / "motion_runner_node.py"
    ).read_text(encoding="utf-8")

    assert "from std_msgs.msg import Bool" in runner
    assert 'CUTTING_TOPIC = "/sonopet/cutting"' in runner
    assert "self._cutting_pub = self.create_publisher(Bool, CUTTING_TOPIC, 10)" in runner
    assert "def _publish_cutting(self, enabled: bool) -> None:" in runner
    assert "self._cutting_pub.publish(Bool(data=enabled))" in runner
    assert 'if segment_name == "raster":' in runner
    assert "self._publish_cutting(True)" in runner
    assert "self._publish_cutting(False)" in runner
    assert "finally:" in runner
    assert "if cutting_active:" in runner
    assert "self._publish_preview_playback(goal_handle, trajectories)" in runner

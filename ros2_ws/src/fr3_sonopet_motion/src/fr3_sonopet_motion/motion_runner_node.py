from __future__ import annotations

import math
from threading import Condition, Event, RLock, Thread
from time import monotonic

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SetHardwareComponentState, SwitchController
from fr3_sonopet_interfaces.action import ExecuteMotion, PreviewMotion, StopMotion
from fr3_sonopet_interfaces.msg import RasterPlan
from franka_msgs.action import ErrorRecovery
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from tf2_ros import Buffer, TransformListener

from fr3_sonopet_motion.motion_geometry import (
    BASE_FRAME,
    CONTROLLER_ACTION,
    FK_SERVICE,
    IK_LINK_FRAME,
    IK_SERVICE,
    IK_TIMEOUT_S,
    JOINT_NAMES,
    PLANNING_GROUP,
    PREVIEW_JOINT_STATES_TOPIC,
    RASTER_PLAN_TOPIC,
    TOOL_FRAME,
    build_cartesian_segments,
    densified_matrices,
    joint_interpolation_points,
    joint_trajectory_points,
    quaternion_from_matrix,
    rotation_matrix_from_quaternion,
    staged_cartesian_segments,
)
from fr3_sonopet_motion.operator_policy import EXECUTE_TOKEN

MOTION_INPUT_ERROR = "Motion runner input is incomplete or inconsistent."
MOTION_VENDOR_ERROR = "Motion vendor command failed."
JOINT_STATE_SILENCE_S = 0.5
JOINT_STATE_MONITOR_PERIOD_S = 0.1
WAITING_FOR_JOINT_STATES = "waiting_for_joint_states"
FRANKA_ERROR_RECOVERY_ACTION = "/action_server/error_recovery"
HARDWARE_STATE_SERVICE = "/controller_manager/set_hardware_component_state"
SWITCH_CONTROLLER_SERVICE = "/controller_manager/switch_controller"
FRANKA_HARDWARE_COMPONENT = "FrankaHardwareInterface"
FRANKA_RECOVERY_CONTROLLERS = (
    "franka_robot_state_broadcaster",
    "joint_state_broadcaster",
    "fr3_arm_controller",
)


class MotionRunnerNode(Node):
    """Compile raster plans against the live FR3 pose and execute them through ros2_control."""

    def __init__(self) -> None:
        super().__init__("motion_runner_node")
        self.declare_parameter("idle_joint_positions")
        self.declare_parameter("motion_speed_m_s")
        self.declare_parameter("raster_speed_m_s")
        self.declare_parameter("parking_lift_m")
        self._latest_plan: RasterPlan | None = None
        self._latest_joint_state: JointState | None = None
        self._joint_state_count = 0
        self._joint_state_received_s: float | None = None
        self._joint_states_available = False
        self._joint_states_lost = False
        self._recovery_active = False
        self._beginning_seed: dict[str, float] | None = None
        self._beginning_base_from_tcp: np.ndarray | None = None
        self._beginning_link_from_tcp: np.ndarray | None = None
        self._active_execute = False
        self._active_preview = False
        self._active_controller_handle = None
        self._controller_done = Event()
        self._controller_done.set()
        self._stop_requested = Event()
        self._state_lock = RLock()
        self._joint_state_condition = Condition(self._state_lock)
        self._callback_group = ReentrantCallbackGroup()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # Cache the latest raster plan and robot joints so actions always compile from live inputs.
        self._preview_joint_pub = self.create_publisher(
            JointState,
            PREVIEW_JOINT_STATES_TOPIC,
            10,
        )
        self._plan_sub = self.create_subscription(
            RasterPlan,
            RASTER_PLAN_TOPIC,
            lambda plan: (
                setattr(self, "_latest_plan", plan),
                setattr(self, "_beginning_seed", None),
                setattr(self, "_beginning_base_from_tcp", None),
                setattr(self, "_beginning_link_from_tcp", None),
            ),
            10,
            callback_group=self._callback_group,
        )
        self._joint_state_monitor = self.create_timer(
            JOINT_STATE_MONITOR_PERIOD_S,
            self._monitor_joint_states,
            callback_group=self._callback_group,
        )
        self._joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            10,
            callback_group=self._callback_group,
        )
        # MoveIt solves Cartesian targets into FR3 joint targets; the arm controller executes them.
        self._ik_client = self.create_client(
            GetPositionIK,
            IK_SERVICE,
            callback_group=self._callback_group,
        )
        self._fk_client = self.create_client(
            GetPositionFK,
            FK_SERVICE,
            callback_group=self._callback_group,
        )
        self._controller_client = ActionClient(
            self,
            FollowJointTrajectory,
            CONTROLLER_ACTION,
            callback_group=self._callback_group,
        )
        self._franka_recovery_client = ActionClient(
            self,
            ErrorRecovery,
            FRANKA_ERROR_RECOVERY_ACTION,
            callback_group=self._callback_group,
        )
        self._hardware_state_client = self.create_client(
            SetHardwareComponentState,
            HARDWARE_STATE_SERVICE,
            callback_group=self._callback_group,
        )
        self._switch_controller_client = self.create_client(
            SwitchController,
            SWITCH_CONTROLLER_SERVICE,
            callback_group=self._callback_group,
        )
        # Preview compiles only; execute compiles the same path and streams it to the vendor controller.
        self._preview_server = ActionServer(
            self,
            PreviewMotion,
            "/sonopet/preview_motion",
            lambda goal_handle: self._run_motion(goal_handle, execute=False),
            callback_group=self._callback_group,
        )
        self._execute_server = ActionServer(
            self,
            ExecuteMotion,
            "/sonopet/execute_motion",
            lambda goal_handle: self._run_motion(goal_handle, execute=True),
            callback_group=self._callback_group,
        )
        self._stop_server = ActionServer(
            self,
            StopMotion,
            "/sonopet/stop_motion",
            self._run_stop_recovery,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            "Motion runner ready: "
            f"base={BASE_FRAME}, tool={TOOL_FRAME}, ik_link={IK_LINK_FRAME}, "
            f"controller={CONTROLLER_ACTION}"
        )

    def _on_joint_state(self, joint_state: JointState) -> None:
        # Idle preview TF mirrors the live arm until playback takes ownership of the preview stream.
        with self._state_lock:
            recovered = self._joint_states_lost
            self._latest_joint_state = joint_state
            self._joint_state_count += 1
            self._joint_state_received_s = monotonic()
            self._joint_states_available = True
            self._joint_states_lost = False
            preview_active = self._active_preview
            self._joint_state_condition.notify_all()
        if recovered:
            self.get_logger().info("Vendor joint states recovered.")
        with self._state_lock:
            self._recovery_active = False
        if not preview_active:
            self._preview_joint_pub.publish(joint_state)

    def _monitor_joint_states(self) -> None:
        # Robot mode switches are detected passively by the vendor joint-state stream going silent.
        with self._state_lock:
            if self._joint_state_received_s is None or not self._joint_states_available:
                return
            if monotonic() - self._joint_state_received_s <= JOINT_STATE_SILENCE_S:
                return
            self._latest_joint_state = None
            self._beginning_seed = None
            self._beginning_base_from_tcp = None
            self._beginning_link_from_tcp = None
            self._joint_states_available = False
            self._joint_states_lost = True
            self._joint_state_condition.notify_all()
        self.get_logger().warn("Vendor joint states unavailable; waiting for robot mode recovery.")
        self._start_vendor_recovery()

    def _start_vendor_recovery(self) -> None:
        with self._state_lock:
            if self._recovery_active:
                return
            self._recovery_active = True
        Thread(target=self._run_vendor_recovery, daemon=True).start()

    def _run_vendor_recovery(self) -> None:
        while True:
            with self._state_lock:
                if self._joint_states_available and self._latest_joint_state is not None:
                    self._recovery_active = False
                    return
                observed_count = self._joint_state_count
            try:
                self._recover_vendor_stack_once()
            except Exception as exc:
                self.get_logger().warn(f"Franka recovery attempt failed: {exc}")
            with self._joint_state_condition:
                if (
                    self._joint_states_available
                    and self._latest_joint_state is not None
                    and self._joint_state_count > observed_count
                ):
                    self._recovery_active = False
                    return
                self._joint_state_condition.wait(JOINT_STATE_SILENCE_S)

    def _recover_vendor_stack_once(self) -> None:
        # Recovery follows the vendor boundary first, then restores the launch-time controllers.
        self._send_franka_error_recovery()
        self._activate_franka_hardware()
        self._activate_franka_controllers()

    def _send_franka_error_recovery(self) -> None:
        if not self._franka_recovery_client.wait_for_server(timeout_sec=JOINT_STATE_SILENCE_S):
            raise RuntimeError(f"{FRANKA_ERROR_RECOVERY_ACTION} is unavailable")
        goal_handle = wait_future(self._franka_recovery_client.send_goal_async(ErrorRecovery.Goal()))
        if not goal_handle.accepted:
            raise RuntimeError("Franka error recovery goal was rejected")
        recovery_result = wait_future(goal_handle.get_result_async())
        if recovery_result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError("Franka error recovery did not succeed")

    def _activate_franka_hardware(self) -> None:
        if not self._hardware_state_client.wait_for_service(timeout_sec=JOINT_STATE_SILENCE_S):
            raise RuntimeError(f"{HARDWARE_STATE_SERVICE} is unavailable")
        request = SetHardwareComponentState.Request()
        request.name = FRANKA_HARDWARE_COMPONENT
        request.target_state = State(id=State.PRIMARY_STATE_ACTIVE, label="active")
        response = wait_future(self._hardware_state_client.call_async(request))
        if not response.ok:
            raise RuntimeError(f"{FRANKA_HARDWARE_COMPONENT} did not activate")

    def _activate_franka_controllers(self) -> None:
        if not self._switch_controller_client.wait_for_service(timeout_sec=JOINT_STATE_SILENCE_S):
            raise RuntimeError(f"{SWITCH_CONTROLLER_SERVICE} is unavailable")
        request = SwitchController.Request()
        request.activate_controllers = list(FRANKA_RECOVERY_CONTROLLERS)
        request.deactivate_controllers = []
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout = Duration(sec=1)
        response = wait_future(self._switch_controller_client.call_async(request))
        if not response.ok:
            raise RuntimeError(f"Controller activation failed: {response.message}")

    def _wait_for_vendor_joint_state(self, goal_handle, feedback_kind: str) -> JointState | None:
        # Motion always compiles from a current physical robot state, never from a stale cache.
        with self._state_lock:
            if self._joint_states_available and self._latest_joint_state is not None:
                return self._latest_joint_state
            observed_count = self._joint_state_count
        while True:
            if feedback_kind == "preview":
                goal_handle.publish_feedback(PreviewMotion.Feedback(phase=WAITING_FOR_JOINT_STATES))
            elif feedback_kind == "execute":
                goal_handle.publish_feedback(
                    ExecuteMotion.Feedback(active_segment=WAITING_FOR_JOINT_STATES)
                )
            elif feedback_kind == "stop":
                goal_handle.publish_feedback(StopMotion.Feedback(phase=WAITING_FOR_JOINT_STATES))
            with self._joint_state_condition:
                if (
                    self._joint_states_available
                    and self._latest_joint_state is not None
                    and self._joint_state_count > observed_count
                ):
                    return self._latest_joint_state
                if feedback_kind in ("preview", "execute") and self._stop_requested.is_set():
                    return None
                self._joint_state_condition.wait(JOINT_STATE_MONITOR_PERIOD_S)

    def _run_motion(self, goal_handle, execute: bool):
        """Compile the latest raster plan and optionally stream each segment to the FR3 controller."""
        result = ExecuteMotion.Result() if execute else PreviewMotion.Result()
        if execute and goal_handle.request.confirmation_token.strip() != EXECUTE_TOKEN:
            goal_handle.abort()
            result.success = False
            result.message = "Execution rejected because the confirmation token is invalid."
            return result

        feedback = ExecuteMotion.Feedback() if execute else PreviewMotion.Feedback()
        if execute:
            with self._state_lock:
                self._active_execute = True
                self._stop_requested.clear()
            feedback.active_segment = "compile"
        else:
            with self._state_lock:
                self._active_preview = True
                self._stop_requested.clear()
            feedback.phase = "compile"
        goal_handle.publish_feedback(feedback)

        if (
            self._latest_plan is None
            or self._latest_plan.header.frame_id != BASE_FRAME
            or len(self._latest_plan.poses.poses) == 0
        ):
            raise RuntimeError(MOTION_INPUT_ERROR)
        joint_state = self._wait_for_vendor_joint_state(
            goal_handle,
            "execute" if execute else "preview",
        )
        if joint_state is None:
            with self._state_lock:
                self._active_execute = False
                self._active_preview = False
                self._stop_requested.clear()
            goal_handle.abort()
            result.success = False
            result.message = "Motion stopped while waiting for vendor joint states."
            return result

        self._ik_client.wait_for_service()
        self._fk_client.wait_for_service()
        if execute:
            self._controller_client.wait_for_server()

        # The first preview or execute after a new raster plan freezes the scan's starting pose.
        if self._beginning_seed is None:
            seed_state = dict(zip(joint_state.name, joint_state.position, strict=True))
            base_from_tcp_msg = self._tf_buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, Time()).transform
            base_from_link_msg = self._tf_buffer.lookup_transform(
                BASE_FRAME, IK_LINK_FRAME, Time()
            ).transform
            base_from_tcp = np.eye(4, dtype=np.float64)
            base_from_link = np.eye(4, dtype=np.float64)
            base_from_tcp[:3, :3] = rotation_matrix_from_quaternion(
                np.array(
                    [
                        base_from_tcp_msg.rotation.x,
                        base_from_tcp_msg.rotation.y,
                        base_from_tcp_msg.rotation.z,
                        base_from_tcp_msg.rotation.w,
                    ],
                    dtype=np.float64,
                )
            )
            base_from_link[:3, :3] = rotation_matrix_from_quaternion(
                np.array(
                    [
                        base_from_link_msg.rotation.x,
                        base_from_link_msg.rotation.y,
                        base_from_link_msg.rotation.z,
                        base_from_link_msg.rotation.w,
                    ],
                    dtype=np.float64,
                )
            )
            base_from_tcp[:3, 3] = np.array(
                [
                    base_from_tcp_msg.translation.x,
                    base_from_tcp_msg.translation.y,
                    base_from_tcp_msg.translation.z,
                ],
                dtype=np.float64,
            )
            base_from_link[:3, 3] = np.array(
                [
                    base_from_link_msg.translation.x,
                    base_from_link_msg.translation.y,
                    base_from_link_msg.translation.z,
                ],
                dtype=np.float64,
            )
            self._beginning_seed = {name: float(seed_state[name]) for name in JOINT_NAMES}
            self._beginning_base_from_tcp = base_from_tcp
            self._beginning_link_from_tcp = np.linalg.inv(base_from_link) @ base_from_tcp

        seed = dict(self._beginning_seed)
        base_from_tcp = self._beginning_base_from_tcp
        link_from_tcp = self._beginning_link_from_tcp
        idle_values = list(self.get_parameter("idle_joint_positions").value)
        idle_seed = {name: float(idle_values[index]) for index, name in enumerate(JOINT_NAMES)}
        motion_speed_m_s = float(self.get_parameter("motion_speed_m_s").value)
        raster_speed_m_s = float(self.get_parameter("raster_speed_m_s").value)
        parking_lift_m = float(self.get_parameter("parking_lift_m").value)

        fk_request = GetPositionFK.Request()
        fk_request.header.frame_id = BASE_FRAME
        fk_request.fk_link_names = [IK_LINK_FRAME]
        fk_request.robot_state.joint_state.name = list(JOINT_NAMES)
        fk_request.robot_state.joint_state.position = [float(idle_seed[name]) for name in JOINT_NAMES]
        fk_response = wait_future(self._fk_client.call_async(fk_request))
        if fk_response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(MOTION_VENDOR_ERROR)
        idle_link_pose = fk_response.pose_stamped[0].pose
        idle_base_from_link = np.eye(4, dtype=np.float64)
        idle_base_from_link[:3, :3] = rotation_matrix_from_quaternion(
            np.array(
                [
                    idle_link_pose.orientation.x,
                    idle_link_pose.orientation.y,
                    idle_link_pose.orientation.z,
                    idle_link_pose.orientation.w,
                ],
                dtype=np.float64,
            )
        )
        idle_base_from_link[:3, 3] = np.array(
            [idle_link_pose.position.x, idle_link_pose.position.y, idle_link_pose.position.z],
            dtype=np.float64,
        )

        trajectories: list[tuple[str, JointTrajectory]] = []
        current_to_idle = JointTrajectory()
        current_to_idle.joint_names = list(JOINT_NAMES)
        current_to_idle.points = joint_interpolation_points(
            seed,
            idle_seed,
            max(
                float(np.linalg.norm((idle_base_from_link @ link_from_tcp)[:3, 3] - base_from_tcp[:3, 3]))
                / motion_speed_m_s,
                0.05,
            ),
        )
        trajectories.append(("current_to_idle", current_to_idle))
        seed = dict(idle_seed)

        raster = build_cartesian_segments(self._latest_plan, base_from_tcp)
        for segment in staged_cartesian_segments(
            idle_base_from_link @ link_from_tcp,
            raster,
            motion_speed_m_s,
            raster_speed_m_s,
            parking_lift_m,
        ):
            solved_points: list[dict[str, float]] = []
            for index, tcp_matrix in enumerate(segment.tcp_matrices):
                if segment.name == "idle_to_parking" and index == 0:
                    seed = dict(idle_seed)
                    solved_points.append(dict(seed))
                    continue
                base_from_ik_link = tcp_matrix @ np.linalg.inv(link_from_tcp)
                sec = int(math.floor(IK_TIMEOUT_S))
                request = GetPositionIK.Request()
                request.ik_request.group_name = PLANNING_GROUP
                request.ik_request.ik_link_name = IK_LINK_FRAME
                request.ik_request.pose_stamped = PoseStamped()
                request.ik_request.pose_stamped.header.frame_id = BASE_FRAME
                request.ik_request.pose_stamped.pose.position.x = float(base_from_ik_link[0, 3])
                request.ik_request.pose_stamped.pose.position.y = float(base_from_ik_link[1, 3])
                request.ik_request.pose_stamped.pose.position.z = float(base_from_ik_link[2, 3])
                request.ik_request.pose_stamped.pose.orientation = quaternion_from_matrix(
                    base_from_ik_link
                )
                request.ik_request.robot_state.joint_state.name = list(JOINT_NAMES)
                request.ik_request.robot_state.joint_state.position = [
                    float(seed[name]) for name in JOINT_NAMES
                ]
                request.ik_request.avoid_collisions = True
                request.ik_request.timeout = Duration(
                    sec=sec,
                    nanosec=int(round((IK_TIMEOUT_S - float(sec)) * 1_000_000_000.0)),
                )
                response = wait_future(self._ik_client.call_async(request))
                if response.error_code.val != MoveItErrorCodes.SUCCESS:
                    raise RuntimeError(MOTION_VENDOR_ERROR)
                solution_state = dict(
                    zip(
                        response.solution.joint_state.name,
                        response.solution.joint_state.position,
                        strict=True,
                    )
                )
                seed = {name: float(solution_state[name]) for name in JOINT_NAMES}
                solved_points.append(dict(seed))
            trajectory = JointTrajectory()
            trajectory.joint_names = list(JOINT_NAMES)
            trajectory.points = joint_trajectory_points(
                solved_points,
                segment.tcp_matrices,
                segment.speed_m_s,
            )
            trajectories.append((segment.name, trajectory))

        return_to_start = JointTrajectory()
        return_to_start.joint_names = list(JOINT_NAMES)
        return_to_start.points = joint_interpolation_points(
            seed,
            self._beginning_seed,
            max(
                float(np.linalg.norm(raster[-1][:3, 3] - base_from_tcp[:3, 3])) / motion_speed_m_s,
                0.05,
            ),
        )
        trajectories.append(("return_to_start", return_to_start))

        if not execute:
            if not self._publish_preview_playback(goal_handle, trajectories):
                with self._state_lock:
                    self._active_preview = False
                    self._stop_requested.clear()
                goal_handle.abort()
                result.success = False
                result.message = "Preview stopped by operator."
                return result

        if execute:
            for segment_name, trajectory in trajectories:
                if self._stop_requested.is_set():
                    with self._state_lock:
                        self._active_execute = False
                    goal_handle.abort()
                    result.success = False
                    result.message = "Execution stopped by operator; recovery is handled by the stop action."
                    return result
                feedback = ExecuteMotion.Feedback()
                feedback.active_segment = segment_name
                goal_handle.publish_feedback(feedback)
                while True:
                    if self._wait_for_vendor_joint_state(goal_handle, "execute") is None:
                        with self._state_lock:
                            self._active_execute = False
                        goal_handle.abort()
                        result.success = False
                        result.message = (
                            "Execution stopped by operator; recovery is handled by the stop action."
                        )
                        return result
                    controller_result = self._send_controller_trajectory(trajectory)
                    if self._stop_requested.is_set():
                        with self._state_lock:
                            self._active_execute = False
                        goal_handle.abort()
                        result.success = False
                        result.message = (
                            "Execution stopped by operator; recovery is handled by the stop action."
                        )
                        return result
                    if controller_result == FollowJointTrajectory.Result.SUCCESSFUL:
                        break
                    with self._state_lock:
                        observed_count = self._joint_state_count
                    if not self._controller_failure_matches_joint_state_loss(observed_count):
                        raise RuntimeError(MOTION_VENDOR_ERROR)

        goal_handle.succeed()
        result.success = True
        verb = "Executed" if execute else "Preview played"
        result.message = (
            f"{verb} {len(trajectories)} segments and "
            f"{sum(len(trajectory.points) for _, trajectory in trajectories)} joint waypoints."
        )
        if execute:
            with self._state_lock:
                self._active_execute = False
        else:
            with self._state_lock:
                self._active_preview = False
                self._stop_requested.clear()
        return result

    def _send_controller_trajectory(self, trajectory: JointTrajectory) -> int:
        controller_goal = FollowJointTrajectory.Goal()
        controller_goal.trajectory = trajectory
        self._controller_done.clear()
        controller_handle = wait_future(self._controller_client.send_goal_async(controller_goal))
        if not controller_handle.accepted:
            self._controller_done.set()
            with self._state_lock:
                self._active_controller_handle = None
                unavailable = not self._joint_states_available
            if unavailable:
                return FollowJointTrajectory.Result.INVALID_GOAL
            raise RuntimeError(MOTION_VENDOR_ERROR)
        with self._state_lock:
            self._active_controller_handle = controller_handle
        controller_result = wait_future(controller_handle.get_result_async()).result
        with self._state_lock:
            self._active_controller_handle = None
        self._controller_done.set()
        return controller_result.error_code

    def _controller_failure_matches_joint_state_loss(self, observed_count: int) -> bool:
        # A vendor abort is retried only when the physical joint-state stream also goes silent.
        deadline_s = monotonic() + JOINT_STATE_SILENCE_S
        with self._joint_state_condition:
            while self._joint_states_available and monotonic() < deadline_s:
                if self._joint_state_count > observed_count:
                    return False
                self._joint_state_condition.wait(
                    max(min(JOINT_STATE_MONITOR_PERIOD_S, deadline_s - monotonic()), 0.0)
                )
            return not self._joint_states_available

    def _publish_preview_playback(self, goal_handle, trajectories: list[tuple[str, JointTrajectory]]) -> bool:
        for segment_name, trajectory in trajectories:
            feedback = PreviewMotion.Feedback(phase=segment_name)
            goal_handle.publish_feedback(feedback)
            previous_s: float | None = None
            for point in trajectory.points:
                if self._stop_requested.is_set():
                    return False
                point_s = float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9
                if previous_s is not None:
                    Event().wait(max(point_s - previous_s, 0.0))
                if self._stop_requested.is_set():
                    return False
                joint_state = JointState()
                joint_state.header.stamp = self.get_clock().now().to_msg()
                joint_state.name = list(JOINT_NAMES)
                joint_state.position = list(point.positions)
                self._preview_joint_pub.publish(joint_state)
                previous_s = point_s
        return True

    def _run_stop_recovery(self, goal_handle):
        result = StopMotion.Result()
        with self._state_lock:
            preview_active = self._active_preview
            active = self._active_execute
            controller_handle = self._active_controller_handle
        if preview_active:
            self._stop_requested.set()
            goal_handle.succeed()
            result.success = True
            result.message = "Preview stopped by operator."
            return result
        if not active:
            goal_handle.succeed()
            result.success = True
            result.message = "No active motion to stop."
            return result

        feedback = StopMotion.Feedback(phase="cancel_current")
        goal_handle.publish_feedback(feedback)
        self._stop_requested.set()
        if controller_handle is not None:
            wait_future(controller_handle.cancel_goal_async())
        self._controller_done.wait()

        feedback.phase = "retract"
        goal_handle.publish_feedback(feedback)
        self._wait_for_vendor_joint_state(goal_handle, "stop")
        for trajectory in self._compute_recovery_trajectory():
            controller_goal = FollowJointTrajectory.Goal()
            controller_goal.trajectory = trajectory
            controller_handle = wait_future(self._controller_client.send_goal_async(controller_goal))
            if not controller_handle.accepted:
                raise RuntimeError(MOTION_VENDOR_ERROR)
            controller_result = wait_future(controller_handle.get_result_async()).result
            if controller_result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                raise RuntimeError(MOTION_VENDOR_ERROR)

        with self._state_lock:
            self._active_execute = False
            self._stop_requested.clear()
        goal_handle.succeed()
        result.success = True
        result.message = "Motion stopped, retracted, and returned to the beginning pose."
        return result

    def _compute_recovery_trajectory(self) -> list[JointTrajectory]:
        if self._latest_joint_state is None or self._beginning_seed is None or self._beginning_link_from_tcp is None:
            raise RuntimeError(MOTION_INPUT_ERROR)
        self._ik_client.wait_for_service()
        self._controller_client.wait_for_server()
        current_seed = dict(zip(self._latest_joint_state.name, self._latest_joint_state.position, strict=True))
        seed = {name: float(current_seed[name]) for name in JOINT_NAMES}
        base_from_tcp_msg = self._tf_buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, Time()).transform
        base_from_tcp = np.eye(4, dtype=np.float64)
        base_from_tcp[:3, :3] = rotation_matrix_from_quaternion(
            np.array(
                [
                    base_from_tcp_msg.rotation.x,
                    base_from_tcp_msg.rotation.y,
                    base_from_tcp_msg.rotation.z,
                    base_from_tcp_msg.rotation.w,
                ],
                dtype=np.float64,
            )
        )
        base_from_tcp[:3, 3] = np.array(
            [
                base_from_tcp_msg.translation.x,
                base_from_tcp_msg.translation.y,
                base_from_tcp_msg.translation.z,
            ],
            dtype=np.float64,
        )
        retract = np.array(base_from_tcp, dtype=np.float64, copy=True)
        retract[2, 3] += float(self.get_parameter("parking_lift_m").value)
        solved_points: list[dict[str, float]] = []
        for tcp_matrix in densified_matrices([base_from_tcp, retract]):
            base_from_ik_link = tcp_matrix @ np.linalg.inv(self._beginning_link_from_tcp)
            sec = int(math.floor(IK_TIMEOUT_S))
            request = GetPositionIK.Request()
            request.ik_request.group_name = PLANNING_GROUP
            request.ik_request.ik_link_name = IK_LINK_FRAME
            request.ik_request.pose_stamped = PoseStamped()
            request.ik_request.pose_stamped.header.frame_id = BASE_FRAME
            request.ik_request.pose_stamped.pose.position.x = float(base_from_ik_link[0, 3])
            request.ik_request.pose_stamped.pose.position.y = float(base_from_ik_link[1, 3])
            request.ik_request.pose_stamped.pose.position.z = float(base_from_ik_link[2, 3])
            request.ik_request.pose_stamped.pose.orientation = quaternion_from_matrix(base_from_ik_link)
            request.ik_request.robot_state.joint_state.name = list(JOINT_NAMES)
            request.ik_request.robot_state.joint_state.position = [float(seed[name]) for name in JOINT_NAMES]
            request.ik_request.avoid_collisions = True
            request.ik_request.timeout = Duration(
                sec=sec,
                nanosec=int(round((IK_TIMEOUT_S - float(sec)) * 1_000_000_000.0)),
            )
            response = wait_future(self._ik_client.call_async(request))
            if response.error_code.val != MoveItErrorCodes.SUCCESS:
                raise RuntimeError(MOTION_VENDOR_ERROR)
            solution_state = dict(
                zip(response.solution.joint_state.name, response.solution.joint_state.position, strict=True)
            )
            seed = {name: float(solution_state[name]) for name in JOINT_NAMES}
            solved_points.append(dict(seed))
        retract_trajectory = JointTrajectory()
        retract_trajectory.joint_names = list(JOINT_NAMES)
        retract_trajectory.points = joint_trajectory_points(
            solved_points,
            densified_matrices([base_from_tcp, retract]),
            float(self.get_parameter("motion_speed_m_s").value),
        )
        return_trajectory = JointTrajectory()
        return_trajectory.joint_names = list(JOINT_NAMES)
        return_trajectory.points = joint_interpolation_points(
            seed,
            self._beginning_seed,
            max(float(np.linalg.norm(retract[:3, 3] - base_from_tcp[:3, 3])) / float(self.get_parameter("motion_speed_m_s").value), 0.05),
        )
        return [retract_trajectory, return_trajectory]


def wait_future(future):
    """keeps the motion sequence linear: 
    call FK/IK/controller, wait for the answer, then proceed to the next waypoint or segment."""
    event = Event()
    future.add_done_callback(lambda _future: event.set())
    event.wait()
    return future.result()


def main() -> None:
    rclpy.init()
    node = MotionRunnerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

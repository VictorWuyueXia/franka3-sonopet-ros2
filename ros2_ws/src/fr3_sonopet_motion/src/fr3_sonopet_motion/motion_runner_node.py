from __future__ import annotations

import math
from dataclasses import dataclass
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
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory

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
STOP_RECOVERY_MESSAGE = (
    "Execution stopped by operator; recovery is handled by the stop action."
)
JOINT_STATE_SILENCE_S = 0.5
JOINT_STATE_MONITOR_PERIOD_S = 0.1
WAITING_FOR_JOINT_STATES = "waiting_for_joint_states"
FRANKA_ERROR_RECOVERY_ACTION = "/action_server/error_recovery"
HARDWARE_STATE_SERVICE = "/controller_manager/set_hardware_component_state"
SWITCH_CONTROLLER_SERVICE = "/controller_manager/switch_controller"
FRANKA_HARDWARE_COMPONENT = "FrankaHardwareInterface"
CUTTING_TOPIC = "/sonopet/cutting"
VENDOR_JOINT_STATE_STALE_TOPIC = "/sonopet/vendor_joint_state_stale"
SET_BEGINNING_POSE_SERVICE = "/fr3/set_beginning_pose"
FRANKA_RECOVERY_CONTROLLERS = (
    "franka_robot_state_broadcaster",
    "joint_state_broadcaster",
    "fr3_arm_controller",
)


@dataclass(frozen=True)
class BeginningPose:
    """Frozen joint and TCP reference used for all raster orientation and return motion."""

    joint_seed: dict[str, float]
    base_from_tcp: np.ndarray
    link_from_tcp: np.ndarray


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
        self._waiting_for_tool_tf = False
        self._recovery_active = False
        self._beginning_pose: BeginningPose | None = None
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
        self._cutting_pub = self.create_publisher(Bool, CUTTING_TOPIC, 10)
        self._vendor_stale_pub = self.create_publisher(
            Bool,
            VENDOR_JOINT_STATE_STALE_TOPIC,
            10,
        )
        self._plan_sub = self.create_subscription(
            RasterPlan,
            RASTER_PLAN_TOPIC,
            lambda plan: setattr(self, "_latest_plan", plan),
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
        self._set_beginning_pose_service = self.create_service(
            Trigger,
            SET_BEGINNING_POSE_SERVICE,
            self._set_beginning_pose,
            callback_group=self._callback_group,
        )
        # Preview compiles only; execute compiles and streams through the vendor controller.
        self._preview_server = ActionServer(
            self,
            PreviewMotion,
            "/fr3/preview_motion",
            lambda goal_handle: self._run_motion(goal_handle, execute=False),
            callback_group=self._callback_group,
        )
        self._execute_server = ActionServer(
            self,
            ExecuteMotion,
            "/fr3/execute_motion",
            lambda goal_handle: self._run_motion(goal_handle, execute=True),
            callback_group=self._callback_group,
        )
        self._stop_server = ActionServer(
            self,
            StopMotion,
            "/fr3/stop_motion",
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
            beginning_pose_missing = self._beginning_pose is None
        if beginning_pose_missing:
            try:
                self._cache_beginning_pose(joint_state)
            except TransformException as exc:
                with self._state_lock:
                    self._joint_states_available = False
                    self._joint_states_lost = True
                    first_wait = not self._waiting_for_tool_tf
                    self._waiting_for_tool_tf = True
                    self._joint_state_condition.notify_all()
                if first_wait:
                    self.get_logger().warn(
                        f"Waiting for Sonopet TCP TF before enabling motion: {exc}"
                    )
                self._vendor_stale_pub.publish(Bool(data=True))
                return
        with self._state_lock:
            self._latest_joint_state = joint_state
            self._joint_state_count += 1
            self._joint_state_received_s = monotonic()
            self._joint_states_available = True
            self._joint_states_lost = False
            self._waiting_for_tool_tf = False
            preview_active = self._active_preview
            self._joint_state_condition.notify_all()
        if recovered:
            self.get_logger().info("Vendor joint states recovered.")
        self._vendor_stale_pub.publish(Bool(data=False))
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
            self._joint_states_available = False
            self._joint_states_lost = True
            self._joint_state_condition.notify_all()
        self.get_logger().warn("Vendor joint states unavailable; waiting for robot mode recovery.")
        self._vendor_stale_pub.publish(Bool(data=True))
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
        goal_handle = wait_future(
            self._franka_recovery_client.send_goal_async(ErrorRecovery.Goal())
        )
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
        request.strictness = SwitchController.Request.BEST_EFFORT
        request.activate_asap = True
        request.timeout = Duration(sec=1)
        response = wait_future(self._switch_controller_client.call_async(request))
        if not response.ok:
            raise RuntimeError(f"Controller activation failed: {response.message}")

    def _joint_state_is_fresh_locked(self) -> bool:
        return (
            self._joint_states_available
            and self._latest_joint_state is not None
            and self._joint_state_received_s is not None
            and monotonic() - self._joint_state_received_s <= JOINT_STATE_SILENCE_S
        )

    def _wait_for_vendor_joint_state(self, goal_handle, feedback_kind: str) -> JointState | None:
        # Motion always compiles from a current physical robot state, never from a stale cache.
        with self._state_lock:
            if self._joint_state_is_fresh_locked():
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
                    self._joint_state_is_fresh_locked()
                    and self._joint_state_count > observed_count
                ):
                    return self._latest_joint_state
                if feedback_kind in ("preview", "execute") and self._stop_requested.is_set():
                    return None
                self._joint_state_condition.wait(JOINT_STATE_MONITOR_PERIOD_S)

    def _set_beginning_pose(self, _request, response):
        # Operator pose setting is an explicit overwrite of the node-lifetime return target.
        try:
            with self._state_lock:
                if not self._joint_state_is_fresh_locked():
                    raise RuntimeError(
                        "Cannot set beginning pose: vendor joint state is unavailable"
                    )
                joint_state = self._latest_joint_state
            self._cache_beginning_pose(joint_state)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            return response
        response.success = True
        response.message = "Beginning pose updated from current robot state."
        return response

    def _cache_beginning_pose(self, joint_state: JointState) -> None:
        # The cached beginning pose binds joint return and TCP orientation to the same robot sample.
        beginning_pose = self._beginning_pose_from_joint_state(joint_state)
        with self._state_lock:
            self._beginning_pose = beginning_pose
        self.get_logger().info("Cached beginning pose from current robot state.")

    def _require_beginning_pose(self) -> BeginningPose:
        with self._state_lock:
            beginning_pose = self._beginning_pose
        if beginning_pose is None:
            raise RuntimeError("Beginning pose has not been cached")
        return beginning_pose

    def _beginning_pose_from_joint_state(self, joint_state: JointState) -> BeginningPose:
        seed_state = dict(zip(joint_state.name, joint_state.position, strict=True))
        base_from_tcp = self._lookup_transform_matrix(BASE_FRAME, TOOL_FRAME)
        base_from_link = self._lookup_transform_matrix(BASE_FRAME, IK_LINK_FRAME)
        return BeginningPose(
            joint_seed={name: float(seed_state[name]) for name in JOINT_NAMES},
            base_from_tcp=base_from_tcp,
            link_from_tcp=np.linalg.inv(base_from_link) @ base_from_tcp,
        )

    def _lookup_transform_matrix(self, target_frame: str, source_frame: str) -> np.ndarray:
        # TF snapshots are converted once into homogeneous matrices for all downstream geometry.
        transform = self._tf_buffer.lookup_transform(target_frame, source_frame, Time()).transform
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation_matrix_from_quaternion(
            np.array(
                [
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ],
                dtype=np.float64,
            )
        )
        matrix[:3, 3] = np.array(
            [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            dtype=np.float64,
        )
        return matrix

    def _run_motion(self, goal_handle, execute: bool):
        """Compile the latest raster plan and optionally stream it to the FR3 controller."""
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
        if self._wait_for_vendor_joint_state(
            goal_handle,
            "execute" if execute else "preview",
        ) is None:
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

        beginning_pose = self._require_beginning_pose()
        seed = dict(beginning_pose.joint_seed)
        base_from_tcp = beginning_pose.base_from_tcp
        link_from_tcp = beginning_pose.link_from_tcp
        idle_values = list(self.get_parameter("idle_joint_positions").value)
        idle_seed = {name: float(idle_values[index]) for index, name in enumerate(JOINT_NAMES)}
        motion_speed_m_s = float(self.get_parameter("motion_speed_m_s").value)
        raster_speed_m_s = float(self.get_parameter("raster_speed_m_s").value)
        parking_lift_m = float(self.get_parameter("parking_lift_m").value)

        fk_request = GetPositionFK.Request()
        fk_request.header.frame_id = BASE_FRAME
        fk_request.fk_link_names = [IK_LINK_FRAME]
        fk_request.robot_state.joint_state.name = list(JOINT_NAMES)
        fk_request.robot_state.joint_state.position = [
            float(idle_seed[name]) for name in JOINT_NAMES
        ]
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
                float(
                    np.linalg.norm(
                        (idle_base_from_link @ link_from_tcp)[:3, 3] - base_from_tcp[:3, 3]
                    )
                )
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
            beginning_pose.joint_seed,
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
            cutting_active = False
            try:
                for segment_name, trajectory in trajectories:
                    if self._stop_requested.is_set():
                        with self._state_lock:
                            self._active_execute = False
                        goal_handle.abort()
                        result.success = False
                        result.message = STOP_RECOVERY_MESSAGE
                        return result
                    feedback = ExecuteMotion.Feedback()
                    feedback.active_segment = segment_name
                    goal_handle.publish_feedback(feedback)
                    if segment_name == "raster":
                        self._publish_cutting(True)
                        cutting_active = True
                    while True:
                        if self._wait_for_vendor_joint_state(goal_handle, "execute") is None:
                            with self._state_lock:
                                self._active_execute = False
                            goal_handle.abort()
                            result.success = False
                            result.message = STOP_RECOVERY_MESSAGE
                            return result
                        controller_result = self._send_controller_trajectory(trajectory)
                        if self._stop_requested.is_set():
                            with self._state_lock:
                                self._active_execute = False
                            goal_handle.abort()
                            result.success = False
                            result.message = STOP_RECOVERY_MESSAGE
                            return result
                        if controller_result == FollowJointTrajectory.Result.SUCCESSFUL:
                            break
                        with self._state_lock:
                            observed_count = self._joint_state_count
                        if not self._controller_failure_matches_joint_state_loss(observed_count):
                            raise RuntimeError(MOTION_VENDOR_ERROR)
                    if segment_name == "raster":
                        self._publish_cutting(False)
                        cutting_active = False
            finally:
                if cutting_active:
                    self._publish_cutting(False)

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

    def _publish_cutting(self, enabled: bool) -> None:
        self._cutting_pub.publish(Bool(data=enabled))

    def _send_controller_trajectory(self, trajectory: JointTrajectory) -> int:
        controller_goal = FollowJointTrajectory.Goal()
        controller_goal.trajectory = trajectory
        self._controller_done.clear()
        controller_handle = wait_future(self._controller_client.send_goal_async(controller_goal))
        if not controller_handle.accepted:
            with self._state_lock:
                unavailable = not self._joint_states_available
            if unavailable:
                self._controller_done.set()
                return FollowJointTrajectory.Result.INVALID_GOAL
            self._activate_franka_controllers()
            controller_handle = wait_future(self._controller_client.send_goal_async(controller_goal))
        if not controller_handle.accepted:
            self._controller_done.set()
            with self._state_lock:
                self._active_controller_handle = None
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

    def _publish_preview_playback(
        self,
        goal_handle,
        trajectories: list[tuple[str, JointTrajectory]],
    ) -> bool:
        for segment_name, trajectory in trajectories:
            feedback = PreviewMotion.Feedback(phase=segment_name)
            goal_handle.publish_feedback(feedback)
            previous_s: float | None = None
            for point in trajectory.points:
                if self._stop_requested.is_set():
                    return False
                point_s = (
                    float(point.time_from_start.sec)
                    + float(point.time_from_start.nanosec) * 1e-9
                )
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
            controller_handle = wait_future(
                self._controller_client.send_goal_async(controller_goal)
            )
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
        if (
            self._latest_joint_state is None
            or self._beginning_pose is None
        ):
            raise RuntimeError(MOTION_INPUT_ERROR)
        beginning_pose = self._require_beginning_pose()
        self._ik_client.wait_for_service()
        self._controller_client.wait_for_server()
        current_seed = dict(
            zip(self._latest_joint_state.name, self._latest_joint_state.position, strict=True)
        )
        seed = {name: float(current_seed[name]) for name in JOINT_NAMES}
        motion_speed_m_s = float(self.get_parameter("motion_speed_m_s").value)
        base_from_tcp = self._lookup_transform_matrix(BASE_FRAME, TOOL_FRAME)
        retract = np.array(base_from_tcp, dtype=np.float64, copy=True)
        retract[2, 3] += float(self.get_parameter("parking_lift_m").value)
        solved_points: list[dict[str, float]] = []
        for tcp_matrix in densified_matrices([base_from_tcp, retract]):
            base_from_ik_link = tcp_matrix @ np.linalg.inv(beginning_pose.link_from_tcp)
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
        retract_trajectory = JointTrajectory()
        retract_trajectory.joint_names = list(JOINT_NAMES)
        retract_trajectory.points = joint_trajectory_points(
            solved_points,
            densified_matrices([base_from_tcp, retract]),
            motion_speed_m_s,
        )
        return_distance_m = float(
            np.linalg.norm(retract[:3, 3] - beginning_pose.base_from_tcp[:3, 3])
        )
        return_trajectory = JointTrajectory()
        return_trajectory.joint_names = list(JOINT_NAMES)
        return_trajectory.points = joint_interpolation_points(
            seed,
            beginning_pose.joint_seed,
            max(return_distance_m / motion_speed_m_s, 0.05),
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

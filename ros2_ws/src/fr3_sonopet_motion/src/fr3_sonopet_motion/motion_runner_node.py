from __future__ import annotations

import math
from threading import Event, RLock

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from fr3_sonopet_interfaces.action import ExecuteMotion, PreviewMotion, StopMotion
from fr3_sonopet_interfaces.msg import RasterPlan
from geometry_msgs.msg import PoseStamped
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
        self._beginning_seed: dict[str, float] | None = None
        self._beginning_base_from_tcp: np.ndarray | None = None
        self._beginning_link_from_tcp: np.ndarray | None = None
        self._active_execute = False
        self._active_controller_handle = None
        self._controller_done = Event()
        self._controller_done.set()
        self._stop_requested = Event()
        self._state_lock = RLock()
        self._callback_group = ReentrantCallbackGroup()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # Cache the latest raster plan and robot joints so actions always compile from live inputs.
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
        self._joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            lambda joint_state: setattr(self, "_latest_joint_state", joint_state),
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
            self._execute_stop_motion,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            "Motion runner ready: "
            f"base={BASE_FRAME}, tool={TOOL_FRAME}, ik_link={IK_LINK_FRAME}, "
            f"controller={CONTROLLER_ACTION}"
        )

    def _run_motion(self, goal_handle, execute: bool):
        """Execute or preview a motion plan."""
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
            feedback.phase = "compile"
        goal_handle.publish_feedback(feedback)

        # Abort early if the planner, robot state, or frame contract is not ready.
        if (
            self._latest_plan is None
            or self._latest_joint_state is None
            or self._latest_plan.header.frame_id != BASE_FRAME
            or len(self._latest_plan.poses.poses) == 0
        ):
            raise RuntimeError(MOTION_INPUT_ERROR)

        self._ik_client.wait_for_service()
        self._fk_client.wait_for_service()
        if execute:
            self._controller_client.wait_for_server()

        # The first preview or execute after a new raster plan freezes the scan's starting pose.
        if self._beginning_seed is None:
            seed_state = dict(
                zip(self._latest_joint_state.name, self._latest_joint_state.position, strict=True)
            )
            base_from_tcp_msg = self._tf_buffer.lookup_transform(
                BASE_FRAME,
                TOOL_FRAME,
                Time(),
            ).transform
            base_from_link_msg = self._tf_buffer.lookup_transform(
                BASE_FRAME,
                IK_LINK_FRAME,
                Time(),
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
        joint_segment_count = 0
        joint_waypoint_count = 0

        # Segment setup: compute the idle TCP pose and absolute raster path used by the motion stages.
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
        idle_base_from_tcp = idle_base_from_link @ link_from_tcp
        raster = build_cartesian_segments(self._latest_plan, base_from_tcp)
        cartesian_segments = staged_cartesian_segments(
            idle_base_from_tcp,
            raster,
            motion_speed_m_s,
            raster_speed_m_s,
            parking_lift_m,
        )

        # Segment A: current to idle, pure joint motion between known robot states.
        trajectory = JointTrajectory()
        trajectory.joint_names = list(JOINT_NAMES)
        trajectory.points = joint_interpolation_points(
            seed,
            idle_seed,
            max(
                float(np.linalg.norm(idle_base_from_tcp[:3, 3] - base_from_tcp[:3, 3]))
                / motion_speed_m_s,
                0.05,
            ),
        )
        joint_segment_count += 1
        joint_waypoint_count += len(trajectory.points)
        if execute:
            if not self._send_controller_trajectory(goal_handle, trajectory, "current_to_idle"):
                return self._finish_stopped_execution(goal_handle, result)
        seed = dict(idle_seed)

        # Segment B: idle to parking, Cartesian TCP motion above the first raster point.
        # Segment C: parking to first, slow Cartesian descent onto the scan start.
        # Segment D: raster, slow Cartesian scan through the planner's base-frame waypoints.
        # Segment E: retract, Cartesian lift away from the final raster point.
        for segment in cartesian_segments:
            if execute and self._stop_requested.is_set():
                return self._finish_stopped_execution(goal_handle, result)
            trajectory = JointTrajectory()
            trajectory.joint_names = list(JOINT_NAMES)
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

            # Convert solved joint samples into a timed trajectory with slower raster motion.
            trajectory.points = joint_trajectory_points(
                solved_points,
                segment.tcp_matrices,
                segment.speed_m_s,
            )
            joint_segment_count += 1
            joint_waypoint_count += len(trajectory.points)

            if execute:
                if not self._send_controller_trajectory(goal_handle, trajectory, segment.name):
                    return self._finish_stopped_execution(goal_handle, result)

        # Segment F: return to start, pure joint motion back to the latched beginning state.
        trajectory = JointTrajectory()
        trajectory.joint_names = list(JOINT_NAMES)
        trajectory.points = joint_interpolation_points(
            seed,
            self._beginning_seed,
            max(
                float(np.linalg.norm(cartesian_segments[-1].tcp_matrices[-1][:3, 3] - base_from_tcp[:3, 3]))
                / motion_speed_m_s,
                0.05,
            ),
        )
        joint_segment_count += 1
        joint_waypoint_count += len(trajectory.points)
        if execute:
            if not self._send_controller_trajectory(goal_handle, trajectory, "return_to_start"):
                return self._finish_stopped_execution(goal_handle, result)

        goal_handle.succeed()
        result.success = True
        verb = "Executed" if execute else "Preview compiled"
        result.message = (
            f"{verb} {joint_segment_count} segments and {joint_waypoint_count} joint waypoints."
        )
        if execute:
            with self._state_lock:
                self._active_execute = False
        return result

    def _send_controller_trajectory(self, goal_handle, trajectory: JointTrajectory, segment_name: str) -> bool:
        # Sending one segment at a time mirrors the reference hardware execution sequence.
        feedback = ExecuteMotion.Feedback()
        feedback.active_segment = segment_name
        goal_handle.publish_feedback(feedback)
        controller_goal = FollowJointTrajectory.Goal()
        controller_goal.trajectory = trajectory
        self._controller_done.clear()
        controller_handle = wait_future(self._controller_client.send_goal_async(controller_goal))
        if not controller_handle.accepted:
            self._controller_done.set()
            raise RuntimeError(MOTION_VENDOR_ERROR)
        with self._state_lock:
            self._active_controller_handle = controller_handle
        controller_result = wait_future(controller_handle.get_result_async()).result
        with self._state_lock:
            self._active_controller_handle = None
        self._controller_done.set()
        if self._stop_requested.is_set():
            return False
        if controller_result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(MOTION_VENDOR_ERROR)
        return True

    def _finish_stopped_execution(self, goal_handle, result):
        with self._state_lock:
            self._active_execute = False
        goal_handle.abort()
        result.success = False
        result.message = "Execution stopped by operator; recovery is handled by the stop action."
        return result

    def _execute_stop_motion(self, goal_handle):
        result = StopMotion.Result()
        with self._state_lock:
            active = self._active_execute
            controller_handle = self._active_controller_handle
        if not active:
            goal_handle.succeed()
            result.success = True
            result.message = "No active motion to stop."
            return result

        feedback = StopMotion.Feedback()
        feedback.phase = "cancel_current"
        goal_handle.publish_feedback(feedback)
        self._stop_requested.set()
        if controller_handle is not None:
            wait_future(controller_handle.cancel_goal_async())
        self._controller_done.wait()

        feedback.phase = "retract"
        goal_handle.publish_feedback(feedback)
        self._run_stop_recovery()

        with self._state_lock:
            self._active_execute = False
            self._stop_requested.clear()
        goal_handle.succeed()
        result.success = True
        result.message = "Motion stopped, retracted, and returned to the beginning pose."
        return result

    def _run_stop_recovery(self) -> None:
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
        retract_trajectory, seed = self._solve_cartesian_trajectory(
            densified_matrices([base_from_tcp, retract]),
            seed,
            self._beginning_link_from_tcp,
            float(self.get_parameter("motion_speed_m_s").value),
        )
        self._send_recovery_trajectory(retract_trajectory)
        return_trajectory = JointTrajectory()
        return_trajectory.joint_names = list(JOINT_NAMES)
        return_trajectory.points = joint_interpolation_points(
            seed,
            self._beginning_seed,
            max(float(np.linalg.norm(retract[:3, 3] - base_from_tcp[:3, 3])) / float(self.get_parameter("motion_speed_m_s").value), 0.05),
        )
        self._send_recovery_trajectory(return_trajectory)

    def _solve_cartesian_trajectory(
        self,
        tcp_matrices: list[np.ndarray],
        seed: dict[str, float],
        link_from_tcp: np.ndarray,
        speed_m_s: float,
    ) -> tuple[JointTrajectory, dict[str, float]]:
        solved_points: list[dict[str, float]] = []
        for tcp_matrix in tcp_matrices:
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
        trajectory = JointTrajectory()
        trajectory.joint_names = list(JOINT_NAMES)
        trajectory.points = joint_trajectory_points(solved_points, tcp_matrices, speed_m_s)
        return trajectory, seed

    def _send_recovery_trajectory(self, trajectory: JointTrajectory) -> None:
        controller_goal = FollowJointTrajectory.Goal()
        controller_goal.trajectory = trajectory
        controller_handle = wait_future(self._controller_client.send_goal_async(controller_goal))
        if not controller_handle.accepted:
            raise RuntimeError(MOTION_VENDOR_ERROR)
        controller_result = wait_future(controller_handle.get_result_async()).result
        if controller_result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(MOTION_VENDOR_ERROR)


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

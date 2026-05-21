from __future__ import annotations

import math
from threading import Event

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from fr3_sonopet_interfaces.action import ExecuteMotion, PreviewMotion
from fr3_sonopet_interfaces.msg import RasterPlan
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
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
    IK_LINK_FRAME,
    IK_SERVICE,
    IK_TIMEOUT_S,
    JOINT_NAMES,
    PLANNING_GROUP,
    RASTER_PLAN_TOPIC,
    TOOL_FRAME,
    build_cartesian_segments,
    joint_trajectory_points,
    quaternion_from_matrix,
    rotation_matrix_from_quaternion,
)
from fr3_sonopet_motion.operator_policy import EXECUTE_TOKEN

MOTION_INPUT_ERROR = "Motion runner input is incomplete or inconsistent."
MOTION_VENDOR_ERROR = "Motion vendor command failed."


class MotionRunnerNode(Node):
    """Compile raster plans against the live FR3 pose and execute them through ros2_control."""

    def __init__(self) -> None:
        super().__init__("motion_runner_node")
        self._latest_plan: RasterPlan | None = None
        self._latest_joint_state: JointState | None = None
        self._callback_group = ReentrantCallbackGroup()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # Cache the latest raster plan and robot joints so actions always compile from live inputs.
        self._plan_sub = self.create_subscription(
            RasterPlan,
            RASTER_PLAN_TOPIC,
            lambda plan: setattr(self, "_latest_plan", plan),
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
        if execute:
            self._controller_client.wait_for_server()

        # Seed IK from the current FR3 joints and hold the current TCP orientation for the scan.
        seed_state = dict(
            zip(self._latest_joint_state.name, self._latest_joint_state.position, strict=True)
        )
        seed = {name: float(seed_state[name]) for name in JOINT_NAMES}
        base_from_tcp_msg = self._tf_buffer.lookup_transform(BASE_FRAME, TOOL_FRAME, Time()).transform
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
        link_from_tcp = np.linalg.inv(base_from_link) @ base_from_tcp
        joint_segment_count = 0
        joint_waypoint_count = 0

        # Each Cartesian segment is solved sequentially so each waypoint biases the next IK call.
        for segment in build_cartesian_segments(self._latest_plan, base_from_tcp):
            trajectory = JointTrajectory()
            trajectory.joint_names = list(JOINT_NAMES)
            solved_points: list[dict[str, float]] = []
            for tcp_matrix in segment.tcp_matrices:
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
                # Sending one segment at a time mirrors the reference hardware execution sequence.
                feedback = ExecuteMotion.Feedback()
                feedback.active_segment = segment.name
                goal_handle.publish_feedback(feedback)
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

        goal_handle.succeed()
        result.success = True
        verb = "Executed" if execute else "Preview compiled"
        result.message = (
            f"{verb} {joint_segment_count} segments and {joint_waypoint_count} joint waypoints."
        )
        return result


def wait_future(future):
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

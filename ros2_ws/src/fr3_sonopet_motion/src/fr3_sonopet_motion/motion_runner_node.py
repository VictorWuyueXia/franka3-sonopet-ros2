from __future__ import annotations

import rclpy
from fr3_sonopet_interfaces.action import ExecuteMotion, PreviewMotion
from rclpy.action import ActionServer
from rclpy.node import Node

from fr3_sonopet_motion.moveit_client import MoveItPolicy, describe_policy
from fr3_sonopet_motion.operator_policy import EXECUTE_TOKEN


class MotionRunnerNode(Node):
    """Action boundary for MoveIt preview and execution requests."""

    def __init__(self) -> None:
        super().__init__("motion_runner_node")
        self.declare_parameter("planning_group", "fr3_arm")
        self.declare_parameter("fake_execution", False)
        self._policy = MoveItPolicy(planning_group=str(self.get_parameter("planning_group").value))
        self._preview_server = ActionServer(
            self,
            PreviewMotion,
            "/sonopet/preview_motion",
            self._execute_preview,
        )
        self._execute_server = ActionServer(
            self,
            ExecuteMotion,
            "/sonopet/execute_motion",
            self._execute_motion,
        )
        self.get_logger().info(f"Motion runner ready: {describe_policy(self._policy)}")

    def _execute_preview(self, goal_handle):
        # Preview validates the action boundary without commanding hardware.
        feedback = PreviewMotion.Feedback()
        feedback.phase = "preview_interface_shell"
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()

        result = PreviewMotion.Result()
        result.success = True
        result.message = "Preview interface shell accepted the raster plan."
        return result

    def _execute_motion(self, goal_handle):
        # Execution requests remain guarded by the operator confirmation token.
        result = ExecuteMotion.Result()
        if goal_handle.request.confirmation_token.strip() != EXECUTE_TOKEN:
            goal_handle.abort()
            result.success = False
            result.message = "Execution rejected because the confirmation token is invalid."
            return result

        feedback = ExecuteMotion.Feedback()
        feedback.active_segment = "interface_shell"
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()

        result.success = True
        result.message = "Execute interface shell accepted the confirmed raster plan."
        return result


def main() -> None:
    rclpy.init()
    node = MotionRunnerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

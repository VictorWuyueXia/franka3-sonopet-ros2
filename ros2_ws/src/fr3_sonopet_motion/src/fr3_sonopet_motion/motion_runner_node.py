from __future__ import annotations

import rclpy
from rclpy.node import Node

from fr3_sonopet_motion.moveit_client import MoveItPolicy, describe_policy


class MotionRunnerNode(Node):
    """Placeholder node for MoveIt planning, preview, and execution."""

    def __init__(self) -> None:
        super().__init__("motion_runner_node")
        self.declare_parameter("planning_group", "fr3_arm")
        self.declare_parameter("fake_execution", False)
        policy = MoveItPolicy(planning_group=str(self.get_parameter("planning_group").value))
        self.get_logger().info(f"Motion runner ready: {describe_policy(policy)}")


def main() -> None:
    rclpy.init()
    node = MotionRunnerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


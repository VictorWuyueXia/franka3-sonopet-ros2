from __future__ import annotations

import rclpy
from rclpy.node import Node


class RasterPlannerNode(Node):
    """Placeholder node for point-cloud to raster-plan generation."""

    def __init__(self) -> None:
        super().__init__("raster_planner_node")
        self.declare_parameter("base_frame", "fr3_link0")
        self.declare_parameter("planning_cloud", "in_hand")
        self.declare_parameter("use_fixture_cloud", False)
        base_frame = self.get_parameter("base_frame").value
        planning_cloud = self.get_parameter("planning_cloud").value
        self.get_logger().info(
            f"Raster planner ready: base_frame={base_frame}, planning_cloud={planning_cloud}"
        )


def main() -> None:
    rclpy.init()
    node = RasterPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


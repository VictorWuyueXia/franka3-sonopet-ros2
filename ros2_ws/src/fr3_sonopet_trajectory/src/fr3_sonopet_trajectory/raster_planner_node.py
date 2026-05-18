from __future__ import annotations

import rclpy
from geometry_msgs.msg import Point, PoseArray
from rclpy.action import ActionServer
from rclpy.node import Node
from std_msgs.msg import Header

from fr3_sonopet_interfaces.action import BuildRasterPlan
from fr3_sonopet_interfaces.msg import RasterPatch, RasterPlan


class RasterPlannerNode(Node):
    """Interface shell for point-cloud to raster-plan generation."""

    def __init__(self) -> None:
        super().__init__("raster_planner_node")
        self.declare_parameter("base_frame", "fr3_link0")
        self.declare_parameter("planning_cloud", "in_hand")
        self.declare_parameter("use_fixture_cloud", False)

        self._base_frame = str(self.get_parameter("base_frame").value)
        self._planning_cloud = str(self.get_parameter("planning_cloud").value)
        self._plan_pub = self.create_publisher(RasterPlan, "/sonopet/raster_plan", 10)
        self._build_server = ActionServer(
            self,
            BuildRasterPlan,
            "/sonopet/build_raster_plan",
            self._execute_build_plan,
        )
        self.get_logger().info(
            f"Raster planner ready: base_frame={self._base_frame}, planning_cloud={self._planning_cloud}"
        )

    def _execute_build_plan(self, goal_handle):
        # The shell validates transport and publishes a structurally valid empty plan.
        feedback = BuildRasterPlan.Feedback()
        feedback.phase = "building_shell_plan"
        goal_handle.publish_feedback(feedback)

        plan = self._empty_plan(goal_handle.request.selected_center)
        self._plan_pub.publish(plan)

        goal_handle.succeed()
        result = BuildRasterPlan.Result()
        result.success = True
        result.message = "Raster plan interface shell published an empty plan."
        result.plan = plan
        return result

    def _empty_plan(self, selected_center: Point) -> RasterPlan:
        # Header and patch fields keep downstream consumers synchronized on frame policy.
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self._base_frame

        patch = RasterPatch()
        patch.header = header
        patch.center = selected_center
        patch.square_side_m = 0.0
        patch.line_spacing_m = 0.0
        patch.pattern = "interface_shell"

        poses = PoseArray()
        poses.header = header

        plan = RasterPlan()
        plan.header = header
        plan.patch = patch
        plan.poses = poses
        plan.segment_names = []
        plan.config_hash = "interface_shell"
        return plan


def main() -> None:
    rclpy.init()
    node = RasterPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

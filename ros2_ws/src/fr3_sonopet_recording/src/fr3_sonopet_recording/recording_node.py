from __future__ import annotations

import rclpy
from rclpy.node import Node

from fr3_sonopet_recording.topic_policy import DEFAULT_TOPICS, require_topics


class RecordingNode(Node):
    """Placeholder for rosbag2 start/stop orchestration and manifests."""

    def __init__(self) -> None:
        super().__init__("recording_node")
        self.declare_parameter("artifact_root", "artifacts/experiments")
        require_topics(DEFAULT_TOPICS)
        artifact_root = self.get_parameter("artifact_root").value
        self.get_logger().info(f"Recording placeholder ready: artifact_root={artifact_root}")


def main() -> None:
    rclpy.init()
    node = RecordingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


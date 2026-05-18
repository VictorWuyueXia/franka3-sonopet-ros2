from __future__ import annotations

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from fr3_sonopet_interfaces.action import RecordExperiment
from fr3_sonopet_recording.topic_policy import DEFAULT_TOPICS, require_topics


class RecordingNode(Node):
    """Action boundary for rosbag2 start and stop orchestration."""

    def __init__(self) -> None:
        super().__init__("recording_node")
        self.declare_parameter("artifact_root", "artifacts/experiments")
        require_topics(DEFAULT_TOPICS)
        self._artifact_root = str(self.get_parameter("artifact_root").value)
        self._record_server = ActionServer(
            self,
            RecordExperiment,
            "/sonopet/record_experiment",
            self._execute_record,
        )
        self.get_logger().info(f"Recording interface ready: artifact_root={self._artifact_root}")

    def _execute_record(self, goal_handle):
        # Deterministic artifact paths give downstream tools a stable run boundary.
        feedback = RecordExperiment.Feedback()
        feedback.phase = "recording_start" if goal_handle.request.start else "recording_stop"
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()

        result = RecordExperiment.Result()
        result.success = True
        result.message = "Recording interface shell accepted the request."
        result.artifact_path = f"{self._artifact_root}/{goal_handle.request.run_id}"
        return result


def main() -> None:
    rclpy.init()
    node = RecordingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

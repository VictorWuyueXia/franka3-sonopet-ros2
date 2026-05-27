from __future__ import annotations

from datetime import datetime

import rclpy
from fr3_sonopet_interfaces.msg import RunState as RunStateMsg
from rclpy.node import Node

from fr3_sonopet_supervisor.run_state import RunState


class ExperimentSupervisorNode(Node):
    """Publishes the experiment lifecycle state for operator-facing tooling."""

    def __init__(self) -> None:
        super().__init__("experiment_supervisor_node")
        self.declare_parameter("fake_run", False)
        run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        self._state = RunState(run_id=run_id, ready=True)
        self._state_pub = self.create_publisher(RunStateMsg, "/sonopet/run_state", 10)
        self._state_timer = self.create_timer(1.0, self._publish_state)
        self.get_logger().info(f"Experiment supervisor ready: run_id={self._state.run_id}")

    def _publish_state(self) -> None:
        # A periodic state topic lets launch tests and operator tools observe liveness.
        msg = RunStateMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        msg.run_id = self._state.run_id
        msg.phase = self._state.phase
        msg.ready = self._state.ready
        msg.blocking_reason = self._state.blocking_reason
        msg.last_error = ""
        self._state_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = ExperimentSupervisorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

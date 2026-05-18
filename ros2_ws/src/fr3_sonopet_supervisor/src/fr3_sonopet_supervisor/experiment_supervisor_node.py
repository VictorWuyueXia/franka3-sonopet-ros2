from __future__ import annotations

import rclpy
from rclpy.node import Node

from fr3_sonopet_supervisor.run_state import RunState


class ExperimentSupervisorNode(Node):
    """Placeholder node for the operator-gated experiment lifecycle."""

    def __init__(self) -> None:
        super().__init__("experiment_supervisor_node")
        self.declare_parameter("fake_run", False)
        self.declare_parameter("run_id", "manual_run")
        state = RunState(run_id=str(self.get_parameter("run_id").value), ready=True)
        self.get_logger().info(f"Experiment supervisor ready: run_id={state.run_id}")


def main() -> None:
    rclpy.init()
    node = ExperimentSupervisorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


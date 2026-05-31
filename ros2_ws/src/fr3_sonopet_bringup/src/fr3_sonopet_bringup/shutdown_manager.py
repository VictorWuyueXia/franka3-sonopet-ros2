from __future__ import annotations

import os
import signal

import rclpy
from fr3_sonopet_interfaces.action import StopMotion
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

SHUTDOWN_SERVICE = "/sonopet/shutdown"
STOP_MOTION_ACTION = "/fr3/stop_motion"


class ShutdownManager(Node):
    """Own the operator-requested shutdown sequence for the experiment launch."""

    def __init__(self) -> None:
        super().__init__("shutdown_manager_node")
        self._requested = False
        self._stop_client = ActionClient(self, StopMotion, STOP_MOTION_ACTION)
        self._shutdown_service = self.create_service(Trigger, SHUTDOWN_SERVICE, self._on_shutdown)

    def _on_shutdown(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        # Shutdown is accepted once, then routed through motion recovery before launch exits.
        if self._requested:
            response.success = False
            response.message = "Shutdown already requested."
            return response
        self._requested = True
        self._stop_client.wait_for_server()
        send_goal_future = self._stop_client.send_goal_async(StopMotion.Goal())
        send_goal_future.add_done_callback(self._on_stop_goal)
        response.success = True
        response.message = "Shutdown accepted; stopping motion before launch exit."
        return response

    def _on_stop_goal(self, future) -> None:
        # StopMotion owns active trajectory cancellation and return-to-start recovery.
        goal_handle = future.result()
        if not goal_handle.accepted:
            raise RuntimeError("StopMotion rejected shutdown recovery.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_stop_result)

    def _on_stop_result(self, future) -> None:
        # The parent ros2 launch process propagates SIGINT to all experiment nodes.
        result = future.result().result
        if not result.success:
            raise RuntimeError(result.message)
        os.kill(os.getppid(), signal.SIGINT)


def main() -> None:
    rclpy.init()
    node = ShutdownManager()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.spin()
    executor.remove_node(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

from __future__ import annotations

import rclpy
from rclpy.node import Node

from fr3_sonopet_microphone.audio_format import AudioFormat, describe_audio_format


class MicrophoneNode(Node):
    """Interface shell for microphone stream settings."""

    def __init__(self) -> None:
        super().__init__("microphone_node")
        self.declare_parameter("sample_rate_hz", 48_000)
        self.declare_parameter("channels", 1)
        audio_format = AudioFormat(
            sample_rate_hz=int(self.get_parameter("sample_rate_hz").value),
            channels=int(self.get_parameter("channels").value),
        )
        self.get_logger().info(f"Microphone interface ready: {describe_audio_format(audio_format)}")


def main() -> None:
    rclpy.init()
    node = MicrophoneNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

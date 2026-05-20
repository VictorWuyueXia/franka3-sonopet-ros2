from __future__ import annotations

from queue import Full, Queue

import numpy as np
import rclpy
from fr3_sonopet_interfaces.msg import AudioChunk
from rclpy.node import Node
from rclpy.parameter import Parameter

from fr3_sonopet_microphone.audio_devices import select_input_device
from fr3_sonopet_microphone.audio_format import AudioFormat, describe_audio_format


class MicrophoneNode(Node):
    """Publish low-latency PCM audio chunks from the configured USB microphone."""

    def __init__(self) -> None:
        super().__init__("microphone_node")
        self.declare_parameter("sample_rate_hz", Parameter.Type.INTEGER)
        self.declare_parameter("channels", Parameter.Type.INTEGER)
        self.declare_parameter("chunk_frames", Parameter.Type.INTEGER)
        self.declare_parameter("preferred_device_names", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("queue_depth", Parameter.Type.INTEGER)

        # Audio format is intentionally controlled only by the bringup parameter file.
        self._audio_format = AudioFormat(
            sample_rate_hz=int(self._required_parameter("sample_rate_hz")),
            channels=int(self._required_parameter("channels")),
            chunk_frames=int(self._required_parameter("chunk_frames")),
        )
        self._queue: Queue[np.ndarray] = Queue(maxsize=int(self._required_parameter("queue_depth")))
        self._dropped_chunks = 0
        self._stream = None
        self._publisher = self.create_publisher(AudioChunk, "/microphone/audio", 10)
        self._publish_timer = self.create_timer(0.005, self._publish_available_chunks)

        self._start_stream()
        self.get_logger().info(
            f"Microphone publisher ready: {describe_audio_format(self._audio_format)}"
        )

    def _required_parameter(self, name: str):
        parameter = self.get_parameter(name)
        if parameter.type_ == Parameter.Type.NOT_SET:
            raise RuntimeError(f"Required microphone parameter is not set: {name}")
        return parameter.value

    def _start_stream(self) -> None:
        """Resolve the configured microphone and start the sounddevice input stream."""

        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is required for microphone publishing") from exc

        preferred_names = [
            str(value) for value in self._required_parameter("preferred_device_names")
        ]
        device = select_input_device(
            devices=list(sd.query_devices()),
            preferred_names=preferred_names,
            explicit_device="",
            fail_if_preferred_not_found=True,
        )

        self._stream = sd.InputStream(
            device=device.index,
            samplerate=self._audio_format.sample_rate_hz,
            channels=self._audio_format.channels,
            dtype="int16",
            blocksize=self._audio_format.chunk_frames,
            callback=self._on_audio,
        )
        self._stream.start()
        self.get_logger().info(
            f"Using microphone [{device.index}]: {device.name} "
            f"({device.channels} input channels, default {device.default_sample_rate_hz} Hz)"
        )

    def _on_audio(self, indata, _frames, _time_info, status) -> None:
        """Copy callback audio into a bounded queue for ROS-side publication."""

        if status:
            self.get_logger().warning(f"Microphone stream status: {status}")
        try:
            self._queue.put_nowait(np.asarray(indata, dtype=np.int16).copy())
        except Full:
            self._dropped_chunks += 1

    def _publish_available_chunks(self) -> None:
        """Publish queued audio outside the real-time audio callback."""

        while not self._queue.empty():
            chunk = self._queue.get_nowait()
            msg = AudioChunk()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "microphone"
            msg.sample_rate_hz = self._audio_format.sample_rate_hz
            msg.channels = self._audio_format.channels
            msg.encoding = self._audio_format.encoding
            msg.samples = chunk.reshape(-1).tolist()
            self._publisher.publish(msg)

        if self._dropped_chunks:
            dropped = self._dropped_chunks
            self._dropped_chunks = 0
            self.get_logger().warning(f"Dropped {dropped} microphone chunks")

    def destroy_node(self) -> bool:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = MicrophoneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

from __future__ import annotations

from queue import Full, Queue

import numpy as np
import rclpy
from fr3_sonopet_interfaces.msg import AudioChunk
from rclpy.node import Node

from fr3_sonopet_microphone.audio_devices import select_input_device
from fr3_sonopet_microphone.audio_format import AudioFormat, describe_audio_format

class MicrophoneNode(Node):
    """Publish low-latency PCM audio chunks from the configured USB microphone."""

    def __init__(self) -> None:
        super().__init__("microphone_node")
        # Declare and initialize configuration parameters
        self.declare_parameter("sample_rate_hz", 48_000)
        self.declare_parameter("channels", 1)
        self.declare_parameter("chunk_frames", 1024)
        self.declare_parameter("device", "")
        self.declare_parameter("preferred_device_names", ["iMM-6C", "imm6c"])
        self.declare_parameter("fail_if_preferred_not_found", True)
        self.declare_parameter("queue_depth", 32)

        # Set up audio format from configuration
        self._audio_format = AudioFormat(
            sample_rate_hz=int(self.get_parameter("sample_rate_hz").value),
            channels=int(self.get_parameter("channels").value),
            chunk_frames=int(self.get_parameter("chunk_frames").value),
        )

        # Bounded internal buffer to hold microphone audio chunks
        self._queue: Queue[np.ndarray] = Queue(maxsize=int(self.get_parameter("queue_depth").value))
        self._dropped_chunks = 0  # Tracks if buffer is full and audio is lost
        self._stream = None  # Will hold the microphone stream object
        # Set up ROS publisher for audio topic
        self._publisher = self.create_publisher(AudioChunk, "/microphone/audio", 10)
        # Periodically check and publish available audio
        self._publish_timer = self.create_timer(0.005, self._publish_available_chunks)

        self._start_stream()  # Start the USB microphone input stream

        self.get_logger().info(
            f"Microphone publisher ready: {describe_audio_format(self._audio_format)}"
        )

    def _start_stream(self) -> None:
        """Resolve the configured microphone and start the sounddevice input stream."""

        # Import sounddevice dependency for audio capture
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is required for microphone publishing") from exc

        # Prepare preferred microphone names from the config
        preferred_names = [
            str(value) for value in self.get_parameter("preferred_device_names").value
        ]
        # Select which USB microphone to use
        device = select_input_device(
            devices=list(sd.query_devices()),
            preferred_names=preferred_names,
            explicit_device=str(self.get_parameter("device").value),
            fail_if_preferred_not_found=bool(
                self.get_parameter("fail_if_preferred_not_found").value
            ),
        )

        # Create and start the stream that reads audio data from the microphone
        self._stream = sd.InputStream(
            device=device.index,
            samplerate=self._audio_format.sample_rate_hz,
            channels=self._audio_format.channels,
            dtype="int16",
            blocksize=self._audio_format.chunk_frames,
            callback=self._on_audio,
        )
        self._stream.start()  # Begin capturing audio

        # Log microphone selection
        self.get_logger().info(
            f"Using microphone [{device.index}]: {device.name} "
            f"({device.channels} input channels, default {device.default_sample_rate_hz} Hz)"
        )

    def _on_audio(self, indata, _frames, _time_info, status) -> None:
        """Handles new audio data from microphone and places into buffer."""

        # Inform if the microphone hardware sends a warning
        if status:
            self.get_logger().warning(f"Microphone stream status: {status}")

        # Place the most recent audio samples into the internal buffer
        try:
            self._queue.put_nowait(np.asarray(indata, dtype=np.int16).copy())
        except Full:
            # Count dropped chunks if buffer is full
            self._dropped_chunks += 1

    def _publish_available_chunks(self) -> None:
        """Periodically publish all buffered audio chunks to ROS."""

        # While there is buffered audio, publish to ROS topic
        while not self._queue.empty():
            chunk = self._queue.get_nowait()
            msg = AudioChunk()
            msg.header.stamp = self.get_clock().now().to_msg()  # Timestamp message
            msg.header.frame_id = "microphone"
            msg.sample_rate_hz = self._audio_format.sample_rate_hz
            msg.channels = self._audio_format.channels
            msg.encoding = self._audio_format.encoding
            msg.samples = chunk.reshape(-1).tolist()  # Flatten and convert to integer list
            self._publisher.publish(msg)  # Publish the audio chunk message

        # Warn if any audio was lost due to buffer overflow
        if self._dropped_chunks:
            dropped = self._dropped_chunks
            self._dropped_chunks = 0
            self.get_logger().warning(f"Dropped {dropped} microphone chunks")

    def destroy_node(self) -> bool:
        # Safely close audio stream when node is destroyed
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return super().destroy_node()


def main() -> None:
    # Main entry point; start the ROS event loop and clean up afterwards
    rclpy.init()
    node = MicrophoneNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

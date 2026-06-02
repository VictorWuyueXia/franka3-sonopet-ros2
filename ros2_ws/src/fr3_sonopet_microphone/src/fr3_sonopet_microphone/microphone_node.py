from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Full, Queue
from threading import Lock
from typing import Deque

import numpy as np
import rclpy
from fr3_sonopet_interfaces.msg import AudioChunk
from rclpy.node import Node
from rclpy.parameter import Parameter

from fr3_sonopet_microphone.audio_devices import select_input_device


@dataclass(frozen=True)
class QueuedAudioChunk:
    """One microphone chunk with its stream index and source-gap status."""

    index: int
    samples: np.ndarray
    gap_fill: bool


class MicrophoneNode(Node):
    """Publish low-latency PCM audio chunks from the configured USB microphone."""

    def __init__(self) -> None:
        super().__init__("microphone_node")
        self.declare_parameter("sample_rate_hz", Parameter.Type.INTEGER)
        self.declare_parameter("channels", Parameter.Type.INTEGER)
        self.declare_parameter("chunk_frames", Parameter.Type.INTEGER)
        self.declare_parameter("preferred_device_names", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("queue_depth", Parameter.Type.INTEGER)

        self._sample_rate_hz = int(self._required_parameter("sample_rate_hz"))
        self._channels = int(self._required_parameter("channels"))
        self._chunk_frames = int(self._required_parameter("chunk_frames"))
        self._queue: Queue[QueuedAudioChunk] = Queue(maxsize=int(self._required_parameter("queue_depth")))
        self._queue_lock = Lock()
        self._pending_gap_chunks: Deque[QueuedAudioChunk] = deque()
        self._next_chunk_index = 0
        self._dropped_chunks = 0
        self._stream = None
        self._publisher = self.create_publisher(AudioChunk, "/microphone/audio", 10)
        self._publish_timer = self.create_timer(0.005, self._publish_available_chunks)

        self._start_stream()
        self.get_logger().info(
            f"Microphone publisher ready: {self._sample_rate_hz}Hz/{self._channels}ch/S16_LE"
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
        )

        self._stream = sd.InputStream(
            device=device.index,
            samplerate=self._sample_rate_hz,
            channels=self._channels,
            dtype="int16",
            blocksize=self._chunk_frames,
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
        with self._queue_lock:
            chunk_index = self._next_chunk_index
            self._next_chunk_index += 1
            if self._pending_gap_chunks:
                self._pending_gap_chunks.append(self._zero_chunk(chunk_index))
                self._dropped_chunks += 1
                return
            chunk = QueuedAudioChunk(
                index=chunk_index,
                samples=np.asarray(indata, dtype=np.int16).copy(),
                gap_fill=False,
            )
            try:
                self._queue.put_nowait(chunk)
            except Full:
                self._pending_gap_chunks.append(self._zero_chunk(chunk_index))
                self._dropped_chunks += 1

    def _zero_chunk(self, chunk_index: int) -> QueuedAudioChunk:
        """Represent one dropped audio callback with silence at its stream index."""

        return QueuedAudioChunk(
            index=chunk_index,
            samples=np.zeros((self._chunk_frames, self._channels), dtype=np.int16),
            gap_fill=True,
        )

    def _next_publish_chunk(self) -> QueuedAudioChunk | None:
        with self._queue_lock:
            if not self._queue.empty():
                return self._queue.get_nowait()
            if self._pending_gap_chunks:
                return self._pending_gap_chunks.popleft()
            return None

    def _take_dropped_chunk_count(self) -> int:
        with self._queue_lock:
            dropped = self._dropped_chunks
            self._dropped_chunks = 0
            return dropped

    def _publish_available_chunks(self) -> None:
        """Publish queued audio outside the real-time audio callback."""

        while True:
            chunk = self._next_publish_chunk()
            if chunk is None:
                break
            msg = AudioChunk()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "microphone"
            msg.sample_rate_hz = self._sample_rate_hz
            msg.channels = self._channels
            msg.encoding = "S16_LE"
            msg.chunk_index = chunk.index
            msg.gap_fill = chunk.gap_fill
            msg.samples = chunk.samples.reshape(-1).tolist()
            self._publisher.publish(msg)

        dropped = self._take_dropped_chunk_count()
        if dropped:
            self.get_logger().warning(f"Filled {dropped} dropped microphone chunks with silence")

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

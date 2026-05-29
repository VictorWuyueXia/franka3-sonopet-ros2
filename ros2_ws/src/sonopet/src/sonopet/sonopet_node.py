from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
from pathlib import Path

import rclpy
import serial
from ament_index_python.packages import get_package_prefix
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from std_msgs.msg import Bool, String

from sonopet.artifacts import (
    FOOTPEDAL_BAUD,
    FOOTPEDAL_PORT,
    SAMPLE_RATE_HZ,
    local_timestamp_label,
    relative_artifact_path,
    sonopet_case_path,
    wall_clock_timestamp,
    write_sonopet_metadata,
)

CUTTING_TOPIC = "/sonopet/cutting"
ARTIFACT_PATH_TOPIC = "/sonopet/artifact_path"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000
SERVER_START_WAIT_S = 0.2
SOCKET_BYTES = 8192


class SonopetNode(Node):
    """Record Sonopet DAQ samples and footpedal state during cutting intervals."""

    def __init__(self) -> None:
        super().__init__("sonopet_node")
        self._artifact_root: Path | None = None
        self._intervals: list[dict] = []
        self._sample_count = 0
        self._sample_file = None
        self._sample_path = Path()
        self._started_at = 0.0
        self._sampling_active = threading.Event()
        self._sampling_thread: threading.Thread | None = None
        self._footpedal = serial.Serial(FOOTPEDAL_PORT, FOOTPEDAL_BAUD, timeout=1)
        self._server_executable = (
            Path(get_package_prefix("sonopet")) / "lib" / "sonopet" / "sonopet_live_data"
        )
        self._start_server()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((SERVER_HOST, SERVER_PORT))
        artifact_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._artifact_subscription = self.create_subscription(
            String,
            ARTIFACT_PATH_TOPIC,
            self._set_artifact_root,
            artifact_qos,
        )
        self._cutting_subscription = self.create_subscription(
            Bool,
            CUTTING_TOPIC,
            self._on_cutting,
            10,
        )
        self.get_logger().info(
            f"Sonopet interface ready: cutting_topic={CUTTING_TOPIC}, "
            f"artifact_topic={ARTIFACT_PATH_TOPIC}, sample_rate_hz={SAMPLE_RATE_HZ}"
        )

    def _start_server(self) -> None:
        # The vendor DAQ process owns live samples; any start failure should stop the node.
        result = subprocess.run(
            [str(self._server_executable), "start"],
            check=True,
            capture_output=True,
            text=True,
        )
        if "ERROR:" in result.stdout or "ERROR:" in result.stderr:
            raise RuntimeError(result.stdout + result.stderr)
        time.sleep(SERVER_START_WAIT_S)

    def _set_artifact_root(self, msg: String) -> None:
        # The recorder publishes the single experiment folder shared by all sensor artifacts.
        artifact_root = Path(msg.data)
        if not artifact_root.is_absolute():
            raise ValueError(f"Sonopet artifact path must be absolute: {artifact_root}")
        self._artifact_root = artifact_root
        (self._artifact_root / "sonopet").mkdir(parents=True, exist_ok=True)

    def _on_cutting(self, msg: Bool) -> None:
        if msg.data and self._sampling_thread is None:
            self._start_interval()
        if not msg.data and self._sampling_thread is not None:
            self._stop_interval()

    def _start_interval(self) -> None:
        if self._artifact_root is None:
            raise RuntimeError(f"Sonopet artifact root has not arrived on {ARTIFACT_PATH_TOPIC}")
        self._sample_count = 0
        self._started_at = wall_clock_timestamp()
        self._sample_path = sonopet_case_path(self._artifact_root / "sonopet", self._started_at)
        self._sample_file = self._sample_path.open("w", encoding="utf-8", buffering=1)
        self._footpedal.write(b"1")
        self._sampling_active.set()
        self._sampling_thread = threading.Thread(target=self._sample_sonopet, daemon=False)
        self._sampling_thread.start()

    def _sample_sonopet(self) -> None:
        # Fixed-rate socket grabs preserve the executable payload without adding per-sample time.
        interval_s = 1.0 / SAMPLE_RATE_HZ
        next_sample = time.perf_counter()
        while self._sampling_active.is_set():
            next_sample += interval_s
            self._socket.sendall(b"grab")
            payload = json.loads(self._socket.recv(SOCKET_BYTES).decode())
            sample = {"timestamp": wall_clock_timestamp(), "data": payload}
            self._sample_file.write(json.dumps(sample) + "\n")
            self._sample_count += 1
            sleep_s = next_sample - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    def _stop_interval(self) -> None:
        stopped_at = wall_clock_timestamp()
        self._sampling_active.clear()
        self._sampling_thread.join()
        self._sampling_thread = None
        self._footpedal.write(b"0")
        self._sample_file.close()
        self._intervals.append(
            {
                "path": relative_artifact_path(self._artifact_root, self._sample_path),
                "started_at": self._started_at,
                "started_at_local": local_timestamp_label(self._started_at),
                "stopped_at": stopped_at,
                "stopped_at_local": local_timestamp_label(stopped_at),
                "sample_count": self._sample_count,
            }
        )
        write_sonopet_metadata(self._artifact_root, self._intervals)
        self._sample_file = None
        self._sample_path = Path()
        self._sample_count = 0

    def destroy_node(self) -> bool:
        if self._sampling_thread is not None:
            self._stop_interval()
        self._socket.sendall(b"stop")
        self._socket.close()
        self._footpedal.write(b"0")
        self._footpedal.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = SonopetNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.spin()
    executor.remove_node(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

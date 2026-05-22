from __future__ import annotations

import wave
from array import array
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from fr3_sonopet_interfaces.action import CapturePointCloud, RecordExperiment
from fr3_sonopet_interfaces.msg import AudioChunk
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_srvs.srv import SetBool

from fr3_sonopet_recording.artifact_writers import write_json, write_pointcloud_pcd
from fr3_sonopet_recording.recording_config import load_recording_config, recording_topics


@dataclass
class PointCloudScan:
    """One in-memory D405 pointcloud sample captured during the node session."""

    label: str
    camera_key: str
    cloud: PointCloud2
    point_count: int
    artifact_path: Path = Path()


@dataclass
class RgbFrame:
    """One in-memory RGB frame held until optional artifact export."""

    camera_key: str
    stamp_sec: int
    stamp_nanosec: int
    frame_bgr: np.ndarray


@dataclass
class AudioPacket:
    """One in-memory microphone packet held until optional artifact export."""

    stamp_sec: int
    stamp_nanosec: int
    sample_rate_hz: int
    channels: int
    encoding: str
    samples: list[int]


@dataclass
class RecordingSession:
    """One node lifecycle is one experiment recording session."""

    run_id: str
    artifact_path: Path
    started_at_iso: str
    scans: list[PointCloudScan] = field(default_factory=list)
    rgb_frames: list[RgbFrame] = field(default_factory=list)
    audio_packets: list[AudioPacket] = field(default_factory=list)


def _camera_dir(artifact_path: Path, camera_key: str) -> Path:
    if camera_key == "in_hand":
        return artifact_path / "camera_in_hand"
    if camera_key == "fixed":
        return artifact_path / "camera_fixed"
    raise ValueError(f"Unsupported recording camera: {camera_key}")


class RecordingNode(Node):
    """Hold one recording session in memory and save artifacts only at node shutdown."""

    def __init__(self) -> None:
        super().__init__("recording_node")
        self.declare_parameter("recording_config", Parameter.Type.STRING)
        self.declare_parameter("camera_config", Parameter.Type.STRING)
        self._callback_group = ReentrantCallbackGroup()
        self._bridge = CvBridge()
        self._config = load_recording_config(
            str(self.get_parameter("recording_config").value),
            str(self.get_parameter("camera_config").value),
        )
        topics = recording_topics(self._config)
        if not topics or any(not topic.startswith("/") for topic in topics):
            raise ValueError("Recording topics must be absolute ROS topic names")
        self._save_artifacts_enabled = True
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._session = RecordingSession(
            run_id=run_id,
            artifact_path=self._config.artifact_root / run_id,
            started_at_iso=datetime.now(UTC).isoformat(),
        )
        self._runtime()
        self.get_logger().info(
            f"Recording interface ready: artifact_root={self._config.artifact_root}"
        )

    def _runtime(self) -> None:
        self._planning_cloud_pub = self.create_publisher(
            PointCloud2,
            "/sonopet/captured_planning_cloud",
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )
        self._rgb_subscriptions = [
            self.create_subscription(
                Image,
                camera.rgb_topic,
                lambda msg, key=camera.key: self._session.rgb_frames.append(
                    RgbFrame(
                        camera_key=key,
                        stamp_sec=msg.header.stamp.sec,
                        stamp_nanosec=msg.header.stamp.nanosec,
                        frame_bgr=self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"),
                    )
                ),
                10,
                callback_group=self._callback_group,
            )
            for camera in self._config.cameras
        ]
        self._audio_subscription = self.create_subscription(
            AudioChunk,
            self._config.audio_topic,
            lambda msg: self._session.audio_packets.append(
                AudioPacket(
                    stamp_sec=msg.header.stamp.sec,
                    stamp_nanosec=msg.header.stamp.nanosec,
                    sample_rate_hz=int(msg.sample_rate_hz),
                    channels=int(msg.channels),
                    encoding=str(msg.encoding),
                    samples=list(msg.samples),
                )
            ),
            10,
            callback_group=self._callback_group,
        )
        self._record_server = ActionServer(
            self,
            RecordExperiment,
            "/sonopet/record_experiment",
            self._execute_record,
            callback_group=self._callback_group,
        )
        self._capture_server = ActionServer(
            self,
            CapturePointCloud,
            "/sonopet/capture_pointcloud",
            self._execute_capture,
            callback_group=self._callback_group,
        )
        self._artifact_saving_service = self.create_service(
            SetBool,
            "/sonopet/set_artifact_saving",
            self._set_artifact_saving,
            callback_group=self._callback_group,
        )
        self._startup_capture_timer = self.create_timer(
            5.0,
            lambda: (
                self.destroy_timer(self._startup_capture_timer),
                self._capture_pointcloud("session_begin", True, self._save_artifacts_enabled, None),
            ),
            callback_group=self._callback_group,
        )

    def _execute_record(self, goal_handle):
        feedback = RecordExperiment.Feedback()
        feedback.phase = "session_running"
        goal_handle.publish_feedback(feedback)
        if goal_handle.request.run_id.strip():
            self._session.run_id = goal_handle.request.run_id.strip()
            self._session.artifact_path = self._config.artifact_root / self._session.run_id
        goal_handle.succeed()
        result = RecordExperiment.Result()
        result.success = True
        result.message = "Recording node lifecycle is the active session."
        result.artifact_path = str(self._session.artifact_path)
        return result

    def _execute_capture(self, goal_handle):
        label = goal_handle.request.label.strip()
        if not label:
            label = datetime.now(UTC).strftime("rescan_%Y%m%dT%H%M%SZ")
        scans = self._capture_pointcloud(
            label,
            bool(goal_handle.request.publish_planning_cloud),
            bool(goal_handle.request.save_artifacts),
            goal_handle,
        )
        goal_handle.succeed()
        result = CapturePointCloud.Result()
        result.success = True
        result.message = f"Captured {len(scans)} pointcloud scans."
        result.camera_keys = [scan.camera_key for scan in scans]
        result.artifact_paths = [str(scan.artifact_path) for scan in scans]
        result.point_counts = [scan.point_count for scan in scans]
        return result

    def _set_artifact_saving(self, request: SetBool.Request, response: SetBool.Response):
        self._save_artifacts_enabled = bool(request.data)
        response.success = True
        response.message = (
            "Session artifacts will be saved."
            if request.data
            else "Session artifacts will be discarded."
        )
        return response

    def _capture_pointcloud(
        self,
        label: str,
        publish_planning_cloud: bool,
        save_to_session: bool,
        goal_handle,
    ) -> list[PointCloudScan]:
        scans: list[PointCloudScan] = []
        self._save_artifacts_enabled = bool(save_to_session)
        for camera in self._config.cameras:
            if goal_handle is not None:
                feedback = CapturePointCloud.Feedback()
                feedback.phase = f"{camera.key}_{label}"
                goal_handle.publish_feedback(feedback)
            client = self.create_client(
                SetParameters,
                camera.parameter_service,
                callback_group=self._callback_group,
            )
            if not client.wait_for_service(timeout_sec=2.0):
                raise RuntimeError(f"Parameter service unavailable: {camera.parameter_service}")
            request = SetParameters.Request()
            request.parameters = [
                Parameter("pointcloud.enable", Parameter.Type.BOOL, True).to_parameter_msg()
            ]
            response = wait_future(client.call_async(request))
            failures = [result.reason for result in response.results if not result.successful]
            if failures:
                raise RuntimeError("; ".join(failures))
            received = Event()
            cloud: dict[str, PointCloud2] = {}

            def _on_cloud(msg: PointCloud2) -> None:
                cloud["msg"] = msg
                received.set()

            subscription = self.create_subscription(
                PointCloud2,
                camera.pointcloud_topic,
                _on_cloud,
                qos_profile_sensor_data,
                callback_group=self._callback_group,
            )
            if not received.wait(timeout=self._config.snapshot_timeout_sec):
                raise RuntimeError(f"Timed out waiting for {camera.pointcloud_topic}")
            self.destroy_subscription(subscription)
            request.parameters = [
                Parameter("pointcloud.enable", Parameter.Type.BOOL, False).to_parameter_msg()
            ]
            response = wait_future(client.call_async(request))
            failures = [result.reason for result in response.results if not result.successful]
            if failures:
                raise RuntimeError("; ".join(failures))
            self.destroy_client(client)
            point_count = sum(
                1
                for _ in point_cloud2.read_points(
                    cloud["msg"],
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
            )
            scan = PointCloudScan(
                label=label,
                camera_key=camera.key,
                cloud=cloud["msg"],
                point_count=point_count,
            )
            scans.append(scan)
            self._session.scans.append(scan)
            if publish_planning_cloud and camera.key == "in_hand":
                self._planning_cloud_pub.publish(cloud["msg"])
                self.get_logger().info(
                    f"Published planning cloud from {camera.key} with {point_count} points."
                )
        return scans

    def _save_artifacts(self) -> None:
        self._session.artifact_path.mkdir(parents=True, exist_ok=False)
        for camera in self._config.cameras:
            _camera_dir(self._session.artifact_path, camera.key).mkdir(parents=True, exist_ok=True)
        audio_dir = self._session.artifact_path / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        for scan in self._session.scans:
            scan_path = _camera_dir(self._session.artifact_path, scan.camera_key) / f"{scan.label}.pcd"
            scan.artifact_path = scan_path
            write_pointcloud_pcd(scan_path, scan.cloud)

        rgb_writers = {camera.key: None for camera in self._config.cameras}
        rgb_counts = {camera.key: 0 for camera in self._config.cameras}
        timestamp_streams = {
            camera.key: (
                _camera_dir(self._session.artifact_path, camera.key) / "rgb_timestamps.csv"
            ).open("w", encoding="utf-8")
            for camera in self._config.cameras
        }
        for stream in timestamp_streams.values():
            stream.write("frame_index,stamp_sec,stamp_nanosec\n")
        for frame in self._session.rgb_frames:
            camera_dir = _camera_dir(self._session.artifact_path, frame.camera_key)
            if rgb_writers[frame.camera_key] is None:
                height, width = frame.frame_bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(camera_dir / "rgb.avi"),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    self._config.video_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open RGB video writer: {camera_dir / 'rgb.avi'}")
                rgb_writers[frame.camera_key] = writer
            rgb_writers[frame.camera_key].write(frame.frame_bgr)
            timestamp_streams[frame.camera_key].write(
                f"{rgb_counts[frame.camera_key]},{frame.stamp_sec},{frame.stamp_nanosec}\n"
            )
            rgb_counts[frame.camera_key] += 1

        audio_meta = {
            "chunks": 0,
            "samples": 0,
            "sample_rate_hz": None,
            "channels": None,
            "encoding": "S16_LE",
        }
        wav = None
        if self._session.audio_packets:
            first_packet = self._session.audio_packets[0]
            wav = wave.open(str(audio_dir / "audio.wav"), "wb")
            wav.setnchannels(first_packet.channels)
            wav.setsampwidth(2)
            wav.setframerate(first_packet.sample_rate_hz)
            audio_meta.update(
                {
                    "sample_rate_hz": first_packet.sample_rate_hz,
                    "channels": first_packet.channels,
                    "encoding": first_packet.encoding,
                }
            )
            for packet in self._session.audio_packets:
                if (
                    packet.encoding != first_packet.encoding
                    or packet.sample_rate_hz != first_packet.sample_rate_hz
                    or packet.channels != first_packet.channels
                ):
                    raise ValueError("Audio format changed during recording")
                wav.writeframes(array("h", (int(value) for value in packet.samples)).tobytes())
                audio_meta["chunks"] += 1
                audio_meta["samples"] += len(packet.samples)

        write_json(audio_dir / "audio_meta.json", audio_meta)
        write_json(
            self._session.artifact_path / "manifest.json",
            {
                "run_id": self._session.run_id,
                "artifact_path": str(self._session.artifact_path),
                "started_at": self._session.started_at_iso,
                "ended_at": datetime.now(UTC).isoformat(),
                "pointcloud_scans": [
                    {
                        "label": scan.label,
                        "camera": scan.camera_key,
                        "path": str(scan.artifact_path.relative_to(self._session.artifact_path)),
                        "point_count": scan.point_count,
                    }
                    for scan in self._session.scans
                ],
                "rgb_video": {
                    camera.key: {
                        "video": str(_camera_dir(Path(), camera.key) / "rgb.avi"),
                        "timestamps": str(_camera_dir(Path(), camera.key) / "rgb_timestamps.csv"),
                        "frames": rgb_counts[camera.key],
                    }
                    for camera in self._config.cameras
                },
                "audio": {
                    "wav": "audio/audio.wav",
                    "metadata": "audio/audio_meta.json",
                    **audio_meta,
                },
            },
        )
        for writer in rgb_writers.values():
            if writer is not None:
                writer.release()
        for stream in timestamp_streams.values():
            stream.close()
        if wav is not None:
            wav.close()

    def destroy_node(self) -> bool:
        for subscription in self._rgb_subscriptions:
            self.destroy_subscription(subscription)
        self.destroy_subscription(self._audio_subscription)
        self._capture_pointcloud("session_end", False, self._save_artifacts_enabled, None)
        if self._save_artifacts_enabled:
            self._save_artifacts()
        return super().destroy_node()


def wait_future(future):
    event = Event()
    future.add_done_callback(lambda _future: event.set())
    event.wait()
    return future.result()


def main() -> None:
    rclpy.init()
    node = RecordingNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    executor.remove_node(node)
    node.destroy_node()
    rclpy.shutdown()

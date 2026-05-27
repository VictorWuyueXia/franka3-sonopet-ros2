from __future__ import annotations

import shutil
import wave
from array import array
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import cv2
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
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from fr3_sonopet_recording.artifact_writers import write_json, write_pointcloud_pcd
from fr3_sonopet_recording.recording_config import load_recording_config, recording_topics


@dataclass
class PointCloudScan:
    """One D405 pointcloud artifact captured at a recording boundary."""

    label: str
    camera_key: str
    cloud: PointCloud2
    point_count: int
    artifact_path: Path = Path()


@dataclass
class ActiveRecordingRun:
    """Open files for one cutting interval."""

    index: int
    started_at: str
    artifact_path: Path
    rgb_paths: dict[str, Path]
    rgb_writers: dict[str, cv2.VideoWriter | None]
    rgb_counts: dict[str, int]
    audio_path: Path
    audio_wav: wave.Wave_write | None = None
    audio_sample_rate_hz: int | None = None
    audio_channels: int | None = None
    audio_encoding: str = "S16_LE"
    pointcloud_scans: list[PointCloudScan] = field(default_factory=list)


def _camera_dir(artifact_path: Path, camera_key: str) -> Path:
    if camera_key == "in_hand":
        return artifact_path / "camera_in_hand"
    if camera_key == "fixed":
        return artifact_path / "camera_fixed"
    raise ValueError(f"Unsupported recording camera: {camera_key}")


def _metadata_timestamp() -> str:
    now = datetime.now(UTC)
    centiseconds = now.microsecond // 10_000
    return f"{now:%Y-%m-%d:%H-%M-%S}-{centiseconds // 10}-{centiseconds % 10}"


class RecordingNode(Node):
    """Record each cutting interval directly to experiment artifacts."""

    def __init__(self) -> None:
        super().__init__("recording_node")
        self.declare_parameter("recording_config", Parameter.Type.STRING)
        self.declare_parameter("camera_config", Parameter.Type.STRING)
        self.declare_parameter("bringup_config_dir", Parameter.Type.STRING)
        self.declare_parameter("description_config_dir", Parameter.Type.STRING)
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
        self._run_count = 0
        self._active_run: ActiveRecordingRun | None = None
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._artifact_path = self._config.artifact_root / run_id
        self._prepare_experiment_folder()
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
                lambda msg, key=camera.key: self._write_rgb_frame(key, msg),
                10,
                callback_group=self._callback_group,
            )
            for camera in self._config.cameras
        ]
        self._audio_subscription = self.create_subscription(
            AudioChunk,
            self._config.audio_topic,
            self._write_audio_packet,
            10,
            callback_group=self._callback_group,
        )
        self._cutting_subscription = self.create_subscription(
            Bool,
            self._config.cutting_topic,
            self._on_cutting_flag,
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

    def _prepare_experiment_folder(self) -> None:
        self._artifact_path.mkdir(parents=True, exist_ok=True)
        for camera in self._config.cameras:
            _camera_dir(self._artifact_path, camera.key).mkdir(parents=True, exist_ok=True)
        (self._artifact_path / "audio").mkdir(parents=True, exist_ok=True)
        self._copy_config_artifacts()

    def _copy_config_artifacts(self) -> None:
        for parameter_name, folder_name in (
            ("bringup_config_dir", "bringup"),
            ("description_config_dir", "description"),
        ):
            source_dir = Path(str(self.get_parameter(parameter_name).value))
            target_dir = self._artifact_path / "config" / folder_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for yaml_path in sorted(source_dir.glob("*.yaml")):
                shutil.copy2(yaml_path, target_dir / yaml_path.name)

    def _execute_record(self, goal_handle):
        feedback = RecordExperiment.Feedback()
        feedback.phase = "waiting_for_cutting"
        goal_handle.publish_feedback(feedback)
        if goal_handle.request.run_id.strip():
            self._artifact_path = self._config.artifact_root / goal_handle.request.run_id.strip()
            self._prepare_experiment_folder()
        goal_handle.succeed()
        result = RecordExperiment.Result()
        result.success = True
        result.message = "Recording starts and stops from /sonopet/cutting."
        result.artifact_path = str(self._artifact_path)
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

    def _on_cutting_flag(self, msg: Bool) -> None:
        if msg.data and self._active_run is None:
            self._start_recording_run()
        if not msg.data and self._active_run is not None:
            self._stop_recording_run()

    def _start_recording_run(self) -> None:
        rgb_paths = {
            camera.key: self._next_existing_name(_camera_dir(self._artifact_path, camera.key) / "rgb.avi")
            for camera in self._config.cameras
        }
        self._active_run = ActiveRecordingRun(
            index=self._run_count,
            started_at=_metadata_timestamp(),
            artifact_path=self._artifact_path,
            rgb_paths=rgb_paths,
            rgb_writers={camera.key: None for camera in self._config.cameras},
            rgb_counts={camera.key: 0 for camera in self._config.cameras},
            audio_path=self._next_existing_name(self._artifact_path / "audio" / "audio.wav"),
        )
        self._run_count += 1
        self._active_run.pointcloud_scans.extend(
            self._capture_pointcloud("pointcloud_start", False, True, None)
        )

    def _write_rgb_frame(self, camera_key: str, msg: Image) -> None:
        if self._active_run is None:
            return
        frame_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        writer = self._active_run.rgb_writers[camera_key]
        if writer is None:
            height, width = frame_bgr.shape[:2]
            writer = cv2.VideoWriter(
                str(self._active_run.rgb_paths[camera_key]),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self._config.video_fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open RGB video writer: {self._active_run.rgb_paths[camera_key]}")
            self._active_run.rgb_writers[camera_key] = writer
        writer.write(frame_bgr)
        self._active_run.rgb_counts[camera_key] += 1

    def _write_audio_packet(self, msg: AudioChunk) -> None:
        if self._active_run is None:
            return
        if self._active_run.audio_wav is None:
            self._active_run.audio_sample_rate_hz = int(msg.sample_rate_hz)
            self._active_run.audio_channels = int(msg.channels)
            self._active_run.audio_encoding = str(msg.encoding)
            self._active_run.audio_wav = wave.open(str(self._active_run.audio_path), "wb")
            self._active_run.audio_wav.setnchannels(self._active_run.audio_channels)
            self._active_run.audio_wav.setsampwidth(2)
            self._active_run.audio_wav.setframerate(self._active_run.audio_sample_rate_hz)
        elif (
            self._active_run.audio_sample_rate_hz != int(msg.sample_rate_hz)
            or self._active_run.audio_channels != int(msg.channels)
            or self._active_run.audio_encoding != str(msg.encoding)
        ):
            raise ValueError("Audio format changed during recording")
        self._active_run.audio_wav.writeframes(
            array("h", (int(value) for value in msg.samples)).tobytes()
        )

    def _stop_recording_run(self) -> None:
        active_run = self._active_run
        active_run.pointcloud_scans.extend(self._capture_pointcloud("pointcloud_stop", False, True, None))
        stopped_at = _metadata_timestamp()
        for writer in active_run.rgb_writers.values():
            if writer is not None:
                writer.release()
        if active_run.audio_wav is not None:
            active_run.audio_wav.close()
        self._write_run_metadata(active_run, stopped_at)
        self._active_run = None

    def _write_run_metadata(self, active_run: ActiveRecordingRun, stopped_at: str) -> None:
        audio_meta_path = self._next_existing_name(self._artifact_path / "audio" / "audio_meta.json")
        manifest_path = self._next_existing_name(self._artifact_path / "manifest.json")
        audio_meta = {
            "started_at": active_run.started_at,
            "stopped_at": stopped_at,
            "sample_rate_hz": active_run.audio_sample_rate_hz,
            "wav": str(active_run.audio_path.relative_to(self._artifact_path)),
        }
        write_json(audio_meta_path, audio_meta)
        write_json(
            manifest_path,
            {
                "started_at": active_run.started_at,
                "stopped_at": stopped_at,
                "rgb_video": {
                    camera.key: {
                        "video": str(active_run.rgb_paths[camera.key].relative_to(self._artifact_path)),
                        "frame_rate": self._config.video_fps,
                        "frames": active_run.rgb_counts[camera.key],
                    }
                    for camera in self._config.cameras
                },
                "audio": {
                    "metadata": str(audio_meta_path.relative_to(self._artifact_path)),
                    **audio_meta,
                },
                "pointcloud_scans": [
                    {
                        "label": scan.label,
                        "camera": scan.camera_key,
                        "path": str(scan.artifact_path.relative_to(self._artifact_path)),
                        "point_count": scan.point_count,
                    }
                    for scan in active_run.pointcloud_scans
                ],
            },
        )

    def _capture_pointcloud(
        self,
        label: str,
        publish_planning_cloud: bool,
        save_to_session: bool,
        goal_handle,
    ) -> list[PointCloudScan]:
        scans: list[PointCloudScan] = []
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

            def _on_cloud(
                msg: PointCloud2,
                cloud_buffer: dict[str, PointCloud2] = cloud,
                received_event: Event = received,
            ) -> None:
                cloud_buffer["msg"] = msg
                received_event.set()

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
            if save_to_session:
                scan.artifact_path = self._next_existing_name(
                    _camera_dir(self._artifact_path, scan.camera_key) / f"{scan.label}.pcd"
                )
                write_pointcloud_pcd(scan.artifact_path, scan.cloud)
            scans.append(scan)
            if publish_planning_cloud and camera.key == "in_hand":
                self._planning_cloud_pub.publish(cloud["msg"])
                self.get_logger().info(
                    f"Published planning cloud from {camera.key} with {point_count} points."
                )
        return scans

    def _next_existing_name(self, path: Path) -> Path:
        if not path.exists():
            return path
        index = 1
        while True:
            candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
            if not candidate.exists():
                return candidate
            index += 1

    def _discard_experiment_folder(self) -> None:
        if self._artifact_path.exists():
            shutil.rmtree(self._artifact_path)

    def destroy_node(self) -> bool:
        for subscription in self._rgb_subscriptions:
            self.destroy_subscription(subscription)
        self.destroy_subscription(self._audio_subscription)
        self.destroy_subscription(self._cutting_subscription)
        if self._active_run is not None:
            self._stop_recording_run()
        if not self._save_artifacts_enabled:
            self._discard_experiment_folder()
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
    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

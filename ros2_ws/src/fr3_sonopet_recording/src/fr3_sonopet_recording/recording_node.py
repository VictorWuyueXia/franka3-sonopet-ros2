from __future__ import annotations

import shutil
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock

import cv2
import rclpy
from cv_bridge import CvBridge
from fr3_sonopet_interfaces.action import CapturePointCloud, RecordExperiment
from fr3_sonopet_interfaces.msg import AudioChunk
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from fr3_sonopet_recording.artifact_writers import write_json
from fr3_sonopet_recording.recording_config import (
    experiment_run_id,
    load_recording_config,
    local_timestamp_label,
    recording_run_label,
    recording_topics,
    wall_clock_timestamp,
)

ARTIFACT_PATH_TOPIC = "/sonopet/artifact_path"


@dataclass
class PointCloudScan:
    """One D405 pointcloud artifact captured at a recording boundary."""

    label: str
    camera_key: str
    point_count: int
    timestamp: float
    timestamp_local: str
    artifact_path: Path = Path()


@dataclass
class ActiveRecordingRun:
    """Open files for one cutting interval."""

    label: str
    index: int
    started_at: float
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


def _relative_artifact_path(root: Path, path: Path) -> str:
    if path == Path():
        raise ValueError(f"Artifact path is missing under {root}")
    return str(path.relative_to(root))


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
        self._run_lock = Lock()
        self._run_id = experiment_run_id()
        self._artifact_path = self._config.artifact_root / self._run_id
        self._prepare_session_folder()
        self._runtime()
        self.get_logger().info(
            f"Recording interface ready: run_id={self._run_id}, "
            f"artifact_path={self._artifact_path}"
        )

    def _runtime(self) -> None:
        artifact_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._artifact_path_pub = self.create_publisher(
            String,
            ARTIFACT_PATH_TOPIC,
            artifact_qos,
        )
        self._artifact_path_pub.publish(String(data=str(self._artifact_path)))
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
            "/realsense/record_experiment",
            self._execute_record,
            callback_group=self._callback_group,
        )
        self._capture_client = ActionClient(
            self,
            CapturePointCloud,
            "/realsense/capture_pointcloud",
            callback_group=self._callback_group,
        )
        self._artifact_saving_service = self.create_service(
            SetBool,
            "/set_artifact_saving",
            self._set_artifact_saving,
            callback_group=self._callback_group,
        )
        self._startup_capture_timer = self.create_timer(
            1.0,
            self._capture_startup_planning_cloud,
            callback_group=self._callback_group,
        )

    def _capture_startup_planning_cloud(self) -> None:
        # The startup cloud gives RViz and raster planning a visible surface before operator input.
        self.destroy_timer(self._startup_capture_timer)
        self._startup_capture_timer = None
        self._capture_pointcloud("pointcloud_start", True, True, None)

    def _prepare_session_folder(self) -> None:
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
        goal_handle.succeed()
        result = RecordExperiment.Result()
        result.success = True
        result.message = "Recording starts and stops from /sonopet/cutting."
        result.artifact_path = str(self._artifact_path)
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
        with self._run_lock:
            recording_active = self._active_run is not None
        if msg.data and not recording_active:
            self._start_recording_run()
        if not msg.data and recording_active:
            self._stop_recording_run()

    def _start_recording_run(self) -> None:
        run_index = self._run_count + 1
        run_label = recording_run_label(run_index)
        rgb_paths = {
            camera.key: self._next_existing_name(
                _camera_dir(self._artifact_path, camera.key) / "rgb.avi"
            )
            for camera in self._config.cameras
        }
        active_run = ActiveRecordingRun(
            label=run_label,
            index=run_index - 1,
            started_at=wall_clock_timestamp(),
            artifact_path=self._artifact_path,
            rgb_paths=rgb_paths,
            rgb_writers={camera.key: None for camera in self._config.cameras},
            rgb_counts={camera.key: 0 for camera in self._config.cameras},
            audio_path=self._next_existing_name(self._artifact_path / "audio" / "audio.wav"),
        )
        with self._run_lock:
            if self._active_run is not None:
                return
            self._active_run = active_run
            self._run_count += 1
        start_scans = self._capture_pointcloud(
            "pointcloud_start",
            False,
            True,
            None,
            active_run.artifact_path,
        )
        with self._run_lock:
            if self._active_run is active_run:
                active_run.pointcloud_scans.extend(start_scans)

    def _write_rgb_frame(self, camera_key: str, msg: Image) -> None:
        frame_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._run_lock:
            if self._active_run is None:
                return
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
                    raise RuntimeError(
                        f"Could not open RGB video writer: {self._active_run.rgb_paths[camera_key]}"
                    )
                self._active_run.rgb_writers[camera_key] = writer
            writer.write(frame_bgr)
            self._active_run.rgb_counts[camera_key] += 1

    def _write_audio_packet(self, msg: AudioChunk) -> None:
        samples = array("h", (int(value) for value in msg.samples)).tobytes()
        with self._run_lock:
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
            self._active_run.audio_wav.writeframes(samples)

    def _stop_recording_run(self) -> None:
        stopped_at = wall_clock_timestamp()
        with self._run_lock:
            active_run = self._active_run
        if active_run is None:
            return
        boundary_error: Exception | None = None
        try:
            active_run.pointcloud_scans.extend(
                self._capture_pointcloud(
                    "pointcloud_stop",
                    False,
                    True,
                    None,
                    active_run.artifact_path,
                )
            )
        except Exception as exc:
            boundary_error = exc
            self.get_logger().error(
                f"Stop pointcloud capture failed after recording stopped: {exc}"
            )
        with self._run_lock:
            self._active_run = None
        self._close_run_files(active_run)
        self._write_run_metadata(active_run, stopped_at)
        if boundary_error is not None:
            raise boundary_error

    def _close_run_files(self, active_run: ActiveRecordingRun) -> None:
        """Finalize C-backed media handles before metadata exposes the artifact paths."""
        for writer in active_run.rgb_writers.values():
            if writer is not None:
                writer.release()
        if active_run.audio_wav is not None:
            active_run.audio_wav.close()

    def _write_run_metadata(self, active_run: ActiveRecordingRun, stopped_at: float) -> None:
        run_root = active_run.artifact_path
        audio_meta_path = self._next_existing_name(run_root / "audio" / "audio_meta.json")
        manifest_path = self._next_existing_name(run_root / "manifest.json")
        audio_meta = {
            "run": active_run.label,
            "started_at": active_run.started_at,
            "started_at_local": local_timestamp_label(active_run.started_at),
            "stopped_at": stopped_at,
            "stopped_at_local": local_timestamp_label(stopped_at),
            "sample_rate_hz": active_run.audio_sample_rate_hz,
            "wav": _relative_artifact_path(run_root, active_run.audio_path),
        }
        manifest = {
            "run": active_run.label,
            "started_at": active_run.started_at,
            "started_at_local": local_timestamp_label(active_run.started_at),
            "stopped_at": stopped_at,
            "stopped_at_local": local_timestamp_label(stopped_at),
            "rgb_video": {
                camera.key: {
                    "video": _relative_artifact_path(run_root, active_run.rgb_paths[camera.key]),
                    "frame_rate": self._config.video_fps,
                    "frames": active_run.rgb_counts[camera.key],
                }
                for camera in self._config.cameras
            },
            "audio": {
                "metadata": _relative_artifact_path(run_root, audio_meta_path),
                **audio_meta,
            },
            "pointcloud_scans": [
                {
                    "label": scan.label,
                    "camera": scan.camera_key,
                    "path": _relative_artifact_path(run_root, scan.artifact_path),
                    "point_count": scan.point_count,
                    "timestamp": scan.timestamp,
                    "timestamp_local": scan.timestamp_local,
                }
                for scan in active_run.pointcloud_scans
            ],
        }
        write_json(manifest_path, manifest)
        write_json(audio_meta_path, audio_meta)

    def _capture_pointcloud(
        self,
        label: str,
        publish_planning_cloud: bool,
        save_to_session: bool,
        _goal_handle,
        artifact_root: Path | None = None,
    ) -> list[PointCloudScan]:
        # Recording delegates all pointcloud work to the dedicated pointcloud action server.
        self._capture_client.wait_for_server()
        goal = CapturePointCloud.Goal()
        goal.label = label
        goal.publish_planning_cloud = bool(publish_planning_cloud)
        goal.save_artifacts = bool(save_to_session)
        goal.artifact_root = str(
            artifact_root if artifact_root is not None else self._artifact_path
        )
        goal_handle = wait_future(self._capture_client.send_goal_async(goal))
        if not goal_handle.accepted:
            raise RuntimeError("Pointcloud capture goal was rejected")
        action_result = wait_future(goal_handle.get_result_async()).result
        if not action_result.success:
            raise RuntimeError(action_result.message)
        return [
            PointCloudScan(
                label=label,
                camera_key=camera_key,
                artifact_path=Path(artifact_path),
                point_count=int(point_count),
                timestamp=float(timestamp),
                timestamp_local=timestamp_local,
            )
            for camera_key, artifact_path, point_count, timestamp, timestamp_local in zip(
                action_result.camera_keys,
                action_result.artifact_paths,
                action_result.point_counts,
                action_result.timestamps,
                action_result.timestamp_locals,
                strict=True,
            )
            if artifact_path
        ]

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
        if self._startup_capture_timer is not None:
            self.destroy_timer(self._startup_capture_timer)
            self._startup_capture_timer = None
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

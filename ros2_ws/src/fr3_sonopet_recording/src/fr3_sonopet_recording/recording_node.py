from __future__ import annotations

import shutil
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock
from time import monotonic

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
SONOPET_READY_TOPIC = "/sonopet/ready"
VIDEO_STREAM_MAX_AGE_S = 1.0


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
    audio_next_chunk_index: int | None = None
    audio_chunks_received: int = 0
    audio_chunks_gap_filled_source: int = 0
    audio_chunks_gap_filled_recorder: int = 0
    audio_chunks_written: int = 0
    audio_samples_written: int = 0
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
        self._boundary_lock = Lock()
        self._sonopet_ready = False
        self._last_rgb_received_s: dict[str, float | None] = {}
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
        self._last_rgb_received_s = {camera.key: None for camera in self._config.cameras}
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
        self._sonopet_ready_subscription = self.create_subscription(
            Bool,
            SONOPET_READY_TOPIC,
            self._on_sonopet_ready,
            artifact_qos,
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
        self._capture_pointcloud("pointcloud_start", True, False, None)

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
        # Boundary work owns the physical D405s, so transitions cannot overlap with each other.
        with self._boundary_lock:
            with self._run_lock:
                recording_active = self._active_run is not None
            if msg.data and not recording_active:
                if not self._require_workflow_ready():
                    return
                self._start_recording_run()
            if not msg.data and recording_active:
                self._stop_recording_run()

    def _on_sonopet_ready(self, msg: Bool) -> None:
        self._sonopet_ready = bool(msg.data)

    def _require_workflow_ready(self) -> bool:
        if not self._sonopet_ready:
            self._report_blocked_run(
                f"SONOPET ERROR: Sonopet is not connected or not ready on {SONOPET_READY_TOPIC}; "
                "not starting this cutting run."
            )
            return False
        now = monotonic()
        stale = [
            camera.key
            for camera in self._config.cameras
            if self._last_rgb_received_s.get(camera.key) is None
            or now - float(self._last_rgb_received_s[camera.key]) > VIDEO_STREAM_MAX_AGE_S
        ]
        if stale:
            self._report_blocked_run(
                f"VIDEO ERROR: no fresh RGB frames from camera(s) {', '.join(stale)} within "
                f"{VIDEO_STREAM_MAX_AGE_S:.1f}s; not starting this cutting run."
            )
            return False
        return True

    def _report_blocked_run(self, message: str) -> None:
        self.get_logger().error(message)

    def _start_recording_run(self) -> None:
        run_index = self._run_count
        run_label = recording_run_label(run_index)
        rgb_paths = {
            camera.key: _camera_dir(self._artifact_path, camera.key) / f"rgb_{run_index}.avi"
            for camera in self._config.cameras
        }
        active_run = ActiveRecordingRun(
            label=run_label,
            index=run_index,
            started_at=0.0,
            artifact_path=self._artifact_path,
            rgb_paths=rgb_paths,
            rgb_writers={camera.key: None for camera in self._config.cameras},
            rgb_counts={camera.key: 0 for camera in self._config.cameras},
            audio_path=self._artifact_path / "audio" / f"audio_{run_index}.wav",
        )
        try:
            start_scans = self._capture_pointcloud(
                "pointcloud_start",
                False,
                True,
                None,
                active_run.artifact_path,
            )
        except Exception as exc:
            self.get_logger().error(
                f"RECORDING ERROR: {run_label} was not recorded because "
                f"pointcloud_start failed: {exc}"
            )
            raise
        # Media recording opens only after the pointcloud start boundary is complete.
        active_run.started_at = wall_clock_timestamp()
        active_run.pointcloud_scans.extend(start_scans)
        with self._run_lock:
            if self._active_run is not None:
                raise RuntimeError("Recording run became active during pointcloud_start")
            self._active_run = active_run
            self._run_count += 1

    def _write_rgb_frame(self, camera_key: str, msg: Image) -> None:
        frame_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._last_rgb_received_s[camera_key] = monotonic()
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
        samples = array("h", (int(value) for value in msg.samples))
        channels = int(msg.channels)
        if channels <= 0 or not samples or len(samples) % channels != 0:
            raise ValueError("Audio chunk samples must be nonempty and channel-aligned")
        with self._run_lock:
            if self._active_run is None:
                return
            active_run = self._active_run
            self._prepare_audio_stream(active_run, msg)
            chunk_index = int(msg.chunk_index)
            if active_run.audio_next_chunk_index is None:
                active_run.audio_next_chunk_index = chunk_index
            if chunk_index < active_run.audio_next_chunk_index:
                raise ValueError(
                    f"Audio chunk index moved backward: got {chunk_index}, "
                    f"expected {active_run.audio_next_chunk_index}"
                )
            missing_chunks = chunk_index - active_run.audio_next_chunk_index
            if missing_chunks:
                zero_samples = array("h", [0]) * len(samples)
                for _ in range(missing_chunks):
                    self._write_audio_samples(active_run, zero_samples)
                active_run.audio_chunks_gap_filled_recorder += missing_chunks
                self.get_logger().warning(
                    f"Filled {missing_chunks} missing recorder audio chunks with silence"
                )
            if msg.gap_fill:
                active_run.audio_chunks_gap_filled_source += 1
            active_run.audio_chunks_received += 1
            self._write_audio_samples(active_run, samples)
            active_run.audio_next_chunk_index = chunk_index + 1

    def _prepare_audio_stream(self, active_run: ActiveRecordingRun, msg: AudioChunk) -> None:
        if active_run.audio_wav is None:
            active_run.audio_sample_rate_hz = int(msg.sample_rate_hz)
            active_run.audio_channels = int(msg.channels)
            active_run.audio_encoding = str(msg.encoding)
            active_run.audio_wav = wave.open(str(active_run.audio_path), "wb")
            active_run.audio_wav.setnchannels(active_run.audio_channels)
            active_run.audio_wav.setsampwidth(2)
            active_run.audio_wav.setframerate(active_run.audio_sample_rate_hz)
        elif (
            active_run.audio_sample_rate_hz != int(msg.sample_rate_hz)
            or active_run.audio_channels != int(msg.channels)
            or active_run.audio_encoding != str(msg.encoding)
        ):
            raise ValueError("Audio format changed during recording")

    def _write_audio_samples(self, active_run: ActiveRecordingRun, samples: array) -> None:
        active_run.audio_wav.writeframes(samples.tobytes())
        active_run.audio_chunks_written += 1
        active_run.audio_samples_written += len(samples)

    def _stop_recording_run(self) -> None:
        stopped_at = wall_clock_timestamp()
        with self._run_lock:
            active_run = self._active_run
            self._active_run = None
        if active_run is None:
            return
        self._close_run_files(active_run)
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
                f"RECORDING ERROR: {active_run.label} is incomplete because "
                f"pointcloud_stop failed after recording stopped: {exc}"
            )
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
        audio_meta_path = run_root / "audio" / f"audio_meta_{active_run.index}.json"
        manifest_path = run_root / f"manifest_{active_run.index}.json"
        audio_meta = {
            "run": active_run.label,
            "started_at": active_run.started_at,
            "started_at_local": local_timestamp_label(active_run.started_at),
            "stopped_at": stopped_at,
            "stopped_at_local": local_timestamp_label(stopped_at),
            "sample_rate_hz": active_run.audio_sample_rate_hz,
            "chunks_received": active_run.audio_chunks_received,
            "chunks_gap_filled_source": active_run.audio_chunks_gap_filled_source,
            "chunks_gap_filled_recorder": active_run.audio_chunks_gap_filled_recorder,
            "chunks_written": active_run.audio_chunks_written,
            "samples_written": active_run.audio_samples_written,
            "duration_s": (
                active_run.audio_samples_written
                / float(active_run.audio_sample_rate_hz * active_run.audio_channels)
                if active_run.audio_sample_rate_hz and active_run.audio_channels
                else 0.0
            ),
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

    def _discard_experiment_folder(self) -> None:
        if self._artifact_path.exists():
            shutil.rmtree(self._artifact_path)

    def destroy_node(self) -> bool:
        for subscription in self._rgb_subscriptions:
            self.destroy_subscription(subscription)
        self.destroy_subscription(self._audio_subscription)
        self.destroy_subscription(self._cutting_subscription)
        self.destroy_subscription(self._sonopet_ready_subscription)
        if self._startup_capture_timer is not None:
            self.destroy_timer(self._startup_capture_timer)
            self._startup_capture_timer = None
        with self._boundary_lock:
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

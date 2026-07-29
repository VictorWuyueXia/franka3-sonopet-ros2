from __future__ import annotations

import csv
import shutil
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import TextIO

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
VIDEO_STREAM_MAX_AGE_S = 1.0
CAPTURE_ACCEPT_TIMEOUT_S = 5.0
CAPTURE_RESULT_TIMEOUT_S = 30.0


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
    rgb_timestamp_paths: dict[str, Path]
    rgb_writers: dict[str, cv2.VideoWriter | None]
    rgb_timestamp_files: dict[str, TextIO | None]
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
    audio_chunks_stale_dropped: int = 0
    audio_chunks_written: int = 0
    audio_samples_written: int = 0
    pointcloud_scans: list[PointCloudScan] = field(default_factory=list)
    stopped_at: float | None = None


@dataclass(frozen=True)
class BoundaryCaptureTask:
    """Serialized pointcloud capture request that must not block media recording."""

    label: str
    active_run: ActiveRecordingRun | None
    publish_planning_cloud: bool
    save_to_session: bool
    artifact_root: Path | None


@dataclass(frozen=True)
class RgbFrameTask:
    """Copied RGB frame that is safe to write outside the ROS callback."""

    active_run: ActiveRecordingRun
    camera_key: str
    frame_bgr: object
    stamp_sec: int
    stamp_nanosec: int
    received_at: float


@dataclass(frozen=True)
class AudioPacketTask:
    """Copied audio packet that is safe to write outside the ROS callback."""

    active_run: ActiveRecordingRun
    samples: array
    sample_rate_hz: int
    channels: int
    encoding: str
    chunk_index: int
    gap_fill: bool


@dataclass(frozen=True)
class CloseRunMediaTask:
    """Run finalization request that drains all earlier media writes first."""

    active_run: ActiveRecordingRun
    done: Event


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
        self._capture_queue: Queue[BoundaryCaptureTask | None] = Queue()
        self._media_queue: Queue[RgbFrameTask | AudioPacketTask | CloseRunMediaTask | None] = Queue()
        self._capture_worker_stop = Event()
        self._capture_worker = Thread(
            target=self._run_boundary_capture_worker,
            name="recording_boundary_capture",
            daemon=True,
        )
        self._media_worker = Thread(
            target=self._run_media_writer,
            name="recording_media_writer",
            daemon=True,
        )
        self._last_rgb_received_s: dict[str, float | None] = {}
        self._run_id = experiment_run_id()
        self._artifact_path = self._config.artifact_root / self._run_id
        self._prepare_session_folder()
        self._runtime()
        self._media_worker.start()
        self._capture_worker.start()
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
            128,
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
        self._queue_boundary_capture("pointcloud_start", None, True, False, None)

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
                active_label = self._active_run.label if self._active_run is not None else "none"
            self.get_logger().info(
                f"Cutting topic edge: value={bool(msg.data)}, "
                f"recording_active={recording_active}, active_run={active_label}, "
                f"publishers={self._cutting_publishers_label()}"
            )
            try:
                if msg.data and not recording_active:
                    if not self._require_recording_inputs():
                        return
                    self._start_recording_run()
                if not msg.data and recording_active:
                    self._stop_recording_run("cutting_false", True)
                if msg.data and recording_active:
                    self.get_logger().debug(
                        f"Ignored cutting=True because {active_label} is already active."
                    )
                if not msg.data and not recording_active:
                    self.get_logger().debug(
                        "Ignored cutting=False because no recording run is active."
                    )
            except Exception as exc:
                self.get_logger().error(
                    f"RECORDING ERROR: cutting transition failed; recorder remains available: {exc}"
                )

    def _cutting_publishers_label(self) -> str:
        publishers = self.get_publishers_info_by_topic(self._config.cutting_topic)
        return ",".join(f"{info.node_namespace}/{info.node_name}" for info in publishers)

    def _require_recording_inputs(self) -> bool:
        # Fresh RGB confirms the media writers can be opened when raster motion begins.
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
                f"{VIDEO_STREAM_MAX_AGE_S:.1f}s; not starting this cutting run; "
                "no fallback media will be created."
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
        rgb_timestamp_paths = {
            camera.key: _camera_dir(self._artifact_path, camera.key)
            / f"rgb_timestamps_{run_index}.csv"
            for camera in self._config.cameras
        }
        active_run = ActiveRecordingRun(
            label=run_label,
            index=run_index,
            started_at=0.0,
            artifact_path=self._artifact_path,
            rgb_paths=rgb_paths,
            rgb_timestamp_paths=rgb_timestamp_paths,
            rgb_writers={camera.key: None for camera in self._config.cameras},
            rgb_timestamp_files={camera.key: None for camera in self._config.cameras},
            rgb_counts={camera.key: 0 for camera in self._config.cameras},
            audio_path=self._artifact_path / "audio" / f"audio_{run_index}.wav",
        )
        active_run.started_at = wall_clock_timestamp()
        with self._run_lock:
            if self._active_run is not None:
                raise RuntimeError("Recording run became active during run start")
            self._active_run = active_run
            self._run_count += 1
        self.get_logger().info(
            f"Recording run {run_label} active: started_at={active_run.started_at:.6f}; "
            "media writers will open on the next RGB/audio samples."
        )
        self._queue_boundary_capture(
            "pointcloud_start",
            active_run,
            False,
            True,
            active_run.artifact_path,
        )

    def _write_rgb_frame(self, camera_key: str, msg: Image) -> None:
        frame_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._last_rgb_received_s[camera_key] = monotonic()
        with self._run_lock:
            if self._active_run is None:
                return
            self._media_queue.put(
                RgbFrameTask(
                    active_run=self._active_run,
                    camera_key=camera_key,
                    frame_bgr=frame_bgr.copy(),
                    stamp_sec=int(msg.header.stamp.sec),
                    stamp_nanosec=int(msg.header.stamp.nanosec),
                    received_at=wall_clock_timestamp(),
                )
            )

    def _write_audio_packet(self, msg: AudioChunk) -> None:
        samples = array("h", msg.samples)
        channels = int(msg.channels)
        if channels <= 0 or not samples or len(samples) % channels != 0:
            raise ValueError("Audio chunk samples must be nonempty and channel-aligned")
        with self._run_lock:
            if self._active_run is None:
                return
            self._media_queue.put(
                AudioPacketTask(
                    active_run=self._active_run,
                    samples=samples,
                    sample_rate_hz=int(msg.sample_rate_hz),
                    channels=channels,
                    encoding=str(msg.encoding),
                    chunk_index=int(msg.chunk_index),
                    gap_fill=bool(msg.gap_fill),
                )
            )

    def _run_media_writer(self) -> None:
        while True:
            task = self._media_queue.get()
            if task is None:
                return
            try:
                if isinstance(task, RgbFrameTask):
                    self._write_rgb_frame_task(task)
                elif isinstance(task, AudioPacketTask):
                    self._write_audio_packet_task(task)
                else:
                    self._close_run_files(task.active_run)
            except Exception as exc:
                self.get_logger().error(f"RECORDING ERROR: media writer failed: {exc}")
            finally:
                if isinstance(task, CloseRunMediaTask):
                    task.done.set()

    def _write_rgb_frame_task(self, task: RgbFrameTask) -> None:
        active_run = task.active_run
        writer = active_run.rgb_writers[task.camera_key]
        if writer is None:
            height, width = task.frame_bgr.shape[:2]
            writer = cv2.VideoWriter(
                str(active_run.rgb_paths[task.camera_key]),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self._config.video_fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not open RGB video writer: {active_run.rgb_paths[task.camera_key]}"
                )
            timestamp_file = active_run.rgb_timestamp_paths[task.camera_key].open(
                "w", newline="", encoding="utf-8"
            )
            csv.writer(timestamp_file).writerow(
                ["frame_index", "stamp_sec", "stamp_nanosec", "received_at"]
            )
            active_run.rgb_writers[task.camera_key] = writer
            active_run.rgb_timestamp_files[task.camera_key] = timestamp_file
            self.get_logger().info(
                f"Opened RGB writer for {task.camera_key}: "
                f"path={active_run.rgb_paths[task.camera_key]}, size={width}x{height}, "
                f"fps={self._config.video_fps:g}, "
                f"timestamps={active_run.rgb_timestamp_paths[task.camera_key]}"
            )
        frame_index = active_run.rgb_counts[task.camera_key]
        writer.write(task.frame_bgr)
        timestamp_file = active_run.rgb_timestamp_files[task.camera_key]
        if timestamp_file is None:
            raise RuntimeError(f"RGB timestamp file is not open: {task.camera_key}")
        csv.writer(timestamp_file).writerow(
            [frame_index, task.stamp_sec, task.stamp_nanosec, f"{task.received_at:.6f}"]
        )
        active_run.rgb_counts[task.camera_key] += 1

    def _write_audio_packet_task(self, task: AudioPacketTask) -> None:
        active_run = task.active_run
        self._prepare_audio_stream(active_run, task)
        if active_run.audio_next_chunk_index is None:
            active_run.audio_next_chunk_index = task.chunk_index
        if task.chunk_index < active_run.audio_next_chunk_index:
            active_run.audio_chunks_stale_dropped += 1
            self.get_logger().error(
                f"RECORDING ERROR: dropped stale audio chunk for {active_run.label}: "
                f"got={task.chunk_index}, expected={active_run.audio_next_chunk_index}"
            )
            return
        missing_chunks = task.chunk_index - active_run.audio_next_chunk_index
        if missing_chunks:
            self._write_audio_silence(active_run, len(task.samples), missing_chunks)
            active_run.audio_chunks_gap_filled_recorder += missing_chunks
            self.get_logger().warning(
                f"Filled {missing_chunks} missing recorder audio chunks with silence"
            )
        if task.gap_fill:
            active_run.audio_chunks_gap_filled_source += 1
        active_run.audio_chunks_received += 1
        self._write_audio_samples(active_run, task.samples)
        active_run.audio_next_chunk_index = task.chunk_index + 1

    def _prepare_audio_stream(
        self, active_run: ActiveRecordingRun, task: AudioPacketTask
    ) -> None:
        if active_run.audio_wav is None:
            active_run.audio_sample_rate_hz = task.sample_rate_hz
            active_run.audio_channels = task.channels
            active_run.audio_encoding = task.encoding
            active_run.audio_wav = wave.open(str(active_run.audio_path), "wb")
            active_run.audio_wav.setnchannels(active_run.audio_channels)
            active_run.audio_wav.setsampwidth(2)
            active_run.audio_wav.setframerate(active_run.audio_sample_rate_hz)
            self.get_logger().info(
                f"Opened audio writer: path={active_run.audio_path}, "
                f"sample_rate_hz={active_run.audio_sample_rate_hz}, "
                f"channels={active_run.audio_channels}, encoding={active_run.audio_encoding}"
            )
        elif (
            active_run.audio_sample_rate_hz != task.sample_rate_hz
            or active_run.audio_channels != task.channels
            or active_run.audio_encoding != task.encoding
        ):
            raise ValueError("Audio format changed during recording")

    def _write_audio_samples(self, active_run: ActiveRecordingRun, samples: array) -> None:
        active_run.audio_wav.writeframes(samples.tobytes())
        active_run.audio_chunks_written += 1
        active_run.audio_samples_written += len(samples)

    def _write_audio_silence(
        self, active_run: ActiveRecordingRun, samples_per_chunk: int, chunk_count: int
    ) -> None:
        active_run.audio_wav.writeframes(b"\x00\x00" * samples_per_chunk * chunk_count)
        active_run.audio_chunks_written += chunk_count
        active_run.audio_samples_written += samples_per_chunk * chunk_count

    def _stop_recording_run(self, stop_origin: str, queue_stop_capture: bool) -> None:
        stopped_at = wall_clock_timestamp()
        with self._run_lock:
            active_run = self._active_run
            self._active_run = None
        if active_run is None:
            return
        self.get_logger().info(
            f"Recording run {active_run.label} stopping: origin={stop_origin}, "
            f"stopped_at={stopped_at:.6f}, rgb_frames={active_run.rgb_counts}, "
            f"audio_chunks={active_run.audio_chunks_written}"
        )
        self._flush_run_media(active_run)
        with self._run_lock:
            active_run.stopped_at = stopped_at
            self._write_run_metadata(active_run, stopped_at)
        if queue_stop_capture:
            self._queue_boundary_capture(
                "pointcloud_stop",
                active_run,
                False,
                True,
                active_run.artifact_path,
            )
        else:
            self.get_logger().warning(
                f"Recording run {active_run.label} stopped during shutdown; "
                "stop pointcloud capture was not queued."
            )

    def _flush_run_media(self, active_run: ActiveRecordingRun) -> None:
        done = Event()
        self._media_queue.put(CloseRunMediaTask(active_run, done))
        done.wait()

    def _close_run_files(self, active_run: ActiveRecordingRun) -> None:
        """Finalize C-backed media handles before metadata exposes the artifact paths."""
        released_rgb = 0
        for writer in active_run.rgb_writers.values():
            if writer is not None:
                writer.release()
                released_rgb += 1
        for timestamp_file in active_run.rgb_timestamp_files.values():
            if timestamp_file is not None:
                timestamp_file.close()
        if active_run.audio_wav is not None:
            active_run.audio_wav.close()
        self.get_logger().info(
            f"Closed media handles for {active_run.label}: "
            f"rgb_writers={released_rgb}, audio_opened={active_run.audio_wav is not None}"
        )

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
            "chunks_stale_dropped": active_run.audio_chunks_stale_dropped,
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
                    "timestamps": _relative_artifact_path(
                        run_root, active_run.rgb_timestamp_paths[camera.key]
                    ),
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
        self.get_logger().info(
            f"Wrote recording metadata for {active_run.label}: "
            f"manifest={manifest_path}, audio_meta={audio_meta_path}, "
            f"pointcloud_scans={len(active_run.pointcloud_scans)}"
        )

    def _queue_boundary_capture(
        self,
        label: str,
        active_run: ActiveRecordingRun | None,
        publish_planning_cloud: bool,
        save_to_session: bool,
        artifact_root: Path | None,
    ) -> None:
        task = BoundaryCaptureTask(
            label=label,
            active_run=active_run,
            publish_planning_cloud=publish_planning_cloud,
            save_to_session=save_to_session,
            artifact_root=artifact_root,
        )
        self._capture_queue.put(task)
        run_label = active_run.label if active_run is not None else "session"
        self.get_logger().info(
            f"Queued boundary capture: label={label}, run={run_label}, "
            f"save_artifacts={save_to_session}, queue_depth={self._capture_queue.qsize()}"
        )

    def _run_boundary_capture_worker(self) -> None:
        while True:
            task = self._capture_queue.get()
            if task is None:
                return
            run_label = task.active_run.label if task.active_run is not None else "session"
            if self._capture_worker_stop.is_set():
                self.get_logger().warning(
                    f"Skipped boundary capture during shutdown: label={task.label}, run={run_label}"
                )
                continue
            try:
                self.get_logger().info(
                    f"Boundary capture started: label={task.label}, run={run_label}"
                )
                scans = self._capture_pointcloud(
                    task.label,
                    task.publish_planning_cloud,
                    task.save_to_session,
                    task.artifact_root,
                )
            except Exception as exc:
                self.get_logger().error(
                    f"RECORDING ERROR: boundary capture failed: "
                    f"label={task.label}, run={run_label}, error={exc}"
                )
                continue
            if task.active_run is None:
                self.get_logger().info(
                    f"Boundary capture completed for session: label={task.label}, scans={len(scans)}"
                )
                continue
            with self._run_lock:
                task.active_run.pointcloud_scans.extend(scans)
                stopped_at = task.active_run.stopped_at
                if stopped_at is not None:
                    self._write_run_metadata(task.active_run, stopped_at)
            self.get_logger().info(
                f"Boundary capture completed: label={task.label}, run={run_label}, "
                f"scans={len(scans)}, stopped={stopped_at is not None}"
            )

    def _capture_pointcloud(
        self,
        label: str,
        publish_planning_cloud: bool,
        save_to_session: bool,
        artifact_root: Path | None = None,
    ) -> list[PointCloudScan]:
        # Recording delegates all pointcloud work to the dedicated pointcloud action server.
        if not self._capture_client.wait_for_server(timeout_sec=CAPTURE_ACCEPT_TIMEOUT_S):
            raise TimeoutError("Timed out waiting for pointcloud capture server")
        goal = CapturePointCloud.Goal()
        goal.label = label
        goal.publish_planning_cloud = bool(publish_planning_cloud)
        goal.save_artifacts = bool(save_to_session)
        goal.artifact_root = str(
            artifact_root if artifact_root is not None else self._artifact_path
        )
        self.get_logger().info(
            f"Sending pointcloud capture goal: label={label}, "
            f"save_artifacts={save_to_session}, artifact_root={goal.artifact_root}"
        )
        goal_handle = wait_future(
            self._capture_client.send_goal_async(goal),
            CAPTURE_ACCEPT_TIMEOUT_S,
            f"Timed out waiting for pointcloud capture acceptance: {label}",
        )
        if not goal_handle.accepted:
            raise RuntimeError("Pointcloud capture goal was rejected")
        self.get_logger().info(f"Pointcloud capture accepted: label={label}; waiting for result.")
        action_result = wait_future(
            goal_handle.get_result_async(),
            CAPTURE_RESULT_TIMEOUT_S,
            f"Timed out waiting for pointcloud capture result: {label}",
        ).result
        if not action_result.success:
            raise RuntimeError(action_result.message)
        scans = [
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
        self.get_logger().info(
            f"Pointcloud capture result: label={label}, success={action_result.success}, "
            f"scans={len(scans)}, message={action_result.message}"
        )
        return scans

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
        with self._boundary_lock:
            if self._active_run is not None:
                self._stop_recording_run("shutdown_active_run", False)
        self._media_queue.put(None)
        self._media_worker.join(timeout=5.0)
        self._capture_worker_stop.set()
        self._capture_queue.put(None)
        if not self._save_artifacts_enabled:
            self._discard_experiment_folder()
        return super().destroy_node()


def wait_future(future, timeout_s: float | None = None, timeout_message: str = "Future timed out"):
    event = Event()
    future.add_done_callback(lambda _future: event.set())
    if not event.wait(timeout_s):
        raise TimeoutError(timeout_message)
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

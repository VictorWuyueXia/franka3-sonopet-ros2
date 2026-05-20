from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, RLock
from typing import Any

import rclpy
from cv_bridge import CvBridge
from fr3_sonopet_interfaces.action import RecordExperiment
from fr3_sonopet_interfaces.msg import AudioChunk
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, PointCloud2

from fr3_sonopet_recording.artifact_writers import (
    AudioWavRecorder,
    RgbVideoRecorder,
    SnapshotResult,
    capture_pointcloud_snapshot,
    write_json,
)
from fr3_sonopet_recording.recording_config import (
    CameraRecordingSpec,
    RecordingConfig,
    load_recording_config,
    recording_topics,
)


@dataclass
class RecordingSession:
    """Runtime resources owned by one active experiment recording."""

    run_id: str
    artifact_path: Path
    started_at_iso: str
    video_writers: dict[str, RgbVideoRecorder]
    rgb_subscriptions: list[Any]
    audio_writer: AudioWavRecorder
    audio_subscription: Any
    snapshots: dict[str, list[SnapshotResult]]


def _camera_dir(artifact_path: Path, camera_key: str) -> Path:
    if camera_key == "in_hand":
        return artifact_path / "camera_in_hand"
    if camera_key == "fixed":
        return artifact_path / "camera_fixed"
    raise ValueError(f"Unsupported recording camera: {camera_key}")


def _validate_recording_topics(config: RecordingConfig) -> None:
    topics = recording_topics(config)
    if not topics or any(not topic.startswith("/") for topic in topics):
        raise ValueError("Recording topics must be absolute ROS topic names")


class RecordingNode(Node):
    """Record RGB video, microphone WAV, and start/end D405 pointcloud snapshots."""

    def __init__(self) -> None:
        super().__init__("recording_node")
        self.declare_parameter("recording_config", Parameter.Type.STRING)
        self.declare_parameter("camera_config", Parameter.Type.STRING)

        self._callback_group = ReentrantCallbackGroup()
        self._bridge = CvBridge()
        self._lock = RLock()
        self._session: RecordingSession | None = None
        self._config = self._load_config()
        _validate_recording_topics(self._config)

        self._record_server = ActionServer(
            self,
            RecordExperiment,
            "/sonopet/record_experiment",
            self._execute_record,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            f"Recording interface ready: artifact_root={self._config.artifact_root}"
        )

    def _required_parameter(self, name: str) -> str:
        parameter = self.get_parameter(name)
        if parameter.type_ == Parameter.Type.NOT_SET:
            raise RuntimeError(f"Required recording parameter is not set: {name}")
        return str(parameter.value)

    def _load_config(self) -> RecordingConfig:
        return load_recording_config(
            self._required_parameter("recording_config"),
            self._required_parameter("camera_config"),
        )

    def _execute_record(self, goal_handle):
        run_id = goal_handle.request.run_id.strip()
        if not run_id:
            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        feedback = RecordExperiment.Feedback()
        feedback.phase = "recording_start" if goal_handle.request.start else "recording_stop"
        goal_handle.publish_feedback(feedback)

        try:
            if goal_handle.request.start:
                artifact_path = self._start_recording(run_id)
                message = "Recording started."
            else:
                artifact_path = self._stop_recording(run_id)
                message = "Recording stopped."
        except Exception as exc:
            goal_handle.abort()
            result = RecordExperiment.Result()
            result.success = False
            result.message = str(exc)
            result.artifact_path = ""
            return result

        goal_handle.succeed()
        result = RecordExperiment.Result()
        result.success = True
        result.message = message
        result.artifact_path = str(artifact_path)
        return result

    def _start_recording(self, run_id: str) -> Path:
        artifact_path = self._config.artifact_root / run_id
        with self._lock:
            if self._session is not None:
                raise RuntimeError(f"Recording is already active: {self._session.run_id}")

            artifact_path.mkdir(parents=True, exist_ok=False)
            video_writers, rgb_subscriptions = self._start_rgb_recorders(artifact_path)
            audio_writer, audio_subscription = self._start_audio_recorder(artifact_path)
            self._session = RecordingSession(
                run_id=run_id,
                artifact_path=artifact_path,
                started_at_iso=datetime.now(UTC).isoformat(),
                video_writers=video_writers,
                rgb_subscriptions=rgb_subscriptions,
                audio_writer=audio_writer,
                audio_subscription=audio_subscription,
                snapshots={camera.key: [] for camera in self._config.cameras},
            )

        self._capture_all_snapshots("start", "pointcloud_start.pcd")
        return artifact_path

    def _start_rgb_recorders(
        self, artifact_path: Path
    ) -> tuple[dict[str, RgbVideoRecorder], list[Any]]:
        video_writers: dict[str, RgbVideoRecorder] = {}
        subscriptions: list[Any] = []
        for camera in self._config.cameras:
            camera_dir = _camera_dir(artifact_path, camera.key)
            camera_dir.mkdir(parents=True, exist_ok=True)
            video_writers[camera.key] = RgbVideoRecorder(
                camera_dir / "rgb.avi",
                camera_dir / "rgb_timestamps.csv",
                self._config.video_fps,
            )
            subscriptions.append(
                self.create_subscription(
                    Image,
                    camera.rgb_topic,
                    lambda msg, key=camera.key: self._on_rgb(key, msg),
                    10,
                    callback_group=self._callback_group,
                )
            )
        return video_writers, subscriptions

    def _start_audio_recorder(self, artifact_path: Path) -> tuple[AudioWavRecorder, Any]:
        audio_dir = artifact_path / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        writer = AudioWavRecorder(audio_dir / "audio.wav")
        subscription = self.create_subscription(
            AudioChunk,
            self._config.audio_topic,
            self._on_audio,
            10,
            callback_group=self._callback_group,
        )
        return writer, subscription

    def _stop_recording(self, run_id: str) -> Path:
        with self._lock:
            session = self._require_session()
            if run_id and run_id != session.run_id:
                raise RuntimeError(f"Cannot stop run_id={run_id}; active run is {session.run_id}")
            artifact_path = session.artifact_path

        self._capture_all_snapshots("end", "pointcloud_end.pcd")
        with self._lock:
            session = self._require_session()
            manifest = self._build_manifest(session)
            self._close_session(session)
            self._session = None

        write_json(artifact_path / "manifest.json", manifest)
        return artifact_path

    def _on_rgb(self, camera_key: str, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                if self._session is None:
                    return
                writer = self._session.video_writers.get(camera_key)
                if writer is None:
                    return
                writer.write_frame(frame, msg.header.stamp.sec, msg.header.stamp.nanosec)
        except Exception as exc:
            self.get_logger().error(f"RGB recording failed for {camera_key}: {exc}")

    def _on_audio(self, msg: AudioChunk) -> None:
        try:
            with self._lock:
                if self._session is None:
                    return
                self._session.audio_writer.write_samples(
                    list(msg.samples),
                    int(msg.sample_rate_hz),
                    int(msg.channels),
                    str(msg.encoding),
                )
        except Exception as exc:
            self.get_logger().error(f"Audio recording failed: {exc}")

    def _capture_all_snapshots(self, label: str, filename: str) -> None:
        for camera in self._config.cameras:
            result = self._capture_snapshot(camera, label, filename)
            with self._lock:
                session = self._require_session()
                session.snapshots[camera.key].append(result)
            if not result.success:
                self.get_logger().warning(
                    f"Pointcloud snapshot failed for {camera.key}/{label}: {result.error}"
                )

    def _capture_snapshot(
        self,
        camera: CameraRecordingSpec,
        label: str,
        filename: str,
    ) -> SnapshotResult:
        with self._lock:
            session = self._require_session()
            path = _camera_dir(session.artifact_path, camera.key) / filename
        return capture_pointcloud_snapshot(
            label,
            path,
            self._config.snapshot_timeout_sec,
            lambda enabled: self._set_pointcloud_enabled(camera, enabled),
            lambda timeout_sec: self._receive_pointcloud(camera, timeout_sec),
        )

    def _set_pointcloud_enabled(self, camera: CameraRecordingSpec, enabled: bool) -> None:
        client = self.create_client(
            SetParameters,
            camera.parameter_service,
            callback_group=self._callback_group,
        )
        try:
            if not client.wait_for_service(timeout_sec=2.0):
                raise RuntimeError(f"Parameter service unavailable: {camera.parameter_service}")

            request = SetParameters.Request()
            request.parameters = [
                Parameter("pointcloud.enable", Parameter.Type.BOOL, enabled).to_parameter_msg()
            ]
            done = Event()
            future = client.call_async(request)
            future.add_done_callback(lambda _future: done.set())
            if not done.wait(timeout=3.0):
                raise RuntimeError(f"Timed out setting pointcloud.enable for {camera.key}")

            response = future.result()
            if response is None or not response.results:
                raise RuntimeError(f"No parameter response from {camera.parameter_service}")
            failures = [result.reason for result in response.results if not result.successful]
            if failures:
                raise RuntimeError("; ".join(failures))
        finally:
            self.destroy_client(client)

    def _receive_pointcloud(self, camera: CameraRecordingSpec, timeout_sec: float) -> PointCloud2:
        received = Event()
        cloud: dict[str, PointCloud2] = {}

        def _on_cloud(msg: PointCloud2) -> None:
            cloud["msg"] = msg
            received.set()

        subscription = self.create_subscription(
            PointCloud2,
            camera.pointcloud_topic,
            _on_cloud,
            1,
            callback_group=self._callback_group,
        )
        try:
            if not received.wait(timeout=timeout_sec):
                raise RuntimeError(f"Timed out waiting for {camera.pointcloud_topic}")
            return cloud["msg"]
        finally:
            self.destroy_subscription(subscription)

    def _require_session(self) -> RecordingSession:
        if self._session is None:
            raise RuntimeError("No recording is active")
        return self._session

    def _close_session(self, session: RecordingSession) -> None:
        for subscription in session.rgb_subscriptions:
            self.destroy_subscription(subscription)
        self.destroy_subscription(session.audio_subscription)

        for writer in session.video_writers.values():
            writer.close()
        session.audio_writer.close()
        write_json(
            session.artifact_path / "audio" / "audio_meta.json",
            session.audio_writer.metadata(),
        )

    def _build_manifest(self, session: RecordingSession) -> dict[str, Any]:
        cameras = {}
        for camera in self._config.cameras:
            camera_dir = _camera_dir(session.artifact_path, camera.key)
            relative_dir = camera_dir.relative_to(session.artifact_path)
            cameras[camera.key] = {
                "directory": str(relative_dir),
                "rgb_topic": camera.rgb_topic,
                "rgb_video": str(relative_dir / "rgb.avi"),
                "rgb_timestamps": str(relative_dir / "rgb_timestamps.csv"),
                "frames": session.video_writers[camera.key].frame_count,
                "snapshots": [
                    {
                        "label": snapshot.label,
                        "path": str(snapshot.path.relative_to(session.artifact_path)),
                        "success": snapshot.success,
                        "point_count": snapshot.point_count,
                        "error": snapshot.error,
                    }
                    for snapshot in session.snapshots[camera.key]
                ],
            }

        return {
            "run_id": session.run_id,
            "artifact_path": str(session.artifact_path),
            "started_at": session.started_at_iso,
            "ended_at": datetime.now(UTC).isoformat(),
            "audio": {
                "topic": self._config.audio_topic,
                "wav": "audio/audio.wav",
                "metadata": "audio/audio_meta.json",
            },
            "rgb_video": {"fps": self._config.video_fps},
            "pointcloud_snapshots": {"timeout_sec": self._config.snapshot_timeout_sec},
            "cameras": cameras,
        }

    def destroy_node(self) -> bool:
        with self._lock:
            if self._session is not None:
                self._close_session(self._session)
                self._session = None
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = RecordingNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

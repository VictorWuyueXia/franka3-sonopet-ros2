from __future__ import annotations

# Import standard Python and ROS2 libraries used in recording functionality
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

# Import project-specific helpers for artifact (file) recording and config
from fr3_sonopet_recording.artifact_writers import (
    AudioWavRecorder,
    RgbVideoRecorder,
    write_json,
    write_pointcloud_pcd,
)
from fr3_sonopet_recording.pointcloud_snapshot import (
    PointCloudSnapshotStateMachine,
    SnapshotResult,
)
from fr3_sonopet_recording.recording_config import (
    CameraRecordingSpec,
    RecordingConfig,
    load_recording_config,
    recording_topics,
)
from fr3_sonopet_recording.run_guard import ActiveRun, RecordingRunGuard
from fr3_sonopet_recording.topic_policy import require_topics

# Holds all runtime resources for one current experiment recording session
@dataclass
class RecordingSession:
    """Runtime resources owned by one active experiment recording."""

    active_run: ActiveRun
    started_at_iso: str
    video_writers: dict[str, RgbVideoRecorder]
    rgb_subscriptions: list[Any]
    audio_writer: AudioWavRecorder
    audio_subscription: Any
    snapshots: dict[str, list[SnapshotResult]]


# Main class: handles all recording (audio, video, pointclouds) to disk
class RecordingNode(Node):
    """Record practical experiment artifacts from live sensor topics."""

    def __init__(self) -> None:
        super().__init__("recording_node")
        self.declare_parameter("recording_config", "")

        # Setup nodes, thread locking, configuration and communications
        self._callback_group = ReentrantCallbackGroup()
        self._bridge = CvBridge()
        self._lock = RLock()
        self._run_guard = RecordingRunGuard()
        self._session: RecordingSession | None = None
        self._config = self._load_config()
        require_topics(recording_topics(self._config))

        # Create the ROS2 Action server to start/stop recordings
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

    # Load the recording configuration YAML file
    def _load_config(self) -> RecordingConfig:
        config_path = str(self.get_parameter("recording_config").value)
        if not config_path.strip():
            raise ValueError("recording_config parameter must point to a YAML file")
        return load_recording_config(config_path)

    # Called when a record action request is received (start/stop)
    def _execute_record(self, goal_handle):
        # Get the requested run ID, or generate a timestamp if none is provided
        run_id = goal_handle.request.run_id.strip()
        if not run_id:
            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        # Publish feedback on whether recording is starting or stopping
        feedback = RecordExperiment.Feedback()
        feedback.phase = "recording_start" if goal_handle.request.start else "recording_stop"
        goal_handle.publish_feedback(feedback)

        try:
            # Start or stop the recording process based on the request type
            artifact_path = (
                self._start_recording(run_id)
                if goal_handle.request.start
                else self._stop_recording(run_id)
            )
        except Exception as exc:
            # If there is an error, abort the operation and return failure details
            goal_handle.abort()
            result = RecordExperiment.Result()
            result.success = False
            result.message = str(exc)
            result.artifact_path = ""
            return result

        # On success, mark the goal as succeeded and provide results
        goal_handle.succeed()
        result = RecordExperiment.Result()
        result.success = True
        result.message = "Recording started." if goal_handle.request.start else "Recording stopped."
        result.artifact_path = str(artifact_path)
        return result

    # Begin recording by setting up all subscriptions and files
    def _start_recording(self, run_id: str) -> Path:
        # Create the unique artifact directory for this run
        artifact_path = self._config.artifact_root / run_id
        # Lock the recording process to ensure only one recording can run at a time
        with self._lock:
            # Check if a recording is already active
            if self._run_guard.active_run is not None:
                raise RuntimeError(
                    f"Recording is already active: {self._run_guard.active_run.run_id}"
                )
            # Create the unique artifact directory for this run and start recording
            artifact_path.mkdir(parents=True, exist_ok=False)
            active_run = self._run_guard.start(run_id, artifact_path)

            # Set up video writers and subscriptions for each camera
            video_writers: dict[str, RgbVideoRecorder] = {}
            rgb_subscriptions: list[Any] = []
            for camera in self._config.cameras:
                # Create the directory for each camera's video files
                camera_dir = artifact_path / camera.directory
                camera_dir.mkdir(parents=True, exist_ok=True)
                # Create the video writer for the camera
                writer = RgbVideoRecorder(
                    camera_dir / "rgb.avi",
                    camera_dir / "rgb_timestamps.csv",
                    self._config.video_fps,
                    self._config.video_codec,
                )
                # Store the video writer for this camera
                video_writers[camera.key] = writer
                # Subscribe to receive and save camera image data as it comes in
                rgb_subscriptions.append(
                    self.create_subscription(
                        Image,
                        camera.rgb_topic,
                        lambda msg, key=camera.key: self._on_rgb(key, msg),
                        10,  # Queue size for the subscription
                        callback_group=self._callback_group,
                    )
                )

            # Set up audio recording and subscription
            audio_dir = artifact_path / self._config.audio_directory
            audio_dir.mkdir(parents=True, exist_ok=True)
            # Create the audio writer for the audio file
            audio_writer = AudioWavRecorder(audio_dir / "audio.wav")
            # Subscribe to receive and save audio data as it comes in
            audio_subscription = self.create_subscription(
                AudioChunk,
                self._config.audio_topic,
                self._on_audio,
                10,  # Queue size for the subscription
                callback_group=self._callback_group,
            )
            # Create the recording session object
            self._session = RecordingSession(
                active_run=active_run,
                started_at_iso=datetime.now(UTC).isoformat(),
                video_writers=video_writers,
                rgb_subscriptions=rgb_subscriptions,
                audio_writer=audio_writer,
                audio_subscription=audio_subscription,
                snapshots={camera.key: [] for camera in self._config.cameras},
            )

        # Attempt to capture initial pointcloud snapshots from all cameras
        try:
            self._capture_all_snapshots(self._config.snapshots.start_label)
        except Exception:
            with self._lock:
                if self._session is not None:
                    self._close_session(self._session)
                    self._session = None
                self._run_guard.active_run = None
            raise
        return artifact_path

    # Ends current recording, saves manifest and closes writers/subscriptions
    def _stop_recording(self, run_id: str) -> Path:
        with self._lock:
            session = self._require_session()
            artifact_path = session.active_run.artifact_path

        # Capture final pointcloud snapshots before finishing
        self._capture_all_snapshots(self._config.snapshots.end_label)
        # Lock the recording process to ensure only one recording can run at a time
        with self._lock:
            session = self._require_session()
            self._run_guard.stop(run_id)
            # Build the manifest for the recording
            manifest = self._build_manifest(session)
            # Close the recording session
            self._close_session(session)
            self._session = None

        write_json(artifact_path / "manifest.json", manifest)
        return artifact_path

    # Called for every incoming camera image; writes to AVI video
    def _on_rgb(self, camera_key: str, msg: Image) -> None:
        """Convert ROS image messages into BGR frames for the per-camera AVI writer."""

        # Try to convert the ROS image message to a BGR frame and write it to the video writer
        try:
            # Convert the ROS image message to a BGR frame
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                # Check if a recording session is active
                if self._session is None:
                    return
                # Get the video writer for the camera
                writer = self._session.video_writers.get(camera_key)
                # If the video writer is not found, return
                if writer is None:
                    return
                writer.write_frame(frame, msg.header.stamp.sec, msg.header.stamp.nanosec)
        except Exception as exc:
            self.get_logger().error(f"RGB recording failed for {camera_key}: {exc}")

    # Called for every new audio segment; appends to WAV file
    def _on_audio(self, msg: AudioChunk) -> None:
        """Append project-owned PCM chunks to the active WAV artifact."""

        # Try to append the audio samples to the WAV file
        try:
            # Check if a recording session is active
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

    # Attempt to capture a pointcloud snapshot from every camera
    def _capture_all_snapshots(self, label: str) -> None:
        if not self._config.snapshots.enabled:
            return
        # Capture a pointcloud snapshot for each camera
        for camera in self._config.cameras:
            result = self._capture_snapshot(camera, label)
            with self._lock:
                session = self._require_session()
                session.snapshots[camera.key].append(result)
            if not result.success:
                message = f"Pointcloud snapshot failed for {camera.key}/{label}: {result.error}"
                if self._config.snapshots.require_success:
                    raise RuntimeError(message)
                self.get_logger().warning(message)

    # Capture a single pointcloud snapshot from a camera
    def _capture_snapshot(self, camera: CameraRecordingSpec, label: str) -> SnapshotResult:
        with self._lock:
            session = self._require_session()
            path = session.active_run.artifact_path / camera.directory / f"pointcloud_{label}.pcd"

        # Create a state machine helper for pointcloud grabbing/cleanup
        state_machine = PointCloudSnapshotStateMachine(
            set_enabled=lambda enabled: self._set_pointcloud_enabled(camera, enabled),
            receive_cloud=lambda timeout: self._receive_pointcloud(camera, timeout),
            write_cloud=write_pointcloud_pcd,
        )
        return state_machine.capture(label, path, self._config.snapshots.timeout_sec)

    # Enables or disables the camera's pointcloud generator for snapshotting
    def _set_pointcloud_enabled(self, camera: CameraRecordingSpec, enabled: bool) -> None:
        """Set the RealSense pointcloud publisher parameter around one snapshot capture."""

        # Create a client to set the pointcloud enable parameter
        client = self.create_client(
            SetParameters,
            camera.parameter_service,
            callback_group=self._callback_group,
        )
        try:
            # Wait for the parameter service to become available
            if not client.wait_for_service(timeout_sec=2.0):
                raise RuntimeError(f"Parameter service unavailable: {camera.parameter_service}")

            request = SetParameters.Request()
            request.parameters = [
                Parameter(
                    "pointcloud.enable",
                    Parameter.Type.BOOL,
                    enabled,
                ).to_parameter_msg()
            ]
            # Create an event to wait for the parameter service to be called
            done = Event()
            # Call the parameter service asynchronously
            future = client.call_async(request)
            future.add_done_callback(lambda _future: done.set())
            if not done.wait(timeout=3.0):
                raise RuntimeError(f"Timed out setting pointcloud.enable for {camera.key}")

            # Wait for the parameter service to be called
            response = future.result()
            # Check if the parameter service returned a response
            if response is None or not response.results:
                raise RuntimeError(f"No parameter response from {camera.parameter_service}")
            # Check if the parameter service returned any failures
            failures = [result.reason for result in response.results if not result.successful]
            # If there are failures, raise an error
            if failures:
                raise RuntimeError("; ".join(failures))
        finally:
            self.destroy_client(client)

    # Waits for a single pointcloud ROS message and returns it
    def _receive_pointcloud(self, camera: CameraRecordingSpec, timeout_sec: float) -> PointCloud2:
        """Wait for one PointCloud2 message after the camera snapshot publisher is enabled."""

        # Create an event to wait for the pointcloud message to be received
        received = Event()
        cloud: dict[str, PointCloud2] = {}
        # Create a callback function to set the event when a pointcloud message is received
        def _on_cloud(msg: PointCloud2) -> None:
            cloud["msg"] = msg
            received.set()

        # Create a subscription to receive pointcloud messages
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

    # Ensures there is an active session, or raises an error
    def _require_session(self) -> RecordingSession:
        if self._session is None:
            raise RuntimeError("No recording is active")
        return self._session

    # Close out files and subscriptions when ending a recording session
    def _close_session(self, session: RecordingSession) -> None:
        """Flush file writers and release ROS subscriptions owned by the active run."""

        for subscription in session.rgb_subscriptions:
            self.destroy_subscription(subscription)
        self.destroy_subscription(session.audio_subscription)

        for writer in session.video_writers.values():
            writer.close()
        session.audio_writer.close()
        # Save audio metadata after recording stops
        write_json(
            session.active_run.artifact_path
            / self._config.audio_directory
            / "audio_meta.json",
            session.audio_writer.metadata(),
        )

    # Builds a summary of all recorded files, topics, and statistics for this run
    def _build_manifest(self, session: RecordingSession) -> dict[str, Any]:
        camera_payload = {}
        # Build the payload for each camera
        for camera in self._config.cameras:
            # Get the video writer for the camera
            writer = session.video_writers[camera.key]
            # Create the payload for the camera
            camera_payload[camera.key] = {
                "directory": camera.directory,
                "rgb_topic": camera.rgb_topic,
                "rgb_video": str(Path(camera.directory) / "rgb.avi"),
                "rgb_timestamps": str(Path(camera.directory) / "rgb_timestamps.csv"),
                "frames": writer.frame_count,
                "snapshots": [
                    {
                        "label": snapshot.label,
                        "path": str(snapshot.path.relative_to(session.active_run.artifact_path)),
                        "success": snapshot.success,
                        "point_count": snapshot.point_count,
                        "error": snapshot.error,
                    }
                    for snapshot in session.snapshots[camera.key]
                ],
            }

        return {
            "run_id": session.active_run.run_id,
            "artifact_path": str(session.active_run.artifact_path),
            "started_at": session.started_at_iso,
            "ended_at": datetime.now(UTC).isoformat(),
            "audio": {
                "topic": self._config.audio_topic,
                "wav": str(Path(self._config.audio_directory) / "audio.wav"),
                "metadata": str(Path(self._config.audio_directory) / "audio_meta.json"),
            },
            "rgb_video": {
                "codec": self._config.video_codec,
                "fps": self._config.video_fps,
            },
            "pointcloud_snapshots": {
                "enabled": self._config.snapshots.enabled,
                "output_format": self._config.snapshots.output_format,
                "timeout_sec": self._config.snapshots.timeout_sec,
            },
            "cameras": camera_payload,
        }

    # Ensure resources are released if ROS node is shutdown
    def destroy_node(self) -> bool:
        with self._lock:
            if self._session is not None:
                self._close_session(self._session)
                self._session = None
                self._run_guard.active_run = None
        return super().destroy_node()


# The main entry point: starts the recording node and its ROS2 executor
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
        rclpy.shutdown()

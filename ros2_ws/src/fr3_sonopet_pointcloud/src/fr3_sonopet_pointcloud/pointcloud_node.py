from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from time import monotonic

import numpy as np
import rclpy
import yaml
from fr3_sonopet_interfaces.action import CapturePointCloud
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener

from fr3_sonopet_pointcloud.cloud_processing import (
    BASE_FRAME,
    artifact_root,
    camera_dir,
    experiment_run_id,
    local_timestamp_label,
    make_colored_cloud,
    matrix_from_transform,
    next_existing_name,
    next_indexed_name,
    pointcloud_xyz_rgb,
    transform_points,
    trim_sensor_cloud,
    wall_clock_timestamp,
    write_colored_pcd,
    write_pointcloud_metadata,
)

CAPTURE_ACTION = "/realsense/capture_pointcloud"
CAPTURED_PLANNING_CLOUD_TOPIC = "/sonopet/captured_planning_cloud"
PLANNING_CLOUD_TOPIC = "/sonopet/planning_cloud"
VENDOR_STALE_TOPIC = "/sonopet/vendor_joint_state_stale"
VENDOR_LIVE_MAX_AGE_S = 0.5
STALE_MESSAGE = "Cannot capture pointcloud: cannot get joint state"


@dataclass(frozen=True)
class CameraSpec:
    key: str
    pointcloud_topic: str
    parameter_service: str


@dataclass(frozen=True)
class PointcloudConfig:
    snapshot_timeout_sec: float
    trim_distance_m: float
    trim_farthest_fraction: float
    cameras: tuple[CameraSpec, ...]


class PointcloudNode(Node):
    """Capture D405 clouds as trimmed colored fr3_link0 planning artifacts."""

    def __init__(self) -> None:
        super().__init__("pointcloud_node")
        self.declare_parameter("recording_config", Parameter.Type.STRING)
        self.declare_parameter("camera_config", Parameter.Type.STRING)
        self.declare_parameter("pointcloud_config", Parameter.Type.STRING)
        self._callback_group = ReentrantCallbackGroup()
        self._config = load_pointcloud_config(
            Path(str(self.get_parameter("recording_config").value)),
            Path(str(self.get_parameter("camera_config").value)),
            Path(str(self.get_parameter("pointcloud_config").value)),
        )
        self._vendor_live_received_s: float | None = None
        self._vendor_is_stale = True
        self._session_root = artifact_root() / experiment_run_id()
        self._captures_by_root: dict[Path, list[dict]] = {}
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._captured_cloud_pub = self.create_publisher(
            PointCloud2,
            CAPTURED_PLANNING_CLOUD_TOPIC,
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )
        self._planning_cloud_pub = self.create_publisher(
            PointCloud2,
            PLANNING_CLOUD_TOPIC,
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )
        self._stale_sub = self.create_subscription(
            Bool,
            VENDOR_STALE_TOPIC,
            self._on_vendor_stale,
            10,
            callback_group=self._callback_group,
        )
        self._capture_server = ActionServer(
            self,
            CapturePointCloud,
            CAPTURE_ACTION,
            self._execute_capture,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            f"Pointcloud node ready: frame={BASE_FRAME}, action={CAPTURE_ACTION}, "
            f"trim_distance_m={self._config.trim_distance_m}, "
            f"trim_farthest_fraction={self._config.trim_farthest_fraction}"
        )

    def _on_vendor_stale(self, msg: Bool) -> None:
        # Capture is permitted only after the motion node reports a live vendor stream.
        self._vendor_is_stale = bool(msg.data)
        if not self._vendor_is_stale:
            self._vendor_live_received_s = monotonic()

    def _execute_capture(self, goal_handle):
        result = CapturePointCloud.Result()
        vendor_live_age_s = (
            None
            if self._vendor_live_received_s is None
            else monotonic() - self._vendor_live_received_s
        )
        if (
            self._vendor_is_stale
            or vendor_live_age_s is None
            or vendor_live_age_s > VENDOR_LIVE_MAX_AGE_S
        ):
            goal_handle.abort()
            result.success = False
            result.message = STALE_MESSAGE
            return result

        label = goal_handle.request.label.strip()
        if not label:
            label = f"rescan_{wall_clock_timestamp():.6f}"
        root = (
            Path(goal_handle.request.artifact_root)
            if goal_handle.request.artifact_root
            else self._session_root
        )
        root.mkdir(parents=True, exist_ok=True)
        camera_keys: list[str] = []
        artifact_paths: list[str] = []
        point_counts: list[int] = []
        timestamps: list[float] = []
        timestamp_locals: list[str] = []
        failures: list[str] = []

        for camera in self._config.cameras:
            feedback = CapturePointCloud.Feedback()
            feedback.phase = f"{camera.key}_{label}"
            goal_handle.publish_feedback(feedback)
            try:
                cloud_msg = self._capture_camera_cloud(camera)
                xyz, rgb = pointcloud_xyz_rgb(cloud_msg)
                trimmed_xyz, trimmed_rgb = trim_sensor_cloud(
                    xyz,
                    rgb,
                    self._config.trim_distance_m,
                    self._config.trim_farthest_fraction,
                )
                base_from_cloud = self._base_from_cloud_matrix(cloud_msg.header.frame_id)
                base_xyz = transform_points(trimmed_xyz, base_from_cloud)
                processed_cloud = make_colored_cloud(base_xyz, trimmed_rgb, cloud_msg.header.stamp)
                captured_at = wall_clock_timestamp()
                timestamp_local = local_timestamp_label(captured_at)
                artifact_path = ""

                if goal_handle.request.save_artifacts:
                    pcd_base = camera_dir(root, camera.key) / f"{label}.pcd"
                    pcd_path = (
                        next_indexed_name(pcd_base)
                        if label in {"pointcloud_start", "pointcloud_stop"}
                        else next_existing_name(pcd_base)
                    )
                    point_count = write_colored_pcd(pcd_path, base_xyz, trimmed_rgb)
                    artifact_path = str(pcd_path)
                    self._append_metadata(
                        root,
                        camera.key,
                        label,
                        pcd_path,
                        point_count,
                        captured_at,
                        timestamp_local,
                    )
                else:
                    point_count = int(base_xyz.shape[0])
                self.get_logger().info(
                    f"Pointcloud capture artifact: label={label}, camera={camera.key}, "
                    f"saved={bool(artifact_path)}, path={artifact_path}, points={point_count}"
                )

                if goal_handle.request.publish_planning_cloud and camera.key == "in_hand":
                    self._captured_cloud_pub.publish(processed_cloud)
                    self._planning_cloud_pub.publish(processed_cloud)

                camera_keys.append(camera.key)
                artifact_paths.append(artifact_path)
                point_counts.append(point_count)
                timestamps.append(captured_at)
                timestamp_locals.append(timestamp_local)
            except Exception as exc:
                failure = f"{camera.key}: {exc}"
                failures.append(failure)
                self.get_logger().error(
                    f"POINTCLOUD ERROR: camera capture failed: label={label}, {failure}"
                )

        if not camera_keys:
            goal_handle.abort()
            result.success = False
            result.message = "; ".join(failures)
            return result

        goal_handle.succeed()
        result.success = True
        result.message = f"Captured {len(camera_keys)} pointcloud scans."
        if failures:
            result.message = f"{result.message} Failed cameras: {'; '.join(failures)}"
        result.camera_keys = camera_keys
        result.artifact_paths = artifact_paths
        result.point_counts = point_counts
        result.timestamps = timestamps
        result.timestamp_locals = timestamp_locals
        return result

    def _capture_camera_cloud(self, camera: CameraSpec) -> PointCloud2:
        # RealSense pointcloud streaming is enabled only around the requested capture.
        client = self.create_client(
            SetParameters,
            camera.parameter_service,
            callback_group=self._callback_group,
        )
        subscription = None
        enabled = False
        try:
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
            enabled = True

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
            return cloud["msg"]
        finally:
            if subscription is not None:
                self.destroy_subscription(subscription)
            if enabled:
                request = SetParameters.Request()
                request.parameters = [
                    Parameter("pointcloud.enable", Parameter.Type.BOOL, False).to_parameter_msg()
                ]
                response = wait_future(client.call_async(request))
                failures = [result.reason for result in response.results if not result.successful]
                if failures:
                    self.get_logger().error("; ".join(failures))
            self.destroy_client(client)

    def _base_from_cloud_matrix(self, cloud_frame: str) -> np.ndarray:
        if cloud_frame == BASE_FRAME:
            return np.eye(4, dtype=np.float64)
        transform = self._tf_buffer.lookup_transform(BASE_FRAME, cloud_frame, Time())
        return matrix_from_transform(transform)

    def _append_metadata(
        self,
        root: Path,
        camera_key: str,
        label: str,
        pcd_path: Path,
        point_count: int,
        timestamp: float,
        timestamp_local: str,
    ) -> None:
        # Metadata is rewritten after each saved PCD so it always describes on-disk artifacts.
        captures = self._captures_by_root.setdefault(root, [])
        captures.append(
            {
                "camera": camera_key,
                "label": label,
                "path": str(pcd_path.relative_to(root)),
                "point_count": int(point_count),
                "timestamp": float(timestamp),
                "timestamp_local": timestamp_local,
            }
        )
        write_pointcloud_metadata(
            root / "pointcloud_meta.json",
            self._config.trim_distance_m,
            self._config.trim_farthest_fraction,
            captures,
        )


def load_pointcloud_config(
    recording_path: Path,
    camera_path: Path,
    pointcloud_path: Path,
) -> PointcloudConfig:
    """Load the pointcloud capture topics and experiment trim policy."""
    recording = yaml.safe_load(recording_path.read_text(encoding="utf-8"))["recording"]
    camera_payload = yaml.safe_load(camera_path.read_text(encoding="utf-8"))
    pointcloud = yaml.safe_load(pointcloud_path.read_text(encoding="utf-8"))["pointcloud"]
    cameras = recording["cameras"]
    return PointcloudConfig(
        snapshot_timeout_sec=float(camera_payload["pointcloud_snapshots"]["timeout_sec"]),
        trim_distance_m=float(pointcloud["trim_distance_m"]),
        trim_farthest_fraction=float(pointcloud["trim_farthest_fraction"]),
        cameras=(
            CameraSpec(
                key="in_hand",
                pointcloud_topic=str(cameras["in_hand"]["pointcloud_topic"]),
                parameter_service=str(cameras["in_hand"]["parameter_service"]),
            ),
            CameraSpec(
                key="fixed",
                pointcloud_topic=str(cameras["fixed"]["pointcloud_topic"]),
                parameter_service=str(cameras["fixed"]["parameter_service"]),
            ),
        ),
    )


def wait_future(future):
    event = Event()
    future.add_done_callback(lambda _future: event.set())
    event.wait()
    return future.result()


def main() -> None:
    rclpy.init()
    node = PointcloudNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    executor.remove_node(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

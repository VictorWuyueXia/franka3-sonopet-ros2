from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rclpy
import yaml
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo

CAMERA_NAMESPACE = "RealSense_D405"
CAMERA_ROLES = ("in_hand", "fixed")
CAMERA_STREAMS = ("color", "depth")
SOURCE_CAMERA_INFO_TOPICS = {
    "color": f"/{CAMERA_NAMESPACE}/in_hand/color/camera_info",
    "depth": f"/{CAMERA_NAMESPACE}/in_hand/depth/camera_info",
}
OUTPUT_CAMERA_INFO_TOPIC = "/sonopet/d405_intrinsics/{role}/{stream}/camera_info"
FRAME_ID = {
    ("in_hand", "color"): "in_hand_color_optical_frame",
    ("in_hand", "depth"): "in_hand_depth_optical_frame",
    ("fixed", "color"): "fixed_color_optical_frame",
    ("fixed", "depth"): "fixed_depth_optical_frame",
}
INTRINSICS_PATH = Path(__file__).resolve().parents[2] / "intrinsics" / "d405_intrinsics.yaml"


@dataclass(frozen=True)
class CameraIntrinsic:
    """CameraInfo intrinsic fields persisted independently from the driver."""

    width: int
    height: int
    distortion_model: str
    d: tuple[float, ...]
    k: tuple[float, ...]
    r: tuple[float, ...]
    p: tuple[float, ...]


def _tuple_of_floats(
    values: list[float],
    expected_length: int,
    field_name: str,
) -> tuple[float, ...]:
    if len(values) != expected_length:
        raise ValueError(f"{field_name} must contain {expected_length} values")
    return tuple(float(value) for value in values)


def intrinsic_from_camera_info(msg: CameraInfo) -> CameraIntrinsic:
    """Extract only the stable calibration fields from a CameraInfo message."""

    return CameraIntrinsic(
        width=int(msg.width),
        height=int(msg.height),
        distortion_model=str(msg.distortion_model),
        d=tuple(float(value) for value in msg.d),
        k=tuple(float(value) for value in msg.k),
        r=tuple(float(value) for value in msg.r),
        p=tuple(float(value) for value in msg.p),
    )


def _intrinsic_to_payload(intrinsic: CameraIntrinsic) -> dict[str, object]:
    return {
        "width": intrinsic.width,
        "height": intrinsic.height,
        "distortion_model": intrinsic.distortion_model,
        "d": list(intrinsic.d),
        "k": list(intrinsic.k),
        "r": list(intrinsic.r),
        "p": list(intrinsic.p),
    }


def _intrinsic_from_payload(payload: dict[str, object]) -> CameraIntrinsic:
    return CameraIntrinsic(
        width=int(payload["width"]),
        height=int(payload["height"]),
        distortion_model=str(payload["distortion_model"]),
        d=tuple(float(value) for value in payload["d"]),
        k=_tuple_of_floats(list(payload["k"]), 9, "k"),
        r=_tuple_of_floats(list(payload["r"]), 9, "r"),
        p=_tuple_of_floats(list(payload["p"]), 12, "p"),
    )


def save_intrinsics(
    color_info: CameraInfo,
    depth_info: CameraInfo,
    path: Path = INTRINSICS_PATH,
) -> None:
    """Save in-hand D405 color and depth calibration as the project reference."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "streams": {
            "color": _intrinsic_to_payload(intrinsic_from_camera_info(color_info)),
            "depth": _intrinsic_to_payload(intrinsic_from_camera_info(depth_info)),
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_intrinsics(path: Path = INTRINSICS_PATH) -> dict[str, CameraIntrinsic]:
    """Load the project-owned D405 intrinsic reference file."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    streams = payload["streams"]
    return {
        "color": _intrinsic_from_payload(streams["color"]),
        "depth": _intrinsic_from_payload(streams["depth"]),
    }


def camera_info_from_intrinsic(
    intrinsic: CameraIntrinsic,
    frame_id: str,
    stamp: Time | None,
) -> CameraInfo:
    """Reconstruct CameraInfo for one camera role while preserving fixed intrinsics."""

    msg = CameraInfo()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = intrinsic.width
    msg.height = intrinsic.height
    msg.distortion_model = intrinsic.distortion_model
    msg.d = list(intrinsic.d)
    msg.k = list(intrinsic.k)
    msg.r = list(intrinsic.r)
    msg.p = list(intrinsic.p)
    return msg


class D405IntrinsicsCaptureNode(Node):
    """Capture one in-hand D405 color/depth CameraInfo pair to the dedicated file."""

    def __init__(self) -> None:
        super().__init__("d405_intrinsics_capture")
        self._messages: dict[str, CameraInfo] = {}
        self.done = False
        for stream, topic in SOURCE_CAMERA_INFO_TOPICS.items():
            self.create_subscription(
                CameraInfo,
                topic,
                lambda msg, stream=stream: self._on_camera_info(stream, msg),
                10,
            )
        self.get_logger().info(f"Waiting for D405 intrinsics on {SOURCE_CAMERA_INFO_TOPICS}")

    def _on_camera_info(self, stream: str, msg: CameraInfo) -> None:
        self._messages[stream] = msg
        if all(stream in self._messages for stream in CAMERA_STREAMS):
            save_intrinsics(self._messages["color"], self._messages["depth"])
            self.done = True
            self.get_logger().info(f"Saved D405 intrinsics to {INTRINSICS_PATH}")


class D405IntrinsicsPublisherNode(Node):
    """Publish fixed project CameraInfo topics from the saved D405 intrinsic file."""

    def __init__(self) -> None:
        super().__init__("d405_intrinsics_publisher")
        self._intrinsics: dict[str, CameraIntrinsic] | None = None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._camera_info_publishers = {
            (role, stream): self.create_publisher(
                CameraInfo,
                OUTPUT_CAMERA_INFO_TOPIC.format(role=role, stream=stream),
                qos,
            )
            for role in CAMERA_ROLES
            for stream in CAMERA_STREAMS
        }
        self._timer = self.create_timer(1.0, self._publish_intrinsics)
        self._publish_intrinsics()

    def _try_load_intrinsics(self) -> bool:
        if self._intrinsics is not None:
            return True
        if not INTRINSICS_PATH.exists():
            self.get_logger().warning(
                f"D405 intrinsics file is not available yet: {INTRINSICS_PATH}"
            )
            return False
        self._intrinsics = load_intrinsics()
        self.get_logger().info(f"Loaded D405 intrinsics from {INTRINSICS_PATH}")
        return True

    def _publish_intrinsics(self) -> None:
        if not self._try_load_intrinsics():
            return
        intrinsics = self._intrinsics
        if intrinsics is None:
            return
        stamp = self.get_clock().now().to_msg()
        for (role, stream), publisher in self._camera_info_publishers.items():
            publisher.publish(
                camera_info_from_intrinsic(
                    intrinsics[stream],
                    FRAME_ID[(role, stream)],
                    stamp,
                )
            )


def capture_main() -> None:
    rclpy.init()
    node = D405IntrinsicsCaptureNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def publisher_main() -> None:
    rclpy.init()
    node = D405IntrinsicsPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

BASE_FRAME = "fr3_link0"


def wall_clock_timestamp() -> float:
    """Return local epoch seconds for artifact metadata."""
    return time.time()


def local_timestamp_label(epoch_seconds: float) -> str:
    """Return compact local wall-clock time aligned with recording metadata."""
    return datetime.fromtimestamp(epoch_seconds).astimezone().strftime("%Y%m%d%H%M")


def experiment_run_id() -> str:
    """Return a local session id for operator-triggered pointcloud captures."""
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")


def artifact_root() -> Path:
    """Return the frozen experiment artifact directory under the ROS workspace."""
    return Path(os.environ["FR3_SONOPET_REPO"]) / "ros2_ws" / "artifacts" / "experiments"


def camera_dir(root: Path, camera_key: str) -> Path:
    """Return the artifact camera folder matching the recording convention."""
    if camera_key == "in_hand":
        return root / "camera_in_hand"
    if camera_key == "fixed":
        return root / "camera_fixed"
    raise ValueError(f"Unsupported pointcloud camera: {camera_key}")


def next_existing_name(path: Path) -> Path:
    """Return path or the next duplicate suffix using the project artifact convention."""
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def next_indexed_name(path: Path) -> Path:
    """Return the first zero-indexed run artifact path for a repeated capture label."""
    index = 0
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def pointcloud_xyz_rgb(cloud_msg: PointCloud2) -> tuple[np.ndarray, np.ndarray]:
    """Read finite XYZ and packed RGB values from a colored PointCloud2."""
    if not cloud_msg.header.frame_id:
        raise RuntimeError("Pointcloud does not carry a frame_id")
    if not any(field.name == "rgb" for field in cloud_msg.fields):
        raise RuntimeError("Pointcloud does not carry an rgb field")
    raw_points = point_cloud2.read_points(
        cloud_msg,
        field_names=("x", "y", "z", "rgb"),
        skip_nans=True,
    )
    raw_array = np.asarray(raw_points)
    if raw_array.dtype.names:
        xyz = np.column_stack(
            (raw_array["x"], raw_array["y"], raw_array["z"])
        ).astype(np.float64)
        rgb = np.asarray(raw_array["rgb"], dtype=np.float32)
        return xyz, rgb
    if raw_array.ndim == 0:
        raw_array = np.asarray(list(raw_points), dtype=np.float64)
    matrix = np.asarray(raw_array, dtype=np.float64).reshape((-1, 4))
    return matrix[:, :3], matrix[:, 3].astype(np.float32)


def trim_sensor_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray,
    trim_distance_m: float,
    trim_farthest_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim the raw sensor cloud by local distance and farthest retained percentage."""
    if trim_distance_m <= 0.0:
        raise ValueError("trim_distance_m must be positive")
    if not 0.0 <= trim_farthest_fraction < 1.0:
        raise ValueError("trim_farthest_fraction must be in [0.0, 1.0)")
    radius_sq = np.einsum("ij,ij->i", xyz, xyz)
    inside = radius_sq <= trim_distance_m * trim_distance_m
    inside_indices = np.flatnonzero(inside)
    if inside_indices.size == 0:
        raise ValueError("Pointcloud has no points inside trim distance")
    keep_count = int(round(inside_indices.size * (1.0 - trim_farthest_fraction)))
    if keep_count <= 0:
        raise ValueError("Pointcloud trim removed all points")
    nearest = np.argpartition(radius_sq[inside], keep_count - 1)[:keep_count]
    keep_indices = inside_indices[nearest]
    return xyz[keep_indices], rgb[keep_indices]


def matrix_from_transform(transform_msg) -> np.ndarray:
    """Convert a ROS transform message into a homogeneous transform matrix."""
    translation = transform_msg.transform.translation
    rotation = transform_msg.transform.rotation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix_from_quaternion(
        np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64)
    )
    matrix[:3, 3] = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
    return matrix


def rotation_matrix_from_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Return a rotation matrix from a normalized or non-normalized XYZW quaternion."""
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply one homogeneous transform to an N x 3 point matrix."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def make_colored_cloud(points: np.ndarray, rgb: np.ndarray, stamp) -> PointCloud2:
    """Create the colored fr3_link0 PointCloud2 used by RViz and raster planning."""
    header = Header()
    header.stamp = stamp
    header.frame_id = BASE_FRAME
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    records = np.column_stack(
        (np.asarray(points, dtype=np.float32).reshape((-1, 3)), rgb.astype(np.float32))
    )
    return point_cloud2.create_cloud(header, fields, records.tolist())


def write_colored_pcd(path: Path, points: np.ndarray, rgb: np.ndarray) -> int:
    """Write the processed fr3_link0 colored cloud to an ASCII PCD file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# .PCD v0.7 - Point Cloud Data file format\n")
        stream.write("VERSION 0.7\n")
        stream.write("FIELDS x y z rgb\n")
        stream.write("SIZE 4 4 4 4\n")
        stream.write("TYPE F F F F\n")
        stream.write("COUNT 1 1 1 1\n")
        stream.write(f"WIDTH {points.shape[0]}\n")
        stream.write("HEIGHT 1\n")
        stream.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        stream.write(f"POINTS {points.shape[0]}\n")
        stream.write("DATA ascii\n")
        for point, color in zip(points, rgb, strict=True):
            stream.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {color:.9g}\n")
    return int(points.shape[0])


def write_pointcloud_metadata(
    path: Path,
    trim_distance_m: float,
    trim_farthest_fraction: float,
    captures: list[dict],
) -> None:
    """Write the session pointcloud manifest with config and capture timestamps."""
    payload = {
        "frame_id": BASE_FRAME,
        "trim_distance_m": trim_distance_m,
        "trim_farthest_fraction": trim_farthest_fraction,
        "captures": captures,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


@dataclass(frozen=True)
class TransformSpec:
    """Rigid transform expressed as translation plus XYZW quaternion."""

    parent_frame: str
    child_frame: str
    translation_xyz: Vec3
    quaternion_xyzw: Quat

    def as_matrix4x4(self) -> np.ndarray:
        qx, qy, qz, qw = self.quaternion_xyzw
        tx, ty, tz = self.translation_xyz
        rot = quaternion_xyzw_to_rotation_matrix(qx, qy, qz, qw)
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = rot
        mat[:3, 3] = [tx, ty, tz]
        return mat

    def to_dict(self) -> dict:
        return {
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "translation_xyz": [
                float(self.translation_xyz[0]),
                float(self.translation_xyz[1]),
                float(self.translation_xyz[2]),
            ],
            "quaternion_xyzw": [
                float(self.quaternion_xyzw[0]),
                float(self.quaternion_xyzw[1]),
                float(self.quaternion_xyzw[2]),
                float(self.quaternion_xyzw[3]),
            ],
        }


def quaternion_xyzw_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert XYZW quaternion into a 3x3 rotation matrix."""

    q = np.asarray([qx, qy, qz, qw], dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be > 0")
    q /= norm
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def build_transform_spec(
    parent_frame: str,
    child_frame: str,
    translation_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> TransformSpec:
    return TransformSpec(
        parent_frame=str(parent_frame),
        child_frame=str(child_frame),
        translation_xyz=(
            float(translation_xyz[0]),
            float(translation_xyz[1]),
            float(translation_xyz[2]),
        ),
        quaternion_xyzw=(
            float(quaternion_xyzw[0]),
            float(quaternion_xyzw[1]),
            float(quaternion_xyzw[2]),
            float(quaternion_xyzw[3]),
        ),
    )


def load_transform_spec(transform_path: Path) -> TransformSpec:
    path = Path(transform_path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Extrinsics file is empty: {path}")
    payload = json.loads(raw)
    return build_transform_spec(
        parent_frame=payload["parent_frame"],
        child_frame=payload["child_frame"],
        translation_xyz=payload["translation_xyz"],
        quaternion_xyzw=payload["quaternion_xyzw"],
    )


def try_load_transform_spec(transform_path: Path) -> tuple[TransformSpec | None, str | None]:
    """
    Load extrinsics JSON for capture-time stitching. Missing or empty file returns (None, reason)
    instead of raising, so one camera can still produce a partial merge.
    """

    path = Path(transform_path)
    if not path.is_file():
        return None, f"Missing extrinsics file: {path}"
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None, "Extrinsics file is empty; add eye-in-hand calibration JSON"
    payload = json.loads(raw)
    spec = build_transform_spec(
        parent_frame=payload["parent_frame"],
        child_frame=payload["child_frame"],
        translation_xyz=payload["translation_xyz"],
        quaternion_xyzw=payload["quaternion_xyzw"],
    )
    return spec, None


def compose_transform_specs(parent_to_mid: TransformSpec, mid_to_child: TransformSpec) -> TransformSpec:
    composed = np.matmul(parent_to_mid.as_matrix4x4(), mid_to_child.as_matrix4x4())
    translation_xyz = (
        float(composed[0, 3]),
        float(composed[1, 3]),
        float(composed[2, 3]),
    )
    quaternion_xyzw = rotation_matrix_to_quaternion_xyzw(composed[:3, :3])
    return TransformSpec(
        parent_frame=parent_to_mid.parent_frame,
        child_frame=mid_to_child.child_frame,
        translation_xyz=translation_xyz,
        quaternion_xyzw=quaternion_xyzw,
    )


def offset_transform_spec_translation(transform_spec: TransformSpec, offset_xyz: Sequence[float]) -> TransformSpec:
    return TransformSpec(
        parent_frame=str(transform_spec.parent_frame),
        child_frame=str(transform_spec.child_frame),
        translation_xyz=(
            float(transform_spec.translation_xyz[0] + float(offset_xyz[0])),
            float(transform_spec.translation_xyz[1] + float(offset_xyz[1])),
            float(transform_spec.translation_xyz[2] + float(offset_xyz[2])),
        ),
        quaternion_xyzw=(
            float(transform_spec.quaternion_xyzw[0]),
            float(transform_spec.quaternion_xyzw[1]),
            float(transform_spec.quaternion_xyzw[2]),
            float(transform_spec.quaternion_xyzw[3]),
        ),
    )


def build_z_yaw_quaternion_xyzw(yaw_deg: float) -> Quat:
    yaw_rad = float(np.deg2rad(float(yaw_deg)))
    half_yaw = float(yaw_rad * 0.5)
    return (0.0, 0.0, float(np.sin(half_yaw)), float(np.cos(half_yaw)))


def apply_parent_frame_trim(
    transform_spec: TransformSpec,
    offset_xyz: Sequence[float],
    yaw_deg: float,
) -> TransformSpec:
    trim_spec = TransformSpec(
        parent_frame=str(transform_spec.parent_frame),
        child_frame=f"{transform_spec.parent_frame}__trim",
        translation_xyz=(
            float(offset_xyz[0]),
            float(offset_xyz[1]),
            float(offset_xyz[2]),
        ),
        quaternion_xyzw=build_z_yaw_quaternion_xyzw(float(yaw_deg)),
    )
    return compose_transform_specs(trim_spec, transform_spec)


def invert_transform_spec(transform_spec: TransformSpec) -> TransformSpec:
    matrix = transform_spec.as_matrix4x4()
    inverse = np.linalg.inv(matrix)
    translation_xyz = (
        float(inverse[0, 3]),
        float(inverse[1, 3]),
        float(inverse[2, 3]),
    )
    quaternion_xyzw = rotation_matrix_to_quaternion_xyzw(inverse[:3, :3])
    return TransformSpec(
        parent_frame=str(transform_spec.child_frame),
        child_frame=str(transform_spec.parent_frame),
        translation_xyz=translation_xyz,
        quaternion_xyzw=quaternion_xyzw,
    )


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> Quat:
    """Convert a 3x3 rotation matrix into an XYZW quaternion."""

    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = float(np.sqrt(trace + 1.0) * 2.0)
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
        return (qx, qy, qz, qw)
    if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = float(np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0)
        return (
            0.25 * s,
            (matrix[0, 1] + matrix[1, 0]) / s,
            (matrix[0, 2] + matrix[2, 0]) / s,
            (matrix[2, 1] - matrix[1, 2]) / s,
        )
    if matrix[1, 1] > matrix[2, 2]:
        s = float(np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0)
        return (
            (matrix[0, 1] + matrix[1, 0]) / s,
            0.25 * s,
            (matrix[1, 2] + matrix[2, 1]) / s,
            (matrix[0, 2] - matrix[2, 0]) / s,
        )
    s = float(np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0)
    return (
        (matrix[0, 2] + matrix[2, 0]) / s,
        (matrix[1, 2] + matrix[2, 1]) / s,
        0.25 * s,
        (matrix[1, 0] - matrix[0, 1]) / s,
    )


def transform_pointcloud(pointcloud, transform_spec: TransformSpec):
    """Copy and transform an Open3D point cloud."""

    import open3d as o3d

    transformed = o3d.geometry.PointCloud(pointcloud)
    transformed.transform(transform_spec.as_matrix4x4())
    return transformed


def merge_pointclouds(pointclouds: Iterable):
    """Merge multiple Open3D point clouds into one cloud."""

    import open3d as o3d

    pointcloud_list = list(pointclouds)
    if not pointcloud_list:
        raise ValueError("At least one point cloud is required for merge")
    merged = o3d.geometry.PointCloud(pointcloud_list[0])
    for cloud in pointcloud_list[1:]:
        merged += cloud
    return merged

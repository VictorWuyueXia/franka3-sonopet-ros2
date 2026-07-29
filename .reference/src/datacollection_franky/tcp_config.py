"""Shared Sonopet TCP trim offsets for simulation and datacollection.

The full Sonopet tool geometry and nominal TCP pose live in `simulation/fr3.urdf`.
This module only applies small XYZ/RPY trims on top of that nominal URDF pose so
physical-world feedback can correct rough hand measurements without changing the
source URDF every time.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Tuple
from xml.etree import ElementTree as ET


_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
_source_urdf_path = os.path.join(_repo_root, "simulation", "fr3.urdf")

# User-facing trim offsets for small measurement corrections.
# These are applied after the nominal `sonopet_tcp` frame. That nominal frame is
# already aligned with `fr3_link8`, because `sonopet_tcp_joint` cancels the
# intermediate EE/tool rotations in the URDF chain.
# Keep the default at zero so the pipeline uses the nominal TCP pose from
# `simulation/fr3.urdf`, which matches the physical and visualized setup.
SONOPET_TCP_X_OFFSET_M = 0.005
SONOPET_TCP_Y_OFFSET_M = 0.010
SONOPET_TCP_Z_OFFSET_M = 0.026
SONOPET_TCP_ROLL_OFFSET_DEG = 0.0
SONOPET_TCP_PITCH_OFFSET_DEG = 0.0
SONOPET_TCP_YAW_OFFSET_DEG = 0.0


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


@dataclass(frozen=True)
class SonopetTcpGeometry:
    nominal_urdf_joint_xyz_m: Vec3
    nominal_urdf_joint_rpy_rad: Vec3
    nominal_link8_to_tcp_xyz_m: Vec3
    nominal_link8_to_tcp_rpy_rad: Vec3
    nominal_link8_to_tcp_quat_xyzw: Quat
    tcp_offset_xyz_m: Vec3
    tcp_offset_rpy_rad: Vec3
    tcp_offset_quat_xyzw: Quat
    trimmed_link8_to_tcp_xyz_m: Vec3
    trimmed_link8_to_tcp_rpy_rad: Vec3
    trimmed_link8_to_tcp_quat_xyzw: Quat


def _quat_from_rpy(roll_rad: float, pitch_rad: float, yaw_rad: float) -> Quat:
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _rot_matrix_from_rpy(roll_rad: float, pitch_rad: float, yaw_rad: float):
    cr = math.cos(roll_rad)
    sr = math.sin(roll_rad)
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)
    return (
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
        ),
        (
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
        ),
        (
            -sp,
            cp * sr,
            cp * cr,
        ),
    )


def _matmul3(a, b):
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


def _matvec3(mat3, vec3: Vec3) -> Vec3:
    return (
        float(mat3[0][0] * vec3[0] + mat3[0][1] * vec3[1] + mat3[0][2] * vec3[2]),
        float(mat3[1][0] * vec3[0] + mat3[1][1] * vec3[1] + mat3[1][2] * vec3[2]),
        float(mat3[2][0] * vec3[0] + mat3[2][1] * vec3[1] + mat3[2][2] * vec3[2]),
    )


def _rpy_from_rot_matrix(rot):
    pitch_rad = math.atan2(-rot[2][0], math.sqrt(rot[0][0] * rot[0][0] + rot[1][0] * rot[1][0]))
    cos_pitch = math.cos(pitch_rad)
    if abs(cos_pitch) > 1e-9:
        roll_rad = math.atan2(rot[2][1], rot[2][2])
        yaw_rad = math.atan2(rot[1][0], rot[0][0])
        return (float(roll_rad), float(pitch_rad), float(yaw_rad))
    roll_rad = math.atan2(-rot[1][2], rot[1][1])
    yaw_rad = 0.0
    return (float(roll_rad), float(pitch_rad), float(yaw_rad))


def _compose_parent_to_child_transform(
    parent_to_mid_xyz_m: Vec3,
    parent_to_mid_rpy_rad: Vec3,
    mid_to_child_xyz_m: Vec3,
    mid_to_child_rpy_rad: Vec3,
) -> Tuple[Vec3, Vec3]:
    parent_to_mid_rot = _rot_matrix_from_rpy(
        parent_to_mid_rpy_rad[0],
        parent_to_mid_rpy_rad[1],
        parent_to_mid_rpy_rad[2],
    )
    mid_to_child_rot = _rot_matrix_from_rpy(
        mid_to_child_rpy_rad[0],
        mid_to_child_rpy_rad[1],
        mid_to_child_rpy_rad[2],
    )
    parent_to_child_xyz_m = (
        float(parent_to_mid_xyz_m[0] + _matvec3(parent_to_mid_rot, mid_to_child_xyz_m)[0]),
        float(parent_to_mid_xyz_m[1] + _matvec3(parent_to_mid_rot, mid_to_child_xyz_m)[1]),
        float(parent_to_mid_xyz_m[2] + _matvec3(parent_to_mid_rot, mid_to_child_xyz_m)[2]),
    )
    parent_to_child_rot = _matmul3(parent_to_mid_rot, mid_to_child_rot)
    return parent_to_child_xyz_m, _rpy_from_rot_matrix(parent_to_child_rot)


def _compose_parent_to_child_with_local_trim(
    base_xyz_m: Vec3,
    base_rpy_rad: Vec3,
    local_xyz_offset_m: Vec3,
    local_rpy_offset_rad: Vec3,
) -> Tuple[Vec3, Vec3]:
    base_rot = _rot_matrix_from_rpy(base_rpy_rad[0], base_rpy_rad[1], base_rpy_rad[2])
    local_rot = _rot_matrix_from_rpy(
        local_rpy_offset_rad[0],
        local_rpy_offset_rad[1],
        local_rpy_offset_rad[2],
    )
    trimmed_xyz_m = (
        float(base_xyz_m[0] + _matvec3(base_rot, local_xyz_offset_m)[0]),
        float(base_xyz_m[1] + _matvec3(base_rot, local_xyz_offset_m)[1]),
        float(base_xyz_m[2] + _matvec3(base_rot, local_xyz_offset_m)[2]),
    )
    trimmed_rot = _matmul3(base_rot, local_rot)
    trimmed_rpy_rad = _rpy_from_rot_matrix(trimmed_rot)
    return trimmed_xyz_m, trimmed_rpy_rad


def _parse_xyz_attribute(origin_element, joint_name: str) -> Vec3:
    xyz_text = str(origin_element.attrib.get("xyz", "")).strip()
    values = [float(value) for value in xyz_text.split()]
    if len(values) != 3:
        raise RuntimeError(f"Expected 3 xyz values in {joint_name} origin, got: {xyz_text}")
    return (float(values[0]), float(values[1]), float(values[2]))


def _parse_rpy_attribute(origin_element, joint_name: str) -> Vec3:
    rpy_text = str(origin_element.attrib.get("rpy", "")).strip()
    values = [float(value) for value in rpy_text.split()]
    if len(values) != 3:
        raise RuntimeError(f"Expected 3 rpy values in {joint_name} origin, got: {rpy_text}")
    return (float(values[0]), float(values[1]), float(values[2]))


def _load_joint_origin_from_urdf(joint_name: str) -> Tuple[Vec3, Vec3]:
    if not os.path.isfile(_source_urdf_path):
        raise FileNotFoundError(f"Source URDF not found: {_source_urdf_path}")

    root = ET.parse(_source_urdf_path).getroot()
    for joint_element in root.findall("joint"):
        if str(joint_element.attrib.get("name")) != str(joint_name):
            continue
        origin_element = joint_element.find("origin")
        if origin_element is None:
            raise RuntimeError(f"{joint_name} is missing an <origin> element in source URDF")
        return _parse_xyz_attribute(origin_element, joint_name), _parse_rpy_attribute(origin_element, joint_name)

    raise RuntimeError(f"Could not find joint name='{joint_name}' in source URDF")


def load_sonopet_tcp_geometry() -> SonopetTcpGeometry:
    """Return nominal Sonopet TCP plus post-TCP trim offsets."""

    ee_base_xyz_m, ee_base_rpy_rad = _load_joint_origin_from_urdf("EE_base_joint")
    nominal_urdf_joint_xyz_m, nominal_urdf_joint_rpy_rad = _load_joint_origin_from_urdf("sonopet_tcp_joint")
    nominal_link8_to_tcp_xyz_m, nominal_link8_to_tcp_rpy_rad = _compose_parent_to_child_transform(
        parent_to_mid_xyz_m=ee_base_xyz_m,
        parent_to_mid_rpy_rad=ee_base_rpy_rad,
        mid_to_child_xyz_m=nominal_urdf_joint_xyz_m,
        mid_to_child_rpy_rad=nominal_urdf_joint_rpy_rad,
    )
    tcp_offset_rpy_rad = (
        float(math.radians(SONOPET_TCP_ROLL_OFFSET_DEG)),
        float(math.radians(SONOPET_TCP_PITCH_OFFSET_DEG)),
        float(math.radians(SONOPET_TCP_YAW_OFFSET_DEG)),
    )
    trimmed_link8_to_tcp_xyz_m, trimmed_link8_to_tcp_rpy_rad = _compose_parent_to_child_with_local_trim(
        base_xyz_m=nominal_link8_to_tcp_xyz_m,
        base_rpy_rad=nominal_link8_to_tcp_rpy_rad,
        local_xyz_offset_m=(
            float(SONOPET_TCP_X_OFFSET_M),
            float(SONOPET_TCP_Y_OFFSET_M),
            float(SONOPET_TCP_Z_OFFSET_M),
        ),
        local_rpy_offset_rad=tcp_offset_rpy_rad,
    )

    return SonopetTcpGeometry(
        nominal_urdf_joint_xyz_m=nominal_urdf_joint_xyz_m,
        nominal_urdf_joint_rpy_rad=nominal_urdf_joint_rpy_rad,
        nominal_link8_to_tcp_xyz_m=nominal_link8_to_tcp_xyz_m,
        nominal_link8_to_tcp_rpy_rad=nominal_link8_to_tcp_rpy_rad,
        nominal_link8_to_tcp_quat_xyzw=_quat_from_rpy(
            float(nominal_link8_to_tcp_rpy_rad[0]),
            float(nominal_link8_to_tcp_rpy_rad[1]),
            float(nominal_link8_to_tcp_rpy_rad[2]),
        ),
        tcp_offset_xyz_m=(
            float(SONOPET_TCP_X_OFFSET_M),
            float(SONOPET_TCP_Y_OFFSET_M),
            float(SONOPET_TCP_Z_OFFSET_M),
        ),
        tcp_offset_rpy_rad=tcp_offset_rpy_rad,
        tcp_offset_quat_xyzw=_quat_from_rpy(
            float(tcp_offset_rpy_rad[0]),
            float(tcp_offset_rpy_rad[1]),
            float(tcp_offset_rpy_rad[2]),
        ),
        trimmed_link8_to_tcp_xyz_m=trimmed_link8_to_tcp_xyz_m,
        trimmed_link8_to_tcp_rpy_rad=trimmed_link8_to_tcp_rpy_rad,
        trimmed_link8_to_tcp_quat_xyzw=_quat_from_rpy(
            float(trimmed_link8_to_tcp_rpy_rad[0]),
            float(trimmed_link8_to_tcp_rpy_rad[1]),
            float(trimmed_link8_to_tcp_rpy_rad[2]),
        ),
    )

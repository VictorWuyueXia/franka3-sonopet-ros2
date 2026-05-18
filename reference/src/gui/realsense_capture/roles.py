from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from realsense_boot import build_device_records, format_device_label, select_device_record

from .constants import CAMERA_FIXED_NAME, CAMERA_IN_HAND_NAME, CAMERA_ROLE_CONFIG_PATH, REPO_ROOT
from .stream import RealSenseStream, log_status


@dataclass(frozen=True)
class CameraRoleConfig:
    """Persistent mapping from logical camera role to physical device."""

    camera_name: str
    serial: str
    extrinsics_path: Path
    requires_robot_pose: bool


@dataclass
class ActiveCamera:
    """One configured camera that is currently available on USB."""

    role_config: CameraRoleConfig
    stream: RealSenseStream
    device_info: dict


def list_connected_devices_from_cli() -> None:
    result = subprocess.run(
        ["rs-enumerate-devices"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to run rs-enumerate-devices")
    print(result.stdout.strip())


def list_connected_devices() -> None:
    try:
        device_records = build_device_records()
    except SystemExit:
        list_connected_devices_from_cli()
        return
    if not device_records:
        raise RuntimeError("No RealSense device detected.")
    log_status("Connected RealSense devices:")
    for idx, record in enumerate(device_records):
        print(f"  [{idx}] {format_device_label(record['info'])}")


def camera_extrinsics_path(camera_name: str) -> Path:
    if camera_name == CAMERA_FIXED_NAME:
        return REPO_ROOT / "config" / "camera_extrinsics" / "fr3_eye_to_hand.json"
    if camera_name == CAMERA_IN_HAND_NAME:
        return REPO_ROOT / "config" / "camera_extrinsics" / "fr3_eye_in_hand.json"
    raise ValueError(f"Unsupported camera role: {camera_name}")


def build_camera_role(camera_name: str, serial: str) -> CameraRoleConfig:
    return CameraRoleConfig(
        camera_name=str(camera_name),
        serial=str(serial),
        extrinsics_path=camera_extrinsics_path(camera_name),
        requires_robot_pose=bool(camera_name == CAMERA_IN_HAND_NAME),
    )


def load_camera_roles(fixed_serial_override: str | None, in_hand_serial_override: str | None) -> list[CameraRoleConfig]:
    payload = json.loads(CAMERA_ROLE_CONFIG_PATH.read_text(encoding="utf-8"))
    fixed_serial = fixed_serial_override or str(payload[CAMERA_FIXED_NAME]["serial"])
    in_hand_serial = in_hand_serial_override or str(payload[CAMERA_IN_HAND_NAME]["serial"])
    return [
        build_camera_role(CAMERA_FIXED_NAME, fixed_serial),
        build_camera_role(CAMERA_IN_HAND_NAME, in_hand_serial),
    ]


def list_camera_map(fixed_serial_override: str | None, in_hand_serial_override: str | None) -> None:
    device_records = build_device_records()
    device_map = {record["info"]["serial"]: record["info"] for record in device_records}
    log_status("Configured camera roles:")
    for role in load_camera_roles(fixed_serial_override, in_hand_serial_override):
        info = device_map.get(role.serial)
        suffix = "available" if info is not None else "missing"
        print(f"  {role.camera_name}: serial={role.serial} [{suffix}]")


def _build_active_camera(
    role: CameraRoleConfig,
    record: dict,
    width: int,
    height: int,
    fps: int,
    clip_distance_max: float,
) -> ActiveCamera:
    return ActiveCamera(
        role_config=role,
        stream=RealSenseStream(
            camera_name=role.camera_name,
            serial_request=str(record["info"]["serial"]),
            width=int(width),
            height=int(height),
            fps=int(fps),
            clip_distance_max=float(clip_distance_max),
        ),
        device_info=record["info"],
    )


def resolve_active_cameras(
    camera_roles: list[CameraRoleConfig],
    width: int,
    height: int,
    fps: int,
    clip_distance_max: float,
) -> list[ActiveCamera]:
    device_records = build_device_records()
    used_serials = set()
    active_cameras = []
    for role in camera_roles:
        try:
            record = select_device_record(device_records, role.serial)
        except RuntimeError as exc:
            log_status(f"{role.camera_name}: unavailable ({exc})")
            continue
        serial_number = str(record["info"]["serial"])
        if serial_number in used_serials:
            log_status(f"{role.camera_name}: duplicate serial {serial_number}; skipping duplicate mapping")
            continue
        used_serials.add(serial_number)
        active_cameras.append(
            _build_active_camera(
                role=role,
                record=record,
                width=width,
                height=height,
                fps=fps,
                clip_distance_max=clip_distance_max,
            )
        )
    if not active_cameras:
        raise RuntimeError("No configured RealSense camera is currently available.")
    return active_cameras

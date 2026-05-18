from __future__ import annotations

from dataclasses import dataclass

from ..constants import BASE_FRAME_NAME
from ..robot_pose import RobotCapturePose, RobotTcpPoseResolver
from ..roles import CameraRoleConfig
from ..stream import log_status
from .transforms import (
    TransformSpec,
    apply_parent_frame_trim,
    compose_transform_specs,
    try_load_transform_spec,
)


@dataclass(frozen=True)
class BaseAlignmentResult:
    """Resolved camera optical frame to robot base frame transform."""

    base_transform: TransformSpec | None
    skip_reason: str | None
    robot_capture: RobotCapturePose | None


def cache_robot_pose_for_role(
    role_config: CameraRoleConfig,
    robot_pose_resolver: RobotTcpPoseResolver,
    robot_capture_by_frame: dict[str, RobotCapturePose | None],
) -> None:
    """Capture live robot pose once per parent frame for a camera snapshot."""

    if not role_config.requires_robot_pose:
        return
    transform_spec, load_reason = try_load_transform_spec(role_config.extrinsics_path)
    if transform_spec is None:
        log_status(f"{role_config.camera_name}: cannot pre-capture robot pose - {load_reason or 'unknown'}")
        return
    parent_frame = str(transform_spec.parent_frame)
    if parent_frame in robot_capture_by_frame:
        return
    robot_capture_by_frame[parent_frame] = robot_pose_resolver.capture_base_to_frame(parent_frame)


def resolve_base_alignment(
    role_config: CameraRoleConfig,
    robot_pose_resolver: RobotTcpPoseResolver,
    robot_capture_by_frame: dict[str, RobotCapturePose | None],
    camera_in_hand_parent_trim_m: tuple[float, float, float],
    camera_in_hand_parent_yaw_trim_deg: float,
) -> BaseAlignmentResult:
    """Resolve the transform that moves one camera cloud into the robot base frame."""

    camera_transform, load_reason = try_load_transform_spec(role_config.extrinsics_path)
    if camera_transform is None:
        return BaseAlignmentResult(None, load_reason, None)
    if not role_config.requires_robot_pose:
        return BaseAlignmentResult(camera_transform, None, None)
    parent_frame = str(camera_transform.parent_frame)
    robot_capture = robot_capture_by_frame.get(parent_frame)
    if robot_capture is None:
        robot_capture = robot_pose_resolver.capture_base_to_frame(parent_frame)
        robot_capture_by_frame[parent_frame] = robot_capture
    if robot_capture is None:
        reason = f"Missing live {BASE_FRAME_NAME}-to-{parent_frame} pose for camera-in-hand capture"
        return BaseAlignmentResult(None, reason, None)
    trimmed_camera_transform = apply_parent_frame_trim(
        transform_spec=camera_transform,
        offset_xyz=camera_in_hand_parent_trim_m,
        yaw_deg=float(camera_in_hand_parent_yaw_trim_deg),
    )
    base_transform = compose_transform_specs(robot_capture.base_to_tcp, trimmed_camera_transform)
    return BaseAlignmentResult(base_transform, None, robot_capture)

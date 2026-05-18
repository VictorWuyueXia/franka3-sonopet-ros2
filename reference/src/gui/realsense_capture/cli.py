from __future__ import annotations

import argparse

from .capture_app import DualCameraCaptureApp
from .legacy_capture import run_legacy_single_capture
from .pointcloud.registration import (
    REFINEMENT_DOFS,
    REFINEMENT_MODES,
    RegistrationSettings,
    build_default_registration_settings,
)
from .robot_pose import RobotTcpPoseResolver
from .roles import list_camera_map, list_connected_devices, load_camera_roles, resolve_active_cameras
from .stream import DEFAULT_CLIP_DISTANCE_M, DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture dual D405 point clouds with standard camera-fixed/camera-in-hand roles."
    )
    parser.add_argument(
        "--serial",
        type=str,
        help="Legacy single-camera mode: use one RealSense full serial number or last 4 digits.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print connected RealSense devices and exit.",
    )
    parser.add_argument(
        "--list-camera-map",
        action="store_true",
        help="Print logical camera role mapping and exit.",
    )
    parser.add_argument(
        "--camera-fixed-serial",
        type=str,
        help="Override serial for the camera-fixed role.",
    )
    parser.add_argument(
        "--camera-in-hand-serial",
        type=str,
        help="Override serial for the camera-in-hand role.",
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default="172.16.0.2",
        help="Robot IP used to probe live fr3_hand_tcp pose for camera-in-hand stitching.",
    )
    _add_stream_args(parser)
    return parser


def _add_stream_args(parser: argparse.ArgumentParser) -> None:
    registration_defaults = build_default_registration_settings()
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="Color and depth stream width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="Color and depth stream height in pixels.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Color and depth stream frame rate.",
    )
    parser.add_argument(
        "--clip-distance-max",
        type=float,
        default=DEFAULT_CLIP_DISTANCE_M,
        help="Maximum depth distance in meters kept in the point cloud.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="pointcloud",
        help="Legacy single-camera label used for scan filenames.",
    )
    parser.add_argument(
        "--registration-min-fitness",
        type=float,
        default=registration_defaults.min_fitness,
        help="Minimum ICP fitness required before the refined in-hand cloud replaces the TF-aligned cloud.",
    )
    parser.add_argument(
        "--registration-max-rmse",
        type=float,
        default=registration_defaults.max_inlier_rmse_m,
        help="Maximum accepted ICP inlier RMSE in meters.",
    )
    parser.add_argument(
        "--registration-overlap-margin",
        type=float,
        default=registration_defaults.overlap_margin_m,
        help="Extra margin in meters added to the overlap crop used only for ICP preprocessing.",
    )
    parser.add_argument(
        "--registration-max-translation-delta-mm",
        type=float,
        default=registration_defaults.max_translation_delta_m * 1000.0,
        help="Maximum accepted ICP translation delta relative to the TF prior, in millimeters.",
    )
    parser.add_argument(
        "--registration-max-rotation-delta-deg",
        type=float,
        default=registration_defaults.max_rotation_delta_deg,
        help="Maximum accepted ICP rotation delta relative to the TF prior, in degrees.",
    )
    parser.add_argument(
        "--registration-refinement-mode",
        type=str,
        choices=REFINEMENT_MODES,
        default=registration_defaults.refinement_mode,
        help="How Open3D stitching refinement is applied: always, never, or auto (threshold-gated).",
    )
    parser.add_argument(
        "--registration-refinement-dofs",
        type=str,
        choices=REFINEMENT_DOFS,
        default=registration_defaults.refinement_dofs,
        help="Which transform DOFs ICP may change: z, translation, or full.",
    )
    parser.add_argument(
        "--camera-in-hand-parent-x-trim-mm",
        type=float,
        default=0.0, # 20.0,
        help="Small parent-frame X trim applied to the camera-in-hand extrinsics before composing into the base frame.",
    )
    parser.add_argument(
        "--camera-in-hand-parent-y-trim-mm",
        type=float,
        default=0.0, # 30,
        help="Small parent-frame Y trim applied to the camera-in-hand extrinsics before composing into the base frame.",
    )
    parser.add_argument(
        "--camera-in-hand-parent-z-trim-mm",
        "--camera-in-hand-base-z-trim-mm",
        dest="camera_in_hand_parent_z_trim_mm",
        type=float,
        default=5,
        help="Small parent-frame Z trim applied to the camera-in-hand extrinsics before composing into the base frame.",
    )
    parser.add_argument(
        "--camera-in-hand-parent-yaw-trim-deg",
        type=float,
        default=0.0,
        help="Small parent-frame yaw trim in degrees applied to the camera-in-hand extrinsics before composing into the base frame.",
    )


def parse_args():
    parser = build_arg_parser()
    return parser.parse_args()


def build_registration_settings(args) -> RegistrationSettings:
    return RegistrationSettings(
        refinement_mode=str(args.registration_refinement_mode),
        refinement_dofs=str(args.registration_refinement_dofs),
        min_fitness=float(args.registration_min_fitness),
        max_inlier_rmse_m=float(args.registration_max_rmse),
        max_translation_delta_m=float(args.registration_max_translation_delta_mm) / 1000.0,
        max_rotation_delta_deg=float(args.registration_max_rotation_delta_deg),
        overlap_margin_m=float(args.registration_overlap_margin),
    )


def main() -> None:
    args = parse_args()
    if args.list_devices:
        list_connected_devices()
        return
    if args.list_camera_map:
        list_camera_map(args.camera_fixed_serial, args.camera_in_hand_serial)
        return
    if args.serial:
        run_legacy_single_capture(
            serial=args.serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            clip_distance_max=args.clip_distance_max,
            label=args.label,
        )
        return
    camera_roles = load_camera_roles(args.camera_fixed_serial, args.camera_in_hand_serial)
    active_cameras = resolve_active_cameras(
        camera_roles=camera_roles,
        width=args.width,
        height=args.height,
        fps=args.fps,
        clip_distance_max=args.clip_distance_max,
    )
    DualCameraCaptureApp(
        active_cameras=active_cameras,
        robot_pose_resolver=RobotTcpPoseResolver(args.robot_ip),
        registration_settings=build_registration_settings(args),
        camera_in_hand_parent_trim_m=(
            float(args.camera_in_hand_parent_x_trim_mm) / 1000.0,
            float(args.camera_in_hand_parent_y_trim_mm) / 1000.0,
            float(args.camera_in_hand_parent_z_trim_mm) / 1000.0,
        ),
        camera_in_hand_parent_yaw_trim_deg=float(args.camera_in_hand_parent_yaw_trim_deg),
    ).run()

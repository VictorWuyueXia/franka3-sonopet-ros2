"""CLI parsing for the Franky datacollection pipeline."""

from __future__ import annotations

import argparse

from .config import PipelineConfig


def parse_pipeline_args(default_cfg: PipelineConfig) -> argparse.Namespace:
    """Parse user-facing pipeline arguments."""

    p = argparse.ArgumentParser()
    p.add_argument(
        "--robot-ip",
        type=str,
        default=str(default_cfg.input_defaults.robot_ip),
        help="FCI IP address of the real robot (optional)",
    )
    p.add_argument(
        "--preview-only",
        action="store_true",
        help="Force preview only, never execute hardware",
    )
    p.add_argument(
        "--pcd-path",
        type=str,
        default=str(default_cfg.pointcloud.pcd_path),
        help="Point cloud file path",
    )
    p.add_argument(
        "--pcd-frame",
        type=str,
        choices=["camera_optical", "fr3_link0"],
        default=str(default_cfg.pointcloud.pointcloud_frame),
        help="Frame of the input point cloud: camera_optical | fr3_link0",
    )
    p.add_argument(
        "--square-side",
        type=float,
        default=float(default_cfg.pointcloud.square_side_m),
        help="Square patch side length in meters",
    )
    p.add_argument(
        "--square-center",
        dest="square_center",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional square center in FR3 base frame. If set, skip interactive Open3D point picking.",
    )
    p.add_argument(
        "--line-spacing",
        dest="line_spacing_m",
        type=float,
        default=float(default_cfg.pointcloud.line_spacing_m),
        help="Spacing between selected-square point-cloud scan lines in meters",
    )
    p.add_argument(
        "--pointcloud-downsample-rate",
        dest="pointcloud_downsample_rate",
        type=int,
        default=int(default_cfg.pointcloud.pointcloud_downsample_rate),
        help="Per-line point-cloud downsample rate; 1 keeps every point",
    )
    p.add_argument(
        "--raster-pattern",
        type=str,
        choices=["boustrophedon", "unidirectional_retract"],
        default=str(default_cfg.pointcloud.raster_pattern),
        help="Raster pattern built from selected square point-cloud points",
    )
    p.add_argument(
        "--raster-z-offset",
        dest="raster_z_offset_m",
        type=float,
        default=float(default_cfg.pointcloud.raster_z_offset_m),
        help="Temporary world-Z offset applied to generated raster waypoints in meters",
    )
    p.add_argument(
        "--dynamics-scale",
        type=float,
        default=1.0,
        help="Global compile-time trajectory dynamics scale (velocity by s, acceleration by s^2)",
    )
    p.add_argument(
        "--sim-time-scale",
        type=float,
        default=float(default_cfg.preview.sim_time_scale),
        help="Preview realtime speed multiplier (3 means 3x realtime)",
    )
    p.add_argument(
        "--orientation-mode",
        type=str,
        choices=["surface_normal", "current", "preset"],
        default="current",
        help="Waypoint orientation policy: surface_normal | current | preset",
    )
    p.add_argument(
        "--final-return-mode",
        type=str,
        choices=["workflow_start", "idle"],
        default="workflow_start",
        help="Final E2 return target: workflow_start | idle",
    )
    p.add_argument(
        "--preset-orientation",
        dest="preset_orientation",
        type=str,
        choices=["idle"],
        default=None,
        help="Named preset orientation, used only when --orientation-mode preset",
    )
    p.add_argument(
        "--preset-rpy-deg",
        dest="preset_rpy_deg",
        nargs=3,
        type=float,
        default=None,
        metavar=("ROLL_DEG", "PITCH_DEG", "YAW_DEG"),
        help="Preset orientation in degrees, used only when --orientation-mode preset",
    )
    p.add_argument(
        "--preset-quat-xyzw",
        dest="preset_quat_xyzw",
        nargs=4,
        type=float,
        default=None,
        metavar=("QX", "QY", "QZ", "QW"),
        help="Preset orientation quaternion, used only when --orientation-mode preset",
    )
    return p.parse_args()


def validate_pipeline_args(args: argparse.Namespace) -> None:
    """Validate user-facing pipeline arguments."""

    if not args.pcd_path:
        raise SystemExit("--pcd-path is required")
    if str(args.pcd_frame) not in {"camera_optical", "fr3_link0"}:
        raise ValueError("--pcd-frame must be camera_optical or fr3_link0")
    if float(args.square_side) <= 0.0:
        raise ValueError("--square-side must be > 0")
    if float(args.line_spacing_m) <= 0.0:
        raise ValueError("--line-spacing must be > 0")
    if float(args.line_spacing_m) >= float(args.square_side) / 2.0:
        raise ValueError("--line-spacing must be < half of --square-side")
    if int(args.pointcloud_downsample_rate) < 1:
        raise ValueError("--pointcloud-downsample-rate must be >= 1")
    if str(args.raster_pattern) not in {"boustrophedon", "unidirectional_retract"}:
        raise ValueError("--raster-pattern must be boustrophedon or unidirectional_retract")
    if float(args.dynamics_scale) <= 0.0:
        raise ValueError("--dynamics-scale must be > 0")
    if float(args.sim_time_scale) <= 0.0:
        raise ValueError("--sim-time-scale must be > 0")
    if str(args.final_return_mode) not in {"workflow_start", "idle"}:
        raise ValueError("--final-return-mode must be workflow_start or idle")
    if str(args.orientation_mode) == "preset":
        preset_count = sum(
            [
                args.preset_orientation is not None,
                args.preset_rpy_deg is not None,
                args.preset_quat_xyzw is not None,
            ]
        )
        if preset_count != 1:
            raise ValueError(
                "Preset mode requires exactly one of --preset-orientation idle, "
                "--preset-rpy-deg, or --preset-quat-xyzw"
            )


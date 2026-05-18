"""Centralized configuration for the Franky joint waypoint pipeline.

Design goals:
- Keep all stable defaults and runtime policies in one file.
- Keep user-facing CLI inputs narrow and focused.
- Make CPU-affinity and runtime behavior deterministic by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# Internal canonical FR3 joint ordering used across compile and execution.
JOINT_NAMES: List[str] = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]


@dataclass(frozen=True)
class IKCheckConfig:
    """Thresholds for accepting an IK solution based on forward kinematics."""

    max_pos_err_m: float = 0.005
    min_quat_abs_dot: float = 0.995


@dataclass(frozen=True)
class PipelineTiming:
    """Timing and discretization parameters."""

    # PyBullet physics timestep for preview.
    sim_dt_s: float = 1.0 / 240.0

    # Segment durations (used for joint interpolation segments).
    reset_current_to_idle_s: float = 10.0
    reset_end_to_idle_s: float = 10.0


@dataclass(frozen=True)
class PathDensifyConfig:
    """Densification constraints in Cartesian space."""

    # Enforce ||p[k+1]-p[k]|| <= max_step_m
    max_step_m: float = 0.01


@dataclass(frozen=True)
class StageConfig:
    """Geometric staging parameters."""

    # Lift above the first raster waypoint before descending.
    parking_lift_m: float = 0.05

    # Lift at end before returning to idle.
    retract_lift_m: float = 0.05


@dataclass(frozen=True)
class FrankyConfig:
    """Hardware execution tuning."""

    # Conservative hardware OTG scaling for position-only waypoint execution.
    relative_dynamics_factor: float = 0.05

    # Slow down contact-adjacent motion for descent and raster scan segments.
    segment_cd_relative_dynamics_factor: float = 0.005

    # Clamp internal compile-time velocity hints used for trajectory shaping checks.
    # Hardware execution currently sends position-only waypoints to Franky.
    max_joint_vel_rad_s: float = 0.79


@dataclass(frozen=True)
class PreviewFallbackConfig:
    """What to do when no physical robot is available."""

    # A "current" pose used when hardware is not connected.
    # This must be feasible and within joint limits.
    # It should NOT equal the idle pose so Segment A is visible in preview.
    current_joint_pose_by_name: Dict[str, float] = None  # set in __post_init__

    def __post_init__(self) -> None:
        if self.current_joint_pose_by_name is None:
            object.__setattr__(
                self,
                "current_joint_pose_by_name",
                {
                    "fr3_joint1": 0.35,
                    "fr3_joint2": -0.55,
                    "fr3_joint3": 0.20,
                    "fr3_joint4": -2.10,
                    "fr3_joint5": 0.15,
                    "fr3_joint6": 1.55,
                    "fr3_joint7": 0.90,
                },
            )


@dataclass(frozen=True)
class RuntimePolicyConfig:
    """Fixed runtime policy knobs (not exposed as CLI flags)."""

    # Always keep runtime diagnostics enabled for easier field diagnosis.
    debug_log: bool = True
    # Always use short-lived subprocess probe for robot pose capture.
    probe_robot_subprocess: bool = True
    # Keep GPU renderer preference as an internal default policy.
    prefer_gpu_renderer: bool = False
    # Always enforce RT priority cap during robot thread mitigation.
    rt_priority_cap: int = 80


@dataclass(frozen=True)
class PointCloudDefaults:
    """Default point-cloud and raster compile parameters."""

    pcd_path: str = "./pointclouds/scans/pointcloud_1773775961.pcd"
    pointcloud_frame: str = "fr3_link0"
    square_side_m: float = 0.02
    line_spacing_m: float = 0.002
    pointcloud_downsample_rate: int = 6
    raster_pattern: str = "unidirectional_retract" #"boustrophedon"
    search_radius_m: float = 0.02
    min_normal_points: int = 3
    tool_offset_m: float = 0.0
    tool_normal_sign: float = -1.0
    raster_z_offset_m: float = 0


@dataclass(frozen=True)
class PreviewDefaults:
    """Default preview behavior and pacing."""

    gui_enabled: bool = True
    realtime_sleep: bool = True
    sim_time_scale: float = 5.0
    tcp_frame_interval: int = 10


@dataclass(frozen=True)
class InputDefaults:
    """Default external runtime inputs."""

    robot_ip: str = "172.16.0.2"


@dataclass(frozen=True)
class PipelineConfig:
    """Full configuration bundle."""

    ik_check: IKCheckConfig = IKCheckConfig()
    timing: PipelineTiming = PipelineTiming()
    densify: PathDensifyConfig = PathDensifyConfig()
    stages: StageConfig = StageConfig()
    franky: FrankyConfig = FrankyConfig()
    preview_fallback: PreviewFallbackConfig = PreviewFallbackConfig()
    runtime_policy: RuntimePolicyConfig = RuntimePolicyConfig()
    pointcloud: PointCloudDefaults = PointCloudDefaults()
    preview: PreviewDefaults = PreviewDefaults()
    input_defaults: InputDefaults = InputDefaults()


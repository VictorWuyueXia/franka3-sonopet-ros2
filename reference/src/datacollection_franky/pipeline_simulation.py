"""Simulation-stage helpers for point cloud, planning, and preview."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import PipelineConfig
from .preview import reset_robot_joints, run_joint_waypoint_preview
from .prompts import info, warn
from .sim_runtime import (
    apply_orientation_mode,
    build_pybullet_gui_env_overrides,
    connect_bullet_sim_with_env,
    prepare_open3d_picker_environment,
    prompt_preview_action,
    pybullet_probe_requires_env_fallback,
    probe_pybullet_gui_subprocess,
    resolve_target_quaternion,
    restore_open3d_picker_environment,
    select_ee_link_index,
    validate_joint_dict,
)
from .trajectory_compile import _tcp_pose_at_joint, compile_full_joint_trajectory
from .visualization import draw_pipeline_visuals


def square_center_override(args) -> Optional[Tuple[float, float, float]]:
    """Normalize optional square-center CLI tuple."""

    if args.square_center is None:
        return None
    return (
        float(args.square_center[0]),
        float(args.square_center[1]),
        float(args.square_center[2]),
    )


def build_pose_waypoints(
    args,
    cfg: PipelineConfig,
    PointCloudWrapper,
    RasterSurfaceSpec,
    PoseWaypoint,
):
    """Create lifted pose waypoints from point cloud sampling."""

    center_override = square_center_override(args)
    if center_override is not None:
        info(f"Using manual square center override: {center_override}")
    glx_vendor_backup = prepare_open3d_picker_environment(center_override)
    pointcloud_vis_enabled = center_override is None
    wrapper = PointCloudWrapper(
        pcd_path=args.pcd_path,
        vis_flag=bool(pointcloud_vis_enabled),
        pcd_frame=str(args.pcd_frame),
    )
    spec = RasterSurfaceSpec(
        square_side_len_m=float(args.square_side),
        line_spacing_m=float(args.line_spacing_m),
        pointcloud_downsample_rate=int(args.pointcloud_downsample_rate),
        raster_pattern=str(args.raster_pattern),
        search_radius_m=float(cfg.pointcloud.search_radius_m),
        normal_projection_m=1e-6,
        min_points_for_normal=int(cfg.pointcloud.min_normal_points),
        use_surface_normal=True,
        tool_offset_m=float(cfg.pointcloud.tool_offset_m),
        tool_normal_sign=float(cfg.pointcloud.tool_normal_sign),
    )
    try:
        pose_waypoints = wrapper.get_sim_waypoints(spec, square_center_override=center_override)
    finally:
        restore_open3d_picker_environment(glx_vendor_backup)
    if not pose_waypoints:
        raise SystemExit("No pose waypoints were generated from point cloud")

    square_center_xyz = tuple(wrapper.last_square_center.tolist())
    square_corners_xyz = None
    if wrapper.last_square_corners is not None:
        square_corners_xyz = [
            (
                float(corner[0]),
                float(corner[1]),
                float(corner[2]),
            )
            for corner in wrapper.last_square_corners
        ]
    lifted: List[PoseWaypoint] = []
    for wp in pose_waypoints:
        lifted.append(
            PoseWaypoint(
                xyz=(
                    float(wp.xyz[0]),
                    float(wp.xyz[1]),
                    float(wp.xyz[2]) + float(args.raster_z_offset_m),
                ),
                quat_xyzw=wp.quat_xyzw,
                normal_xyz=wp.normal_xyz,
            )
        )
    return lifted, square_center_xyz, square_corners_xyz


def run_preview_stage(sim, compiled, ee_link_index: int, cfg: PipelineConfig, args, log_runtime_snapshot) -> None:
    """Run preview interaction and playback for compiled trajectory."""

    if not bool(cfg.preview.gui_enabled):
        info("Preview skipped because GUI is disabled by config.")
        return
    preview_action = prompt_preview_action(bool(cfg.preview.gui_enabled))
    if preview_action == "run":
        run_joint_waypoint_preview(
            sim=sim,
            trajectory_points=compiled.points,
            realtime_sleep=bool(cfg.preview.realtime_sleep),
            sim_time_scale=float(args.sim_time_scale),
            ee_link_index=ee_link_index,
            tcp_frame_interval=int(cfg.preview.tcp_frame_interval),
        )
        info("Preview finished.")
        log_runtime_snapshot("after_preview", enabled=bool(cfg.runtime_policy.debug_log))
        return
    info("Preview skipped by user input <s>.")


def run_pointcloud_stage(args, cfg: PipelineConfig, PointCloudWrapper, RasterSurfaceSpec, PoseWaypoint, log_runtime_snapshot):
    """Run point-cloud pipeline and return simulation-ready waypoints."""

    gui_enabled = bool(cfg.preview.gui_enabled)
    probe_pybullet_gui_subprocess("before_pointcloud", gui_enabled, {})
    pose_waypoints, square_center_xyz, square_corners_xyz = build_pose_waypoints(
        args=args,
        cfg=cfg,
        PointCloudWrapper=PointCloudWrapper,
        RasterSurfaceSpec=RasterSurfaceSpec,
        PoseWaypoint=PoseWaypoint,
    )
    info(f"Pose waypoints: {len(pose_waypoints)}")
    log_runtime_snapshot("after_pointcloud_waypoints", enabled=bool(cfg.runtime_policy.debug_log))
    return gui_enabled, pose_waypoints, square_center_xyz, square_corners_xyz


def prepare_sim_connect(
    cfg: PipelineConfig,
    gui_enabled: bool,
    ik_cpu_set,
    repin_main_thread_to_ik,
):
    """Resolve environment strategy for live PyBullet connection."""

    repin_main_thread_to_ik(ik_cpu_set)
    pybullet_gui_env_overrides = build_pybullet_gui_env_overrides(
        bool(gui_enabled and cfg.runtime_policy.prefer_gpu_renderer)
    )
    before_sim_probe_payload = probe_pybullet_gui_subprocess(
        "before_sim_connect",
        gui_enabled,
        pybullet_gui_env_overrides,
    )
    if bool(gui_enabled and cfg.runtime_policy.prefer_gpu_renderer):
        if pybullet_probe_requires_env_fallback(before_sim_probe_payload):
            warn(
                "Preferred PyBullet GPU env probe failed or looked unsafe. "
                "Falling back to plain GUI env for live connect."
            )
            pybullet_gui_env_overrides = {}
    return pybullet_gui_env_overrides


def configure_pose_orientation(sim, q_current: Dict[str, float], ee_link_index: int, pose_waypoints, args):
    """Validate simulation states and apply orientation-mode policy."""

    validate_joint_dict(sim, q_current, name="q_current")
    reset_robot_joints(sim, q_current)
    _, current_tcp_quat = sim.get_tcp_pose(ee_link_index)
    q_idle = sim.compute_idle_pose_from_urdf()
    validate_joint_dict(sim, q_idle, name="q_idle")
    idle_tcp_pose = _tcp_pose_at_joint(sim, ee_link_index, q_idle)
    target_quat_xyzw = resolve_target_quaternion(
        args=args,
        current_quat_xyzw=(
            float(current_tcp_quat[0]),
            float(current_tcp_quat[1]),
            float(current_tcp_quat[2]),
            float(current_tcp_quat[3]),
        ),
        idle_quat_xyzw=idle_tcp_pose.quat_xyzw,
    )
    adjusted_pose_waypoints = apply_orientation_mode(
        pose_waypoints=pose_waypoints,
        orientation_mode=str(args.orientation_mode),
        target_quat_xyzw=target_quat_xyzw,
    )
    info(
        "Orientation mode applied: "
        f"mode={args.orientation_mode}, "
        f"target_quat_xyzw=({target_quat_xyzw[0]:.6f}, {target_quat_xyzw[1]:.6f}, "
        f"{target_quat_xyzw[2]:.6f}, {target_quat_xyzw[3]:.6f})"
    )
    return q_idle, adjusted_pose_waypoints


def compile_offline_trajectory(
    sim,
    ee_link_index: int,
    pose_waypoints,
    q_current: Dict[str, float],
    q_idle,
    cfg: PipelineConfig,
    dynamics_scale: float,
    final_return_mode: str,
    log_runtime_snapshot,
):
    """Compile full runtime trajectory offline in PyBullet."""

    info("Compiling joint trajectory (offline IK in PyBullet)...")
    import time

    t_compile0 = float(time.monotonic())
    compiled = compile_full_joint_trajectory(
        sim=sim,
        ee_link_index=ee_link_index,
        pose_waypoints=pose_waypoints,
        q_current=q_current,
        q_idle=q_idle,
        cfg=cfg,
        dynamics_scale=float(dynamics_scale),
        final_return_mode=str(final_return_mode),
    )
    info(f"Offline IK compile elapsed_s={float(time.monotonic()) - t_compile0:.3f}")
    log_runtime_snapshot("after_offline_compile", enabled=bool(cfg.runtime_policy.debug_log))
    return compiled


def log_compiled_summary(compiled, dynamics_scale: float) -> None:
    """Log total points and segment boundaries from compiled trajectory."""

    info(f"Total runtime trajectory points: {len(compiled.points)}")
    for seg, a, b in compiled.segments:
        info(f"Segment {seg}: [{a}, {b}) len={b-a}")
    info(
        "Runtime trajectory prepared: "
        f"waypoints={len(compiled.points)}, "
        f"segment_a_interp_dt_s={float(compiled.segment_a_interp_dt_s):.4f}, "
        f"dynamics_scale={float(dynamics_scale):.3f}"
    )


def run_preview_and_disconnect(
    sim,
    compiled,
    ee_link_index: int,
    cfg: PipelineConfig,
    args,
    pose_waypoints,
    q_idle,
    square_center_xyz,
    square_corners_xyz,
    gui_enabled: bool,
    log_runtime_snapshot,
) -> None:
    """Draw helper visuals, run preview flow, and close simulation."""

    if bool(gui_enabled):
        draw_pipeline_visuals(
            sim=sim,
            ee_link_index=ee_link_index,
            pose_waypoints=pose_waypoints,
            q_idle=q_idle,
            parking_lift_m=float(cfg.stages.parking_lift_m),
            square_center_xyz=square_center_xyz,
            square_side_m=float(args.square_side),
            square_corners_xyz=square_corners_xyz,
        )
    run_preview_stage(sim, compiled, ee_link_index, cfg, args, log_runtime_snapshot)
    if not bool(args.preview_only):
        sim.disconnect()
        info("Closed PyBullet simulation client after preview stage.")


def run_sim_compile_preview_stage(
    cfg: PipelineConfig,
    args,
    state: Dict[str, object],
    repin_main_thread_to_ik,
    log_runtime_snapshot,
) -> Dict[str, object]:
    """Run simulation connect, compile, preview, and cleanup stages."""

    pybullet_gui_env_overrides = prepare_sim_connect(
        cfg,
        bool(state["gui_enabled"]),
        state["ik_cpu_set"],
        repin_main_thread_to_ik,
    )
    sim = state["BulletRobotSim"](
        gui=bool(state["gui_enabled"]),
        time_step=float(cfg.timing.sim_dt_s),
    )
    connect_bullet_sim_with_env(sim, pybullet_gui_env_overrides)
    sim.load_robot(state["urdf_path"], fixed_base=True)
    ee_link_index = select_ee_link_index(
        sim.robot_id,
        state["find_joint_index_by_name"],
        state["find_last_link_index"],
    )
    q_idle, pose_waypoints = configure_pose_orientation(
        sim=sim,
        q_current=state["q_current"],
        ee_link_index=ee_link_index,
        pose_waypoints=state["pose_waypoints"],
        args=args,
    )
    compiled = compile_offline_trajectory(
        sim=sim,
        ee_link_index=ee_link_index,
        pose_waypoints=pose_waypoints,
        q_current=state["q_current"],
        q_idle=q_idle,
        cfg=cfg,
        dynamics_scale=float(args.dynamics_scale),
        final_return_mode=str(args.final_return_mode),
        log_runtime_snapshot=log_runtime_snapshot,
    )
    log_compiled_summary(compiled, float(args.dynamics_scale))
    run_preview_and_disconnect(
        sim=sim,
        compiled=compiled,
        ee_link_index=ee_link_index,
        cfg=cfg,
        args=args,
        pose_waypoints=pose_waypoints,
        q_idle=q_idle,
        square_center_xyz=state["square_center_xyz"],
        square_corners_xyz=state.get("square_corners_xyz"),
        gui_enabled=bool(state["gui_enabled"]),
        log_runtime_snapshot=log_runtime_snapshot,
    )
    state["pose_waypoints"] = pose_waypoints
    return state


"""Main orchestrator for Sonopet data-collection pipeline."""

from __future__ import annotations

import os
import sys
import time
from typing import Dict

from .cli import parse_pipeline_args, validate_pipeline_args
from .config import PipelineConfig
from .pipeline_hardware import run_hardware_stage_if_needed
from .pipeline_runtime import (
    acquire_initial_robot_state,
    initialize_runtime_state,
    log_runtime_snapshot,
    repin_main_thread_to_ik,
)
from .pipeline_simulation import run_pointcloud_stage, run_sim_compile_preview_stage
from .prompts import info
from .sim_paths import find_simulation_dir
from .urdf_tcp import find_fr3_urdf


def _load_sim_environment():
    """Locate simulation workspace and import simulation-only dependencies."""

    sim_dir = find_simulation_dir(os.path.dirname(__file__), 6)
    if sim_dir is None:
        raise SystemExit("Could not locate <root>/simulation (containing simWithPyBullet.py)")
    if sim_dir not in sys.path:
        sys.path.insert(0, sim_dir)
    from cloudpoint_wrapper import PointCloudWrapper, PoseWaypoint, RasterSurfaceSpec
    from simWithPyBullet import BulletRobotSim, find_joint_index_by_name, find_last_link_index

    urdf_path = find_fr3_urdf(sim_dir)
    return (
        sim_dir,
        urdf_path,
        PointCloudWrapper,
        PoseWaypoint,
        RasterSurfaceSpec,
        BulletRobotSim,
        find_joint_index_by_name,
        find_last_link_index,
    )


def _prepare_pipeline_inputs(cfg: PipelineConfig, args) -> Dict[str, object]:
    """Prepare runtime state and simulation-stage inputs before planning."""

    (
        sim_dir,
        urdf_path,
        PointCloudWrapper,
        PoseWaypoint,
        RasterSurfaceSpec,
        BulletRobotSim,
        find_joint_index_by_name,
        find_last_link_index,
    ) = _load_sim_environment()
    info(f"Using simulation dir: {sim_dir}")
    info(f"Using URDF: {urdf_path}")
    log_runtime_snapshot("startup", enabled=bool(cfg.runtime_policy.debug_log))
    rt_cpu_set, ik_cpu_set = initialize_runtime_state(cfg)
    q_current, connected, robot_ip_for_execution = acquire_initial_robot_state(
        args=args,
        cfg=cfg,
        ik_cpu_set=ik_cpu_set,
    )
    gui_enabled, pose_waypoints, square_center_xyz, square_corners_xyz = run_pointcloud_stage(
        args=args,
        cfg=cfg,
        PointCloudWrapper=PointCloudWrapper,
        RasterSurfaceSpec=RasterSurfaceSpec,
        PoseWaypoint=PoseWaypoint,
        log_runtime_snapshot=log_runtime_snapshot,
    )
    return {
        "urdf_path": urdf_path,
        "BulletRobotSim": BulletRobotSim,
        "find_joint_index_by_name": find_joint_index_by_name,
        "find_last_link_index": find_last_link_index,
        "rt_cpu_set": rt_cpu_set,
        "ik_cpu_set": ik_cpu_set,
        "q_current": q_current,
        "connected": bool(connected),
        "robot_ip_for_execution": robot_ip_for_execution,
        "gui_enabled": bool(gui_enabled),
        "pose_waypoints": pose_waypoints,
        "square_center_xyz": square_center_xyz,
        "square_corners_xyz": square_corners_xyz,
    }


def main() -> None:
    """Run parse -> prepare -> simulation -> optional hardware stages."""

    cfg = PipelineConfig()
    args = parse_pipeline_args(cfg)
    validate_pipeline_args(args)
    t0 = float(time.monotonic())
    state = _prepare_pipeline_inputs(cfg, args)
    state = run_sim_compile_preview_stage(
        cfg=cfg,
        args=args,
        state=state,
        repin_main_thread_to_ik=repin_main_thread_to_ik,
        log_runtime_snapshot=log_runtime_snapshot,
    )
    run_hardware_stage_if_needed(
        connected=bool(state["connected"]),
        robot_ip_for_execution=state["robot_ip_for_execution"],
        args=args,
        cfg=cfg,
        rt_cpu_set=state["rt_cpu_set"],
        ik_cpu_set=state["ik_cpu_set"],
        urdf_path=state["urdf_path"],
        pose_waypoints=state["pose_waypoints"],
        dynamics_scale=float(args.dynamics_scale),
        BulletRobotSim=state["BulletRobotSim"],
        find_joint_index_by_name=state["find_joint_index_by_name"],
        find_last_link_index=state["find_last_link_index"],
        repin_main_thread_to_ik=repin_main_thread_to_ik,
        log_runtime_snapshot=log_runtime_snapshot,
    )
    info(f"Total elapsed_s={float(time.monotonic()) - t0:.3f}")


if __name__ == "__main__":
    main()

"""Hardware execution helpers for reconnection, recompilation, and segmented motion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pybullet as p

from .config import JOINT_NAMES, PipelineConfig
from .franky_backend import FrankyRobot
from .prompts import confirm_or_abort, error, info, warn
from .sim_runtime import select_ee_link_index, validate_joint_dict
from .trajectory_compile import (
    compile_full_joint_trajectory,
    downsample_runtime_points_for_hardware,
    split_runtime_points_into_motion_segments,
)
from .util.thread_affinity import apply_rt_thread_policy


SEGMENT_CD_NAMES = ("C_parking_to_first", "D_raster")


@dataclass(frozen=True)
class HardwareSegmentPlan:
    """Single hardware command built from one motion segment."""

    segment_name: str
    command_name: str
    hardware_points: List[object]
    relative_dynamics_factor: float


def close_preview_window_if_open() -> None:
    """Close an existing PyBullet preview connection before hardware execution."""

    if not bool(p.isConnected()):
        return
    p.disconnect()
    info("Closed existing PyBullet preview window before hardware execution.")


def connect_robot_for_execution(
    robot_ip: str,
    cfg: PipelineConfig,
    rt_cpu_set,
    ik_cpu_set,
    repin_main_thread_to_ik,
    log_runtime_snapshot,
) -> FrankyRobot:
    """Reconnect hardware robot and apply RT policy before motion."""

    robot = FrankyRobot(robot_ip, cfg.franky)
    if not robot.try_connect():
        raise RuntimeError(f"Failed to reconnect to robot at {robot_ip} for hardware execution")
    info(f"Reconnected to robot at {robot_ip} for hardware execution.")
    policy_result = apply_rt_thread_policy(rt_cpu_set, int(cfg.runtime_policy.rt_priority_cap))
    info(
        "Applied RT policy for hardware execution: "
        f"pinned_rt_threads={int(policy_result['pinned_rt_threads'])}, "
        f"priority_updates={int(policy_result['priority_updates'])}"
    )
    repin_main_thread_to_ik(ik_cpu_set)
    log_runtime_snapshot("after_rt_policy_execute", enabled=bool(cfg.runtime_policy.debug_log))
    return robot


def segment_relative_dynamics_factor(cfg: PipelineConfig, segment_name: str) -> float:
    """Resolve per-segment Franky OTG scaling."""

    if str(segment_name) in SEGMENT_CD_NAMES:
        return float(cfg.franky.segment_cd_relative_dynamics_factor)
    return float(cfg.franky.relative_dynamics_factor)


def build_single_hardware_segment_plan(
    cfg: PipelineConfig,
    segment_name: str,
    segment_points: List[object],
) -> Optional[HardwareSegmentPlan]:
    """Downsample one runtime segment into one hardware motion command."""

    hardware_points = downsample_runtime_points_for_hardware(
        points=segment_points,
        joint_names=JOINT_NAMES,
        near_point_tol_rad=0.01,
        turning_cos_threshold=0.995,
        duplicate_tol_rad=1e-9,
        max_joint_vel_rad_s=float(cfg.franky.max_joint_vel_rad_s),
    )
    if len(hardware_points) < 2:
        warn(
            "Skipping hardware segment with too few waypoints after downsample: "
            f"segment={segment_name}, hardware_points={len(hardware_points)}"
        )
        return None
    return HardwareSegmentPlan(
        segment_name=str(segment_name),
        command_name=str(segment_name),
        hardware_points=hardware_points,
        relative_dynamics_factor=segment_relative_dynamics_factor(cfg, segment_name),
    )


def build_hardware_segment_plans(
    cfg: PipelineConfig,
    segment_name: str,
    segment_points: List[object],
) -> List[HardwareSegmentPlan]:
    """Build one or more hardware commands for one runtime segment."""

    base_plan = build_single_hardware_segment_plan(
        cfg=cfg,
        segment_name=segment_name,
        segment_points=segment_points,
    )
    if base_plan is None:
        return []
    return [base_plan]


def log_final_execution_waypoints(plan: HardwareSegmentPlan) -> None:
    """Log the exact final joint waypoints queued for one hardware move command."""

    n_points = len(plan.hardware_points)
    info(
        "Final execution waypoint queue: "
        f"segment={plan.command_name}, "
        f"points={n_points}, "
        f"relative_dynamics_factor={plan.relative_dynamics_factor:.6f}"
    )
    for idx, point in enumerate(plan.hardware_points, start=1):
        q_summary = ", ".join(
            f"{joint_name}={float(point.q_by_name[joint_name]):.5f}"
            for joint_name in JOINT_NAMES
        )
        hint_label = "with_velocity_hint" if point.dq_hint_by_name is not None else "position_only"
        info(
            "Final execution waypoint: "
            f"segment={plan.command_name}, "
            f"point={idx}/{n_points}, "
            f"{hint_label}, "
            f"{q_summary}"
        )


def compile_execution_hardware_segments(
    robot: FrankyRobot,
    cfg: PipelineConfig,
    urdf_path: str,
    pose_waypoints: List[object],
    dynamics_scale: float,
    final_return_mode: str,
    BulletRobotSim,
    find_joint_index_by_name,
    find_last_link_index,
):
    """Recompile from live state and convert each motion segment into one hardware command."""

    sim_execute = BulletRobotSim(gui=False, time_step=float(cfg.timing.sim_dt_s))
    sim_execute.connect()

    try:
        sim_execute.load_robot(urdf_path, fixed_base=True)
        ee_link_index_execute = select_ee_link_index(
            sim_execute.robot_id,
            find_joint_index_by_name,
            find_last_link_index,
        )
        q_idle_execute = sim_execute.compute_idle_pose_from_urdf()
        q_current_execute = robot.current_joint_positions_by_name()
        validate_joint_dict(sim_execute, q_current_execute, name="q_current_execute")
        validate_joint_dict(sim_execute, q_idle_execute, name="q_idle_execute")
        import time

        t_recompile0 = float(time.monotonic())
        compiled_execute = compile_full_joint_trajectory(
            sim=sim_execute,
            ee_link_index=ee_link_index_execute,
            pose_waypoints=pose_waypoints,
            q_current=q_current_execute,
            q_idle=q_idle_execute,
            cfg=cfg,
            dynamics_scale=float(dynamics_scale),
            final_return_mode=str(final_return_mode),
        )
        info(
            "Execution trajectory recompiled from live robot state: "
            f"elapsed_s={float(time.monotonic()) - t_recompile0:.3f}, "
            f"waypoints={len(compiled_execute.points)}"
        )

        segment_plans: List[HardwareSegmentPlan] = []
        motion_segments = split_runtime_points_into_motion_segments(compiled_execute.points)
        for segment_name, segment_points in motion_segments:
            plans = build_hardware_segment_plans(
                cfg=cfg,
                segment_name=str(segment_name),
                segment_points=segment_points,
            )
            if not plans:
                continue
            hardware_point_count = sum(len(plan.hardware_points) for plan in plans)
            n_vel_hints = sum(
                1
                for plan in plans
                for point in plan.hardware_points
                if point.dq_hint_by_name is not None
            )
            info(
                "Prepared hardware segment: "
                f"segment={segment_name}, "
                f"runtime_points={len(segment_points)}, "
                f"commands={len(plans)}, "
                f"hardware_points={hardware_point_count}, "
                f"relative_dynamics_factor={plans[0].relative_dynamics_factor:.6f}, "
                f"velocity_hints={n_vel_hints}"
            )
            segment_plans.extend(plans)

        if not segment_plans:
            raise RuntimeError("No executable hardware segments were produced after downsample")
        return segment_plans, len(compiled_execute.points)
    finally:
        sim_execute.disconnect()


def run_hardware_execution(
    robot_ip: str,
    cfg: PipelineConfig,
    rt_cpu_set,
    ik_cpu_set,
    urdf_path: str,
    pose_waypoints: List[object],
    dynamics_scale: float,
    final_return_mode: str,
    BulletRobotSim,
    find_joint_index_by_name,
    find_last_link_index,
    repin_main_thread_to_ik,
    log_runtime_snapshot,
) -> None:
    """Execute motion on hardware one segment at a time with reconnect and cleanup."""

    robot = connect_robot_for_execution(
        robot_ip=robot_ip,
        cfg=cfg,
        rt_cpu_set=rt_cpu_set,
        ik_cpu_set=ik_cpu_set,
        repin_main_thread_to_ik=repin_main_thread_to_ik,
        log_runtime_snapshot=log_runtime_snapshot,
    )
    try:
        segment_plans, runtime_point_count = compile_execution_hardware_segments(
            robot=robot,
            cfg=cfg,
            urdf_path=urdf_path,
            pose_waypoints=pose_waypoints,
            dynamics_scale=float(dynamics_scale),
            final_return_mode=str(final_return_mode),
            BulletRobotSim=BulletRobotSim,
            find_joint_index_by_name=find_joint_index_by_name,
            find_last_link_index=find_last_link_index,
        )
        total_hardware_points = sum(len(plan.hardware_points) for plan in segment_plans)
        total_vel_hints = sum(
            1
            for plan in segment_plans
            for point in plan.hardware_points
            if point.dq_hint_by_name is not None
        )
        info(
            "Hardware segment preparation complete: "
            f"runtime_points={runtime_point_count}, "
            f"segments={len(segment_plans)}, "
            f"hardware_points={total_hardware_points}, "
            f"raster_mid_velocity_hints={total_vel_hints}"
        )
        for idx, plan in enumerate(segment_plans, start=1):
            info(
                "Executing hardware segment: "
                f"index={idx}/{len(segment_plans)}, "
                f"segment={plan.command_name}, "
                f"hardware_points={len(plan.hardware_points)}, "
                f"relative_dynamics_factor={plan.relative_dynamics_factor:.6f}"
            )
            log_final_execution_waypoints(plan)
            robot.execute_joint_waypoints(
                trajectory_points=plan.hardware_points,
                relative_dynamics_factor=float(plan.relative_dynamics_factor),
            )
            info(f"Completed hardware segment {plan.command_name}.")
        info("Hardware execution completed.")
    except Exception as exc:
        error(f"Hardware execution failed: {exc}")
        raise
    finally:
        robot.disconnect()
        info("Robot disconnected after hardware execution.")


def run_hardware_stage_if_needed(
    connected: bool,
    robot_ip_for_execution: Optional[str],
    args,
    cfg: PipelineConfig,
    rt_cpu_set,
    ik_cpu_set,
    urdf_path: str,
    pose_waypoints,
    dynamics_scale: float,
    BulletRobotSim,
    find_joint_index_by_name,
    find_last_link_index,
    repin_main_thread_to_ik,
    log_runtime_snapshot,
) -> None:
    """Run hardware stage only when connection and mode allow execution."""

    if not connected or bool(args.preview_only) or robot_ip_for_execution is None:
        warn("Hardware execution skipped (no robot connected or preview-only mode).")
        return
    confirm_or_abort(
        "Hardware execution will now move the real robot.\n"
        "Ensure: workspace is clear, E-stop is reachable, and speed is acceptable.",
        token="EXECUTE",
    )
    close_preview_window_if_open()
    run_hardware_execution(
        robot_ip=str(robot_ip_for_execution),
        cfg=cfg,
        rt_cpu_set=rt_cpu_set,
        ik_cpu_set=ik_cpu_set,
        urdf_path=urdf_path,
        pose_waypoints=pose_waypoints,
        dynamics_scale=float(dynamics_scale),
        final_return_mode=str(args.final_return_mode),
        BulletRobotSim=BulletRobotSim,
        find_joint_index_by_name=find_joint_index_by_name,
        find_last_link_index=find_last_link_index,
        repin_main_thread_to_ik=repin_main_thread_to_ik,
        log_runtime_snapshot=log_runtime_snapshot,
    )


#!/usr/bin/env python3
"""
rasterTraj_PyBullet.py
by Victor Xia on 20260218

PyBullet simulator for an FR3 URDF in the same folder.

- Trajectory related code is in the trajectories.py file.
- PyBullet simulation backend is in the simWithPyBullet.py file.
- Visualization code is in the visual_aid.py file.
- Future: backend can be ROS2 control via MoveIt execution.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import math
from typing import List, Sequence, Tuple

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
_repo_root = os.path.dirname(_here)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from trajectories import IKRasterTrajectory, IKPoseSequenceTrajectory, RasterPlanSpec, generate_raster_waypoints
from visual_aid import (
    draw_frame,
    draw_frame_at_pose,
    remove_debug_items,
    draw_square_on_plane,
    draw_points,
    draw_normals,
    draw_polyline,
)

from simWithPyBullet import BulletRobotSim, find_last_link_index
from datacollection_franky.path_utils import find_fr3_urdf
import pybullet as p


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Run with PyBullet GUI")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Simulation time step")
    parser.add_argument("--duration", type=float, default=5.0, help="Simulation duration in seconds")
    parser.add_argument("--realtime", action="store_true", help="Sleep to approximate real time in GUI")
    parser.add_argument("--pointcloud-mode", action="store_true", help="Use pointcloud-driven raster and normals")
    parser.add_argument("--pcd-path", type=str, default="", help="Point cloud file path for pointcloud mode")
    parser.add_argument("--pcd-frame", type=str, default="fr3_link0", choices=["camera_optical", "fr3_link0"], help="Point cloud frame")
    parser.add_argument("--cloud-vis", action="store_true", help="Show extra pointcloud waypoints and normals in GUI")
    parser.add_argument("--square-side", type=float, default=0.02, help="Square patch side length in meters")
    parser.add_argument("--line-spacing", type=float, default=0.01, help="Spacing between selected-square scan lines")
    parser.add_argument("--pointcloud-downsample-rate", type=int, default=6, help="Per-line point-cloud downsample rate")
    parser.add_argument("--raster-pattern", type=str, default="unidirectional_retract", choices=["boustrophedon", "unidirectional_retract"], help="Selected-square raster pattern")
    parser.add_argument("--search-radius", type=float, default=0.02, help="Normal search radius in meters")
    parser.add_argument("--normal-projection", type=float, default=0.02, help="Normal projection distance in meters")
    parser.add_argument("--min-normal-points", type=int, default=3, help="Minimum points to estimate normal")
    parser.add_argument("--tool-offset", type=float, default=0.0, help="Final tool offset along normal in meters")
    parser.add_argument("--tool-normal-sign", type=float, default=-1.0, help="Normal sign for tool axis alignment")
    parser.add_argument("--cloud-time-scale", type=float, default=0.01, help="Pointcloud trajectory time scale")
    parser.add_argument(
        "--cloud-waypoint-densify",
        type=int,
        default=3,
        help="Interpolated waypoints inserted between neighboring pointcloud waypoints",
    )
    parser.add_argument("--max-ee-pos-error", type=float, default=0.002, help="EE error threshold in meters")
    parser.add_argument("--max-error-streak", type=int, default=80, help="Consecutive error steps before skip-ahead")
    parser.add_argument("--skip-ahead-steps", type=int, default=40, help="How many dt steps to skip after error streak")
    parser.add_argument("--surface-z-lift", type=float, default=0.02, help="Additional world-Z lift above surface waypoints in meters")
    return parser.parse_args()


def _get_ee_pose_for_joint_targets(
    sim: BulletRobotSim,
    ee_link_index: int,
    joint_targets_by_name: dict[str, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Compute EE pose for a hypothetical joint configuration, then restore current state.
    """
    current = sim.get_current_joint_positions()
    for js in sim.joints:
        q = float(joint_targets_by_name.get(js.name, current[js.name]))
        p.resetJointState(sim.robot_id, js.index, targetValue=q, targetVelocity=0.0)
    ee_pos, ee_quat = sim.get_tcp_pose(ee_link_index)
    for js in sim.joints:
        p.resetJointState(sim.robot_id, js.index, targetValue=float(current[js.name]), targetVelocity=0.0)
    return (
        (float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])),
        (float(ee_quat[0]), float(ee_quat[1]), float(ee_quat[2]), float(ee_quat[3])),
    )


def _move_to_joint_pose_via_ik(
    sim: BulletRobotSim,
    ee_link_index: int,
    joint_targets_by_name: dict[str, float],
    move_time_sec: float,
    realtime_sleep: bool,
    stage_name: str,
) -> None:
    target_pos, target_quat = _get_ee_pose_for_joint_targets(
        sim=sim,
        ee_link_index=ee_link_index,
        joint_targets_by_name=joint_targets_by_name,
    )
    ik_cmd = _solve_ik_with_precheck(
        sim=sim,
        ee_link_index=ee_link_index,
        target_pos=target_pos,
        target_quat=target_quat,
        stage_name=stage_name,
    )
    print(
        "[Main] pre-track IK move:",
        f"stage={stage_name},",
        f"target_ee=({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f})",
    )
    sim.goto_joint_pose_slow(
        q_target_by_name=ik_cmd,
        move_time_sec=float(move_time_sec),
        realtime_sleep=realtime_sleep,
    )


def _evaluate_ik_command_error(
    sim: BulletRobotSim,
    ee_link_index: int,
    cmd: dict[str, float],
    target_pos: Tuple[float, float, float],
    target_quat: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """
    Temporarily apply IK command, measure TCP error, then restore current joint state.
    Returns (position_error_m, quaternion_abs_dot).
    """
    current = sim.get_current_joint_positions()
    for js in sim.joints:
        q = float(cmd.get(js.name, current[js.name]))
        p.resetJointState(sim.robot_id, js.index, targetValue=q, targetVelocity=0.0)
    pred_pos, pred_quat = sim.get_tcp_pose(ee_link_index)
    for js in sim.joints:
        p.resetJointState(sim.robot_id, js.index, targetValue=float(current[js.name]), targetVelocity=0.0)
    dx = float(pred_pos[0]) - float(target_pos[0])
    dy = float(pred_pos[1]) - float(target_pos[1])
    dz = float(pred_pos[2]) - float(target_pos[2])
    pos_err = float(math.sqrt(dx * dx + dy * dy + dz * dz))
    quat_abs_dot = _quat_abs_dot(
        (float(pred_quat[0]), float(pred_quat[1]), float(pred_quat[2]), float(pred_quat[3])),
        (float(target_quat[0]), float(target_quat[1]), float(target_quat[2]), float(target_quat[3])),
    )
    return pos_err, quat_abs_dot


def _solve_ik_with_precheck(
    sim: BulletRobotSim,
    ee_link_index: int,
    target_pos: Tuple[float, float, float],
    target_quat: Tuple[float, float, float, float],
    stage_name: str,
) -> dict[str, float]:
    """
    Solve IK from current pose seed and validate FK error before any motion.
    """
    rest = sim.get_current_joint_positions()
    ik_cmd = sim.solve_ik(
        ee_link_index=ee_link_index,
        target_pos=target_pos,
        target_quat_xyzw=target_quat,
        rest_pose_by_name=rest,
    )
    pos_err, quat_abs_dot = _evaluate_ik_command_error(
        sim=sim,
        ee_link_index=ee_link_index,
        cmd=ik_cmd,
        target_pos=target_pos,
        target_quat=target_quat,
    )
    max_pos_err = 0.01
    min_quat_abs_dot = 0.995
    if pos_err > max_pos_err or quat_abs_dot < min_quat_abs_dot:
        raise RuntimeError(
            f"[IK precheck failed] stage={stage_name}, pos_err={pos_err:.5f} m, "
            f"quat_abs_dot={quat_abs_dot:.5f}, "
            f"target=({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f})"
        )
    print(
        "[IK precheck] pass:",
        f"stage={stage_name}, pos_err={pos_err:.5f} m, quat_abs_dot={quat_abs_dot:.5f}",
    )
    return ik_cmd


def _move_to_idle(sim: BulletRobotSim, ee_link_index: int, realtime_sleep: bool) -> None:
    idle = sim.compute_idle_pose_from_urdf()
    _validate_joint_targets_before_motion(
        sim=sim,
        q_target_by_name=idle,
        stage_name="idle",
    )
    sim.goto_joint_pose_slow(
        q_target_by_name=idle,
        move_time_sec=7.0,
        settle_max_steps=4000,
        realtime_sleep=realtime_sleep,
    )


def _build_limit_safe_neutral_pose(sim: BulletRobotSim) -> dict[str, float]:
    """
    Build a neutral-like pose by targeting zero on each actuated joint,
    then clamping to URDF joint limits when zero is out of range.
    """
    neutral: dict[str, float] = {}
    clamped = 0
    for js in sim.joints:
        q = 0.0
        if js.upper > js.lower:
            if q < float(js.lower):
                q = float(js.lower)
                clamped += 1
            if q > float(js.upper):
                q = float(js.upper)
                clamped += 1
        neutral[js.name] = float(q)
    print(
        "[Joint target] neutral build:",
        f"joints={len(neutral)}, clamped={clamped}",
    )
    return neutral


def _validate_joint_targets_before_motion(
    sim: BulletRobotSim,
    q_target_by_name: dict[str, float],
    stage_name: str,
) -> None:
    """
    Validate joint-space target before executing a large joint move.
    Raises if any target is out of limits.
    """
    current = sim.get_current_joint_positions()
    max_delta = 0.0
    for js in sim.joints:
        if js.name not in q_target_by_name:
            continue
        q = float(q_target_by_name[js.name])
        if js.upper > js.lower:
            if q < float(js.lower) or q > float(js.upper):
                raise RuntimeError(
                    f"[Joint precheck failed] stage={stage_name}, joint={js.name}, "
                    f"target={q:.6f}, limits=({float(js.lower):.6f}, {float(js.upper):.6f})"
                )
        dq = abs(q - float(current[js.name]))
        if dq > max_delta:
            max_delta = dq
    print(
        "[Joint precheck] pass:",
        f"stage={stage_name}, max_joint_delta={max_delta:.5f} rad",
    )


def _report_pose_waypoint_stats(pose_waypoints: Sequence[object]) -> None:
    if not pose_waypoints:
        print("[PointCloudMode] No pose waypoints generated.")
        return
    normals = []
    quat_min_dot = 1.0
    for idx in range(len(pose_waypoints)):
        wp = pose_waypoints[idx]
        normals.append(float(wp.normal_xyz[2]))
        if idx > 0:
            q0 = pose_waypoints[idx - 1].quat_xyzw
            q1 = pose_waypoints[idx].quat_xyzw
            dot = abs(float(q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]))
            if dot < quat_min_dot:
                quat_min_dot = dot
    print(f"[PointCloudMode] pose waypoints: {len(pose_waypoints)}")
    print(f"[PointCloudMode] normal z range: min={min(normals):.4f}, max={max(normals):.4f}")
    print(f"[PointCloudMode] min abs quat dot between neighbors: {quat_min_dot:.4f}")


def _apply_world_z_lift(
    pose_waypoints: Sequence[object],
    z_lift_m: float,
    waypoint_cls: type,
) -> List[object]:
    """
    Return a new waypoint list with an added world-frame Z lift.
    """
    lifted: List[object] = []
    for wp in pose_waypoints:
        xyz = wp.xyz
        lifted.append(
            waypoint_cls(
                xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2]) + float(z_lift_m)),
                quat_xyzw=wp.quat_xyzw,
                normal_xyz=wp.normal_xyz,
            )
        )
    return lifted


def _derive_surface_waypoints_from_lifted(
    lifted_waypoints: Sequence[object],
    z_lift_m: float,
) -> List[Tuple[float, float, float]]:
    """
    Recover local surface waypoints from lifted waypoints by undoing the world-Z lift.
    """
    surface_pts: List[Tuple[float, float, float]] = []
    for wp in lifted_waypoints:
        xyz = wp.xyz
        surface_pts.append(
            (
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]) - float(z_lift_m),
            )
        )
    return surface_pts


def _quat_xyzw_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    n = math.sqrt(float(q[0]) * float(q[0]) + float(q[1]) * float(q[1]) + float(q[2]) * float(q[2]) + float(q[3]) * float(q[3]))
    if n <= 1e-12:
        raise RuntimeError("Quaternion normalization failed: near-zero norm.")
    return (float(q[0]) / n, float(q[1]) / n, float(q[2]) / n, float(q[3]) / n)


def _quat_slerp_xyzw(
    q0: Tuple[float, float, float, float],
    q1: Tuple[float, float, float, float],
    alpha: float,
) -> Tuple[float, float, float, float]:
    qa = _quat_xyzw_normalize(q0)
    qb = _quat_xyzw_normalize(q1)
    dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]
    if dot < 0.0:
        qb = (-qb[0], -qb[1], -qb[2], -qb[3])
        dot = -dot
    if dot > 0.9995:
        q = (
            qa[0] + alpha * (qb[0] - qa[0]),
            qa[1] + alpha * (qb[1] - qa[1]),
            qa[2] + alpha * (qb[2] - qa[2]),
            qa[3] + alpha * (qb[3] - qa[3]),
        )
        return _quat_xyzw_normalize(q)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return (
        s0 * qa[0] + s1 * qb[0],
        s0 * qa[1] + s1 * qb[1],
        s0 * qa[2] + s1 * qb[2],
        s0 * qa[3] + s1 * qb[3],
    )


def _densify_pose_waypoints(
    pose_waypoints: Sequence[object],
    waypoint_cls: type,
    interp_count: int,
) -> List[object]:
    if int(interp_count) <= 0 or len(pose_waypoints) < 2:
        return list(pose_waypoints)
    dense: List[object] = []
    for idx in range(len(pose_waypoints) - 1):
        wp0 = pose_waypoints[idx]
        wp1 = pose_waypoints[idx + 1]
        if idx == 0:
            dense.append(wp0)
        for j in range(1, int(interp_count) + 1):
            t = float(j) / float(int(interp_count) + 1)
            x = float(wp0.xyz[0]) + t * (float(wp1.xyz[0]) - float(wp0.xyz[0]))
            y = float(wp0.xyz[1]) + t * (float(wp1.xyz[1]) - float(wp0.xyz[1]))
            z = float(wp0.xyz[2]) + t * (float(wp1.xyz[2]) - float(wp0.xyz[2]))
            q = _quat_slerp_xyzw(wp0.quat_xyzw, wp1.quat_xyzw, t)
            nx = float(wp0.normal_xyz[0]) + t * (float(wp1.normal_xyz[0]) - float(wp0.normal_xyz[0]))
            ny = float(wp0.normal_xyz[1]) + t * (float(wp1.normal_xyz[1]) - float(wp0.normal_xyz[1]))
            nz = float(wp0.normal_xyz[2]) + t * (float(wp1.normal_xyz[2]) - float(wp0.normal_xyz[2]))
            nmag = math.sqrt(nx * nx + ny * ny + nz * nz)
            if nmag > 1e-12:
                nx, ny, nz = nx / nmag, ny / nmag, nz / nmag
            dense.append(
                waypoint_cls(
                    xyz=(x, y, z),
                    quat_xyzw=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
                    normal_xyz=(nx, ny, nz),
                )
            )
        dense.append(wp1)
    print(
        "[PointCloudMode] densified waypoints:",
        f"input={len(pose_waypoints)}, output={len(dense)}, interp_per_segment={int(interp_count)}",
    )
    return dense


def _keep_waypoints_one_keep_one_drop(
    pose_waypoints: Sequence[object],
) -> List[object]:
    if not pose_waypoints:
        return []
    kept: List[object] = []
    for idx in range(len(pose_waypoints)):
        if idx % 2 == 0:
            kept.append(pose_waypoints[idx])
    if not kept:
        kept.append(pose_waypoints[0])
    print(
        "[PointCloudMode] waypoint downsample:",
        f"raw_input={len(pose_waypoints)}, kept_one_drop_one={len(kept)}",
    )
    return kept


def _build_demo_trajectory(
    sim: BulletRobotSim,
    ee_link_index: int,
    dt: float,
    gui: bool,
) -> tuple[IKRasterTrajectory, List[int]]:
    raster_center_xy = (0.35, 0.00)
    raster_z = 0.20
    tool_quat_xyzw = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
    raster_spec = RasterPlanSpec(
        center_xy=raster_center_xy,
        side_len=0.02,
        z=raster_z,
        dx=0.001,
        dy=0.001,
        tool_quat_wxyz=tool_quat_xyzw,
    )
    raster_pts = generate_raster_waypoints(raster_spec)
    vis_ids: List[int] = []
    if gui:
        vis_ids.extend(draw_square_on_plane(raster_center_xy, raster_spec.side_len, raster_z, life_time=0.0, line_width=2.0))
        vis_ids.extend(draw_points(raster_pts, life_time=0.0, point_size=2.0))
    rest = sim.get_current_joint_positions()
    traj = IKRasterTrajectory(
        sim=sim,
        ee_link_index=ee_link_index,
        raster=raster_spec,
        dt=dt,
        time_scale=0.05,
        rest_pose_by_name=rest,
    )
    return traj, vis_ids


def _build_pointcloud_trajectory(
    args: argparse.Namespace,
    sim: BulletRobotSim,
    ee_link_index: int,
) -> tuple[IKPoseSequenceTrajectory, List[int], List[int]]:
    if not args.pcd_path:
        raise RuntimeError("pointcloud mode requires --pcd-path")
    from cloudpoint_wrapper import PointCloudWrapper, RasterSurfaceSpec, PoseWaypoint

    wrapper = PointCloudWrapper(
        pcd_path=args.pcd_path,
        vis_flag=bool(args.gui),
        pcd_frame=str(args.pcd_frame),
    )
    spec = RasterSurfaceSpec(
        square_side_len_m=float(args.square_side),
        line_spacing_m=float(args.line_spacing),
        pointcloud_downsample_rate=int(args.pointcloud_downsample_rate),
        raster_pattern=str(args.raster_pattern),
        search_radius_m=float(args.search_radius),
        # Keep positional waypoints anchored to the sampled surface first.
        # Use a tiny non-zero projection so normal line pairs are valid for downstream normal decoding.
        # World-Z lift is applied later via --surface-z-lift.
        normal_projection_m=1e-6,
        min_points_for_normal=int(args.min_normal_points),
        use_surface_normal=True,
        tool_offset_m=float(args.tool_offset),
        tool_normal_sign=float(args.tool_normal_sign),
    )
    pose_waypoints = wrapper.get_sim_waypoints(spec)
    if not pose_waypoints:
        raise RuntimeError("No pose waypoints were generated from point cloud.")
    pose_waypoints = _apply_world_z_lift(
        pose_waypoints=pose_waypoints,
        z_lift_m=float(args.surface_z_lift),
        waypoint_cls=PoseWaypoint,
    )
    _report_pose_waypoint_stats(pose_waypoints)
    rest = sim.get_current_joint_positions()
    traj = IKPoseSequenceTrajectory(
        sim=sim,
        ee_link_index=ee_link_index,
        pose_waypoints=pose_waypoints,
        dt=float(args.dt),
        time_scale=float(args.cloud_time_scale),
        rest_pose_by_name=rest,
    )
    waypoint_vis_ids: List[int] = []
    normal_vis_ids: List[int] = []
    if args.gui and args.cloud_vis:
        # Visualize the exact path that IK tracks, so start alignment is unambiguous.
        pts = [wp.xyz for wp in pose_waypoints]
        waypoint_vis_ids.extend(draw_points(pts, life_time=0.0, point_size=2.0, color_rgb=(0.2, 0.9, 0.2)))
        waypoint_vis_ids.extend(draw_polyline(pts, life_time=0.0, line_width=1.2))
    return traj, waypoint_vis_ids, normal_vis_ids


def _compute_ee_pos_error(sim: BulletRobotSim, ee_link_index: int, target_xyz: tuple[float, float, float]) -> float:
    ee_pos, _ = sim.get_tcp_pose(ee_link_index)
    dx = float(ee_pos[0]) - float(target_xyz[0])
    dy = float(ee_pos[1]) - float(target_xyz[1])
    dz = float(ee_pos[2]) - float(target_xyz[2])
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def _get_ee_tip_pose(
    sim: BulletRobotSim,
    ee_link_index: int,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    return sim.get_tcp_pose(ee_link_index)


def _quat_abs_dot(q0: Tuple[float, float, float, float], q1: Tuple[float, float, float, float]) -> float:
    return abs(float(q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]))


def _log_tracking_pose_debug(
    sim: BulletRobotSim,
    ee_link_index: int,
    traj: IKPoseSequenceTrajectory,
) -> None:
    ee_pos, ee_quat = _get_ee_tip_pose(sim, ee_link_index)
    target_pos = (
        float(traj.last_target_pos[0]),
        float(traj.last_target_pos[1]),
        float(traj.last_target_pos[2]),
    )
    wp_idx = int(traj.last_index)
    target_quat: Tuple[float, float, float, float] | None = None
    if wp_idx >= 0 and wp_idx < len(traj.pose_waypoints):
        q = traj.pose_waypoints[wp_idx].quat_xyzw
        target_quat = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    dx = ee_pos[0] - target_pos[0]
    dy = ee_pos[1] - target_pos[1]
    dz = ee_pos[2] - target_pos[2]
    pos_err = math.sqrt(dx * dx + dy * dy + dz * dz)
    print(
        "[PointCloudDebug] ee_tip:",
        f"pos=({ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}),",
        f"altitude={ee_pos[2]:.4f},",
        f"quat=({ee_quat[0]:.4f}, {ee_quat[1]:.4f}, {ee_quat[2]:.4f}, {ee_quat[3]:.4f})",
    )
    if target_quat is not None:
        quat_dot = _quat_abs_dot(ee_quat, target_quat)
        print(
            "[PointCloudDebug] target_wp:",
            f"idx={wp_idx},",
            f"pos=({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}),",
            f"altitude={target_pos[2]:.4f},",
            f"quat=({target_quat[0]:.4f}, {target_quat[1]:.4f}, {target_quat[2]:.4f}, {target_quat[3]:.4f}),",
            f"pos_err={pos_err:.4f},",
            f"quat_abs_dot={quat_dot:.4f}",
        )
    else:
        print(
            "[PointCloudDebug] target_wp:",
            f"idx={wp_idx},",
            f"pos=({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}),",
            f"altitude={target_pos[2]:.4f},",
            f"pos_err={pos_err:.4f},",
            "quat=unavailable",
        )


def _move_to_tracking_parking_pose(
    sim: BulletRobotSim,
    ee_link_index: int,
    first_wp: object,
    parking_lift_m: float,
    wait_sec: float,
    realtime_sleep: bool,
) -> None:
    """Move EE to a parking pose directly above first trajectory waypoint."""
    parking_xyz = (
        float(first_wp.xyz[0]),
        float(first_wp.xyz[1]),
        float(first_wp.xyz[2]) + float(parking_lift_m),
    )
    # Keep parking orientation consistent with first tracking waypoint.
    parking_quat = (
        float(first_wp.quat_xyzw[0]),
        float(first_wp.quat_xyzw[1]),
        float(first_wp.quat_xyzw[2]),
        float(first_wp.quat_xyzw[3]),
    )
    parking_cmd = _solve_ik_with_precheck(
        sim=sim,
        ee_link_index=ee_link_index,
        target_pos=parking_xyz,
        target_quat=parking_quat,
        stage_name="parking",
    )
    print(
        "[PointCloudMode] moving to parking pose:",
        f"xyz=({parking_xyz[0]:.4f}, {parking_xyz[1]:.4f}, {parking_xyz[2]:.4f}),",
        "orientation=first-waypoint",
    )
    sim.goto_joint_pose_slow(
        q_target_by_name=parking_cmd,
        move_time_sec=3.0,
        realtime_sleep=realtime_sleep,
    )
    wait_steps = max(1, int(float(wait_sec) / float(sim.time_step)))
    for _ in range(wait_steps):
        sim.step()
        if realtime_sleep and sim.gui:
            time.sleep(sim.time_step)
    print(f"[PointCloudMode] parking hold complete: {float(wait_sec):.3f} s")


def _move_to_first_tracking_pose(
    sim: BulletRobotSim,
    ee_link_index: int,
    first_wp: object,
    realtime_sleep: bool,
) -> None:
    """
    After parking, descend to the first waypoint before starting trajectory time.
    """
    first_xyz = (
        float(first_wp.xyz[0]),
        float(first_wp.xyz[1]),
        float(first_wp.xyz[2]),
    )
    first_quat = (
        float(first_wp.quat_xyzw[0]),
        float(first_wp.quat_xyzw[1]),
        float(first_wp.quat_xyzw[2]),
        float(first_wp.quat_xyzw[3]),
    )
    first_cmd = _solve_ik_with_precheck(
        sim=sim,
        ee_link_index=ee_link_index,
        target_pos=first_xyz,
        target_quat=first_quat,
        stage_name="first_waypoint",
    )
    print(
        "[PointCloudMode] aligning to first tracking waypoint:",
        f"xyz=({first_xyz[0]:.4f}, {first_xyz[1]:.4f}, {first_xyz[2]:.4f})",
    )
    sim.goto_joint_pose_slow(
        q_target_by_name=first_cmd,
        move_time_sec=2.5,
        realtime_sleep=realtime_sleep,
    )


# -----------------------------
# Main entry point
# -----------------------------

def main() -> int:
    """
    Main program: parses command line arguments, starts simulation, loads robot,
    runs a simple trajectory (hold-still demo), and steps the simulation.
    Cleans up on exit.
    """
    args = _parse_args()

    # Resolve the URDF path (assumed to be in same directory as this script)
    here = os.path.dirname(os.path.abspath(__file__))
    urdf_path = find_fr3_urdf(here)

    # Instantiate simulation class with chosen options
    sim = BulletRobotSim(gui=args.gui, time_step=args.dt)

    raster_vis_ids: List[int] = []
    normal_vis_ids: List[int] = []
    try:
        # Establish connection and load robot
        sim.connect()
        sim.load_robot(urdf_path=urdf_path, fixed_base=True)

        # Resolve end-effector link as the last link in the loaded chain.
        ee_link_index = find_last_link_index(sim.robot_id)

        show_ee_frame = True
        ee_frame_ids: List[int] = []
        frame_update_stride = 10  # redraw every 10 sim steps to reduce overhead
        step_count = 0

        print("[Main] skip move_to_idle: robot initialized at idle during load.")
        if args.pointcloud_mode:
            print("[Main] Running in pointcloud mode.")
            traj, raster_vis_ids, normal_vis_ids = _build_pointcloud_trajectory(args, sim, ee_link_index)
            if len(traj.pose_waypoints) == 0:
                raise RuntimeError("Pointcloud trajectory has no waypoints for parking pre-move.")
            _move_to_tracking_parking_pose(
                sim=sim,
                ee_link_index=ee_link_index,
                first_wp=traj.pose_waypoints[0],
                parking_lift_m=0.05,
                wait_sec=0.5,
                realtime_sleep=(args.gui and args.realtime),
            )
            _move_to_first_tracking_pose(
                sim=sim,
                ee_link_index=ee_link_index,
                first_wp=traj.pose_waypoints[0],
                realtime_sleep=(args.gui and args.realtime),
            )
        else:
            print("[Main] Running in demo fixed-plane mode.")
            traj, raster_vis_ids = _build_demo_trajectory(sim, ee_link_index, args.dt, args.gui)


        paused = False
        t = 0.0
        error_streak = 0
        # Main PyBullet simulation loop: listens for keyboard events to control the simulation state.
        # - ESC or 'q' key to quit
        # - Spacebar toggles pause
        # - 'r' resets the simulation time to zero
        # While paused, steps simulation without joint motion (keeps GUI responsive).
        while True:
            keys = p.getKeyboardEvents()
            if ord('q') in keys and (keys[ord('q')] & p.KEY_WAS_TRIGGERED):
                break
            if ord(' ') in keys and (keys[ord(' ')] & p.KEY_WAS_TRIGGERED):
                paused = not paused
            if ord('r') in keys and (keys[ord('r')] & p.KEY_WAS_TRIGGERED):
                t = 0.0
            # Toggle EE frame display with 'f'
            if ord('f') in keys and (keys[ord('f')] & p.KEY_WAS_TRIGGERED):
                show_ee_frame = not show_ee_frame
                remove_debug_items(ee_frame_ids)
                ee_frame_ids.clear()

            if not paused:
                # one time code to print out the command keys and the future joint position
                cmd = traj.sample(t)
                if step_count % 120 == 0:
                    print("cmd keys:", list(cmd.keys())[:3], "..., q2:", cmd.get("fr3_joint2"))
                sim.set_joint_positions(cmd)
                sim.step()
                if args.pointcloud_mode:
                    ee_err = _compute_ee_pos_error(sim, ee_link_index, traj.last_target_pos)
                    if ee_err > float(args.max_ee_pos_error):
                        error_streak += 1
                        if error_streak % 20 == 0:
                            print(
                                "[PointCloudMode] warning: EE tracking error",
                                f"{ee_err:.4f} m, waypoint_idx={traj.last_index}",
                            )
                            _log_tracking_pose_debug(sim, ee_link_index, traj)
                        if error_streak >= int(args.max_error_streak):
                            # Do not fast-forward trajectory time when tracking is poor.
                            # Keep stepping at normal dt to avoid abrupt target jumps.
                            print(
                                "[PointCloudMode] tracking error streak reached limit; continuing without skip-ahead:",
                                f"error_streak={error_streak}, t={t:.4f}",
                            )
                            _log_tracking_pose_debug(sim, ee_link_index, traj)
                            error_streak = 0
                            t += args.dt
                        else:
                            t += args.dt
                    else:
                        error_streak = 0
                        t += args.dt
                else:
                    t += args.dt
            else:
                # Step without applying new motions to keep the GUI interactive
                sim.step()
            step_count += 1
            # Update EE frame
            if args.gui and show_ee_frame and (step_count % frame_update_stride == 0):
                remove_debug_items(ee_frame_ids)
                tcp_pos, tcp_quat = sim.get_tcp_pose(ee_link_index)
                ee_frame_ids = draw_frame_at_pose(tcp_pos, tcp_quat, axis_len=0.08, life_time=0.0)


            if args.gui and args.realtime:
                time.sleep(args.dt)
        return 0  # Normal exit

    except KeyboardInterrupt:
        # Allow graceful exit on Ctrl-C
        return 0
    finally:
        # Clean up persistent debug drawings (GUI)
        if args.gui:
            try:
                remove_debug_items(raster_vis_ids)
                remove_debug_items(normal_vis_ids)
            except Exception:
                pass

        # Always disconnect on exit for cleanup
        sim.disconnect()


if __name__ == "__main__":
    # Script entry point: run main and exit with its return code
    raise SystemExit(main())

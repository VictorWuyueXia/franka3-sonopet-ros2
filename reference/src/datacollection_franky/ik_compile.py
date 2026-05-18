"""ik_compile.py

Offline IK compilation utilities.

Key behavior:
- For each Cartesian waypoint, solve IK in PyBullet.
- Validate IK by forward kinematics error (TCP pose error).
- If validation fails, re-seed once and retry.
- If still fails, abort with a detailed error.

This module depends on simWithPyBullet.BulletRobotSim from your existing codebase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pybullet as p

from .config import IKCheckConfig
from .math_utils import quat_abs_dot
from .prompts import info, warn


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


@dataclass(frozen=True)
class IKFailure(Exception):
    stage: str
    index: int
    target_pos: Vec3
    target_quat: Quat
    pos_err_m: float
    quat_abs_dot_val: float
    message: str

    def __str__(self) -> str:
        return (
            f"IKFailure(stage={self.stage}, index={self.index}, "
            f"pos_err_m={self.pos_err_m:.6f}, quat_abs_dot={self.quat_abs_dot_val:.6f}, "
            f"target_pos={self.target_pos}, target_quat={self.target_quat})\n"
            f"{self.message}"
        )


def _temporary_reset_joints(sim, q_by_name: Dict[str, float]) -> Dict[str, float]:
    """Reset joints to q_by_name (using resetJointState) and return prior joint state."""

    current = sim.get_current_joint_positions()
    for js in sim.joints:
        q = float(q_by_name.get(js.name, current[js.name]))
        p.resetJointState(sim.robot_id, js.index, targetValue=q, targetVelocity=0.0)
    return current


def _restore_joints(sim, q_by_name: Dict[str, float]) -> None:
    for js in sim.joints:
        q = float(q_by_name.get(js.name, 0.0))
        p.resetJointState(sim.robot_id, js.index, targetValue=q, targetVelocity=0.0)


def _fk_tcp_pose_for_joint_dict(sim, ee_link_index: int, q_by_name: Dict[str, float]) -> Tuple[Vec3, Quat]:
    prev = _temporary_reset_joints(sim, q_by_name)
    pos, quat = sim.get_tcp_pose(ee_link_index)
    _restore_joints(sim, prev)
    return (
        (float(pos[0]), float(pos[1]), float(pos[2])),
        (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    )


def _pose_error(sim, ee_link_index: int, q_by_name: Dict[str, float], target_pos: Vec3, target_quat: Quat) -> Tuple[float, float]:
    pred_pos, pred_quat = _fk_tcp_pose_for_joint_dict(sim, ee_link_index, q_by_name)

    dx = float(pred_pos[0]) - float(target_pos[0])
    dy = float(pred_pos[1]) - float(target_pos[1])
    dz = float(pred_pos[2]) - float(target_pos[2])
    pos_err = float((dx * dx + dy * dy + dz * dz) ** 0.5)

    qdot = quat_abs_dot(pred_quat, target_quat)
    return pos_err, qdot


def solve_ik_checked(
    sim,
    ee_link_index: int,
    target_pos: Vec3,
    target_quat: Quat,
    rest_pose_by_name: Dict[str, float],
    ik_check: IKCheckConfig,
    stage: str,
    index: int,
    max_iters: int,
    residual_thresh: float,
) -> Dict[str, float]:
    """Solve IK and validate it by FK error."""

    # Bias PyBullet's internal init by setting the robot joint state to the rest pose.
    prior = _temporary_reset_joints(sim, rest_pose_by_name)
    try:
        q_cmd = sim.solve_ik(
            ee_link_index=ee_link_index,
            target_pos=target_pos,
            target_quat_xyzw=target_quat,
            rest_pose_by_name=rest_pose_by_name,
            max_iters=int(max_iters),
            residual_thresh=float(residual_thresh),
        )
    finally:
        _restore_joints(sim, prior)

    pos_err, qdot = _pose_error(sim, ee_link_index, q_cmd, target_pos, target_quat)

    if pos_err > float(ik_check.max_pos_err_m) or qdot < float(ik_check.min_quat_abs_dot):
        raise IKFailure(
            stage=stage,
            index=int(index),
            target_pos=target_pos,
            target_quat=target_quat,
            pos_err_m=float(pos_err),
            quat_abs_dot_val=float(qdot),
            message=(
                "IK precheck failed. "
                f"thresholds: max_pos_err_m={ik_check.max_pos_err_m}, min_quat_abs_dot={ik_check.min_quat_abs_dot}. "
                "Consider increasing parking lift or reducing max_step_m."
            ),
        )

    return q_cmd


def compile_cartesian_waypoints_to_joint_waypoints(
    sim,
    ee_link_index: int,
    pose_waypoints: Sequence[object],
    initial_rest_pose_by_name: Dict[str, float],
    ik_check: IKCheckConfig,
    stage: str,
) -> List[Dict[str, float]]:
    """Compile a pose waypoint sequence into joint targets.

    Retry policy per waypoint:
    - attempt 1: max_iters=120, residual=5e-5
    - attempt 2: max_iters=240, residual=1e-4

    The second attempt re-seeds from the current rest pose again (your requirement),
    but with more iterations and a slightly looser residual threshold.
    """

    q_prev = dict(initial_rest_pose_by_name)
    out: List[Dict[str, float]] = []
    total = int(len(pose_waypoints))
    stage_t0 = float(time.monotonic())
    info(f"[IK:{stage}] start compile with {total} waypoints")

    for k, wp in enumerate(pose_waypoints):
        wp_t0 = float(time.monotonic())
        target_pos = (float(wp.xyz[0]), float(wp.xyz[1]), float(wp.xyz[2]))
        target_quat = (
            float(wp.quat_xyzw[0]),
            float(wp.quat_xyzw[1]),
            float(wp.quat_xyzw[2]),
            float(wp.quat_xyzw[3]),
        )

        try:
            q_cmd = solve_ik_checked(
                sim=sim,
                ee_link_index=ee_link_index,
                target_pos=target_pos,
                target_quat=target_quat,
                rest_pose_by_name=q_prev,
                ik_check=ik_check,
                stage=stage,
                index=k,
                max_iters=120,
                residual_thresh=5e-5,
            )
        except IKFailure as e1:
            # One retry, re-seeded from the "current" rest pose again.
            try:
                q_cmd = solve_ik_checked(
                    sim=sim,
                    ee_link_index=ee_link_index,
                    target_pos=target_pos,
                    target_quat=target_quat,
                    rest_pose_by_name=q_prev,
                    ik_check=ik_check,
                    stage=stage,
                    index=k,
                    max_iters=240,
                    residual_thresh=1e-4,
                )
            except IKFailure as e2:
                # Raise the second failure, but include the first failure info as context.
                raise IKFailure(
                    stage=e2.stage,
                    index=e2.index,
                    target_pos=e2.target_pos,
                    target_quat=e2.target_quat,
                    pos_err_m=e2.pos_err_m,
                    quat_abs_dot_val=e2.quat_abs_dot_val,
                    message=(
                        "IK failed twice at the same waypoint.\n"
                        f"First attempt: pos_err_m={e1.pos_err_m:.6f}, quat_abs_dot={e1.quat_abs_dot_val:.6f}\n"
                        f"Second attempt: pos_err_m={e2.pos_err_m:.6f}, quat_abs_dot={e2.quat_abs_dot_val:.6f}"
                    ),
                )

        out.append(q_cmd)
        q_prev = dict(q_cmd)
        wp_elapsed_s = float(time.monotonic()) - wp_t0
        if wp_elapsed_s > 1.0:
            warn(f"[IK:{stage}] slow waypoint index={k} elapsed_s={wp_elapsed_s:.3f}")
        if (k % 50) == 0 or k == (total - 1):
            total_elapsed_s = float(time.monotonic()) - stage_t0
            info(f"[IK:{stage}] progress {k + 1}/{total} elapsed_s={total_elapsed_s:.2f}")

    info(f"[IK:{stage}] finished in {float(time.monotonic()) - stage_t0:.2f}s")

    return out

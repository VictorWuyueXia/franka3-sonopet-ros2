"""preview.py

PyBullet preview runner for a precompiled joint waypoint list.

We treat PyBullet as a visualization and sanity check tool.
The preview uses POSITION_CONTROL and advances physics in smaller timesteps.
"""

from __future__ import annotations

import time
from typing import Dict, List

import pybullet as p

from .visualization import update_tcp_frame


def reset_robot_joints(sim, q_by_name: Dict[str, float]) -> None:
    """Hard reset joint states (no dynamics), useful for initializing the sim."""

    for js in sim.joints:
        if js.name not in q_by_name:
            continue
        p.resetJointState(sim.robot_id, js.index, targetValue=float(q_by_name[js.name]), targetVelocity=0.0)


def run_joint_waypoint_preview(
    sim,
    trajectory_points: List[object],
    realtime_sleep: bool,
    sim_time_scale: float,
    ee_link_index: int,
    tcp_frame_interval: int,
) -> None:
    """Run a preview following the joint waypoints.

    Args:
        sim: BulletRobotSim instance.
        trajectory_points: list of trajectory points with q_by_name and dt_s.
        realtime_sleep: if True and GUI is enabled, sleep wall time to match sim time.
        sim_time_scale: realtime speed multiplier for preview sleep pacing.
        ee_link_index: end-effector link index for TCP frame visualization.
        tcp_frame_interval: redraw the live TCP frame every N waypoint steps.
    """

    if not trajectory_points:
        raise ValueError("trajectory_points is empty")
    if float(sim_time_scale) <= 0.0:
        raise ValueError("sim_time_scale must be > 0")

    tcp_frame_ids: List[int] = []

    for step_idx, point in enumerate(trajectory_points):
        q = point.q_by_name
        dt_s = float(point.dt_s)
        if dt_s <= 0.0:
            raise ValueError(f"trajectory_points[{step_idx}].dt_s must be > 0")
        n_sub = max(1, int(round(dt_s / float(sim.time_step))))
        sim.set_joint_positions(q)
        for _ in range(n_sub):
            sim.step()
            if realtime_sleep and sim.gui:
                time.sleep(float(sim.time_step) / float(sim_time_scale))

        # Periodically redraw the live TCP frame (only in GUI mode)
        if sim.gui and (step_idx % int(tcp_frame_interval) == 0):
            tcp_frame_ids = update_tcp_frame(sim, ee_link_index, tcp_frame_ids)

    # Final TCP frame at the last position
    if sim.gui:
        update_tcp_frame(sim, ee_link_index, tcp_frame_ids)

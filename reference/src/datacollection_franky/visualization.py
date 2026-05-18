"""visualization.py

PyBullet debug visualization for the datacollection_franky pipeline.

Draws static markers (square surface, raster key points, idle/parking TCP frames)
and provides a dynamic TCP frame updater for the preview loop.

NOTE: Requires pybullet to be importable (same environment as the rest of the pipeline).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pybullet as p

from .prompts import info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_link_index_by_name(robot_id: int, link_name: str) -> int:
    """Return the PyBullet link index whose link name matches exactly."""
    for joint_index in range(p.getNumJoints(robot_id)):
        current_link_name = p.getJointInfo(robot_id, joint_index)[12].decode("utf-8")
        if current_link_name == str(link_name):
            return int(joint_index)
    raise RuntimeError(f"Link not found: {link_name}")


def _get_link_pose(sim, link_name: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Return world pose for a named link."""
    cache_attr = f"_{str(link_name)}_index_for_visual"
    link_index = getattr(sim, cache_attr, None)
    if link_index is None:
        link_index = _find_link_index_by_name(sim.robot_id, str(link_name))
        setattr(sim, cache_attr, int(link_index))
    link_state = p.getLinkState(sim.robot_id, int(link_index), computeForwardKinematics=True)
    link_pos = link_state[4]
    link_quat = link_state[5]
    return (
        (float(link_pos[0]), float(link_pos[1]), float(link_pos[2])),
        (float(link_quat[0]), float(link_quat[1]), float(link_quat[2]), float(link_quat[3])),
    )

def _find_turning_indices(waypoints_xyz: np.ndarray) -> List[int]:
    """
    Detect indices where the raster path reverses direction.
    A turning point is where consecutive displacement vectors form
    an angle > ~60 degrees (dot-product / norms < 0.5).
    """
    n = waypoints_xyz.shape[0]
    if n < 3:
        return []
    turning: List[int] = []
    for i in range(1, n - 1):
        d_prev = waypoints_xyz[i] - waypoints_xyz[i - 1]
        d_next = waypoints_xyz[i + 1] - waypoints_xyz[i]
        len_prev = float(np.linalg.norm(d_prev))
        len_next = float(np.linalg.norm(d_next))
        if len_prev < 1e-10 or len_next < 1e-10:
            continue
        cos_angle = float(np.dot(d_prev, d_next)) / (len_prev * len_next)
        if cos_angle < 0.5:
            turning.append(i)
    return turning


def _draw_frame_at_pose(
    pos_xyz: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float],
    axis_len: float,
    life_time: float,
) -> List[int]:
    """Draw XYZ frame axes at an arbitrary pose. X=red, Y=green, Z=blue."""
    pos = (float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
    orn = (float(quat_xyzw[0]), float(quat_xyzw[1]),
           float(quat_xyzw[2]), float(quat_xyzw[3]))
    rot = p.getMatrixFromQuaternion(orn)
    axes = [
        ([rot[0], rot[3], rot[6]], [1, 0, 0]),
        ([rot[1], rot[4], rot[7]], [0, 1, 0]),
        ([rot[2], rot[5], rot[8]], [0, 0, 1]),
    ]
    ids: List[int] = []
    for vec, color in axes:
        end = [pos[0] + axis_len * vec[0],
               pos[1] + axis_len * vec[1],
               pos[2] + axis_len * vec[2]]
        ids.append(p.addUserDebugLine(pos, end, color, lineWidth=2, lifeTime=life_time))
    return ids


def _draw_square_on_plane(
    center_xyz: Tuple[float, float, float],
    side_len: float,
    life_time: float,
    line_width: float,
) -> List[int]:
    """Draw a yellow square in the XY plane at the given center's Z height."""
    cx = float(center_xyz[0])
    cy = float(center_xyz[1])
    cz = float(center_xyz[2])
    half = float(side_len) / 2.0
    corners = [
        (cx - half, cy - half, cz),
        (cx + half, cy - half, cz),
        (cx + half, cy + half, cz),
        (cx - half, cy + half, cz),
    ]
    ids: List[int] = []
    for i in range(4):
        ids.append(p.addUserDebugLine(
            corners[i], corners[(i + 1) % 4],
            [1, 1, 0], lineWidth=line_width, lifeTime=life_time,
        ))
    return ids


def _draw_square_from_corners(
    corners_xyz: list,
    life_time: float,
    line_width: float,
) -> List[int]:
    """Draw a square from already surface-aligned 3D corners."""

    if len(corners_xyz) != 4:
        return []
    corners = [
        (
            float(corner[0]),
            float(corner[1]),
            float(corner[2]),
        )
        for corner in corners_xyz
    ]
    ids: List[int] = []
    for idx in range(4):
        ids.append(
            p.addUserDebugLine(
                corners[idx],
                corners[(idx + 1) % 4],
                [1, 1, 0],
                lineWidth=float(line_width),
                lifeTime=float(life_time),
            )
        )
    return ids


def _tcp_pose_at_joints(
    sim,
    ee_link_index: int,
    q_by_name: Dict[str, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Compute the displayed offset-applied Sonopet TCP pose via FK, then restore sim state."""
    cur = sim.get_current_joint_positions()
    for js in sim.joints:
        if js.name in q_by_name:
            p.resetJointState(sim.robot_id, js.index,
                              targetValue=float(q_by_name[js.name]),
                              targetVelocity=0.0)
    tcp_pos, tcp_quat = sim.get_tcp_pose(ee_link_index)
    for js in sim.joints:
        if js.name in cur:
            p.resetJointState(sim.robot_id, js.index,
                              targetValue=float(cur[js.name]),
                              targetVelocity=0.0)
    return tcp_pos, tcp_quat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_raster_key_points(
    pose_waypoints: list,
    point_size: float,
    life_time: float,
) -> List[int]:
    """
    Draw points at start, end, and turning positions of the raster trajectory.
    Start=green, End=red, Turning=magenta.
    """
    if not pose_waypoints:
        return []

    xyz_arr = np.array([wp.xyz for wp in pose_waypoints], dtype=float)
    turning_indices = _find_turning_indices(xyz_arr)
    ids: List[int] = []

    # Start point (green)
    uid = p.addUserDebugPoints(
        [tuple(xyz_arr[0])], [[0, 1, 0]],
        pointSize=point_size, lifeTime=life_time,
    )
    ids.append(uid)

    # End point (red)
    uid = p.addUserDebugPoints(
        [tuple(xyz_arr[-1])], [[1, 0, 0]],
        pointSize=point_size, lifeTime=life_time,
    )
    ids.append(uid)

    # Turning points (magenta)
    if turning_indices:
        turn_pts = [tuple(xyz_arr[i]) for i in turning_indices]
        turn_cols = [[1, 0, 1]] * len(turn_pts)
        uid = p.addUserDebugPoints(
            turn_pts, turn_cols,
            pointSize=point_size, lifeTime=life_time,
        )
        ids.append(uid)

    info(f"Raster key points: 1 start + 1 end + {len(turning_indices)} turning")
    return ids


def draw_raster_curve(
    pose_waypoints: list,
    line_width: float,
    life_time: float,
) -> List[int]:
    """Draw raster trajectory as batched debug points."""
    if not pose_waypoints:
        return []
    xyz_arr = np.array([wp.xyz for wp in pose_waypoints], dtype=float)
    points = [tuple(xyz) for xyz in xyz_arr]
    colors = [[0, 1, 1]] * len(points)
    uid = p.addUserDebugPoints(
        points,
        colors,
        pointSize=max(1.0, float(line_width)),
        lifeTime=float(life_time),
    )
    info(f"Drew raster waypoint points: {len(points)}")
    return [uid]


def draw_pipeline_visuals(
    sim,
    ee_link_index: int,
    pose_waypoints: list,
    q_idle: Dict[str, float],
    parking_lift_m: float,
    square_center_xyz: Tuple[float, float, float],
    square_side_m: float,
    square_corners_xyz=None,
) -> List[int]:
    """
    Draw all static visualization markers for the pipeline.
    Called once after trajectory compilation, before preview.
    Returns collected debug item IDs for potential cleanup.
    """
    all_ids: List[int] = []

    # Square surface patch (yellow). Prefer the PCA tangent-plane corners from point-cloud stage.
    if square_corners_xyz is not None:
        sq_ids = _draw_square_from_corners(
            square_corners_xyz,
            life_time=0.0,
            line_width=2.0,
        )
    else:
        sq_ids = _draw_square_on_plane(
            square_center_xyz, square_side_m,
            life_time=0.0, line_width=2.0,
        )
    all_ids.extend(sq_ids)
    info(f"Drew square surface: center={[round(float(x), 4) for x in square_center_xyz]}, "
         f"side={square_side_m:.4f} m")

    # Raster key points mode (disabled per current request).
    # raster_ids = draw_raster_key_points(
    #     pose_waypoints, point_size=8.0, life_time=0.0,
    # )

    # Raster curve mode (enabled).
    raster_ids = draw_raster_curve(
        pose_waypoints, line_width=1.2, life_time=0.0,
    )
    all_ids.extend(raster_ids)

    # Idle TCP frame (shorter axes to distinguish from live TCP)
    idle_pos, idle_quat = _tcp_pose_at_joints(sim, ee_link_index, q_idle)
    idle_ids = _draw_frame_at_pose(idle_pos, idle_quat, axis_len=0.06, life_time=0.0)
    all_ids.extend(idle_ids)
    info(f"Drew idle TCP frame at {[round(float(x), 4) for x in idle_pos]}")

    # Parking TCP frame (above first raster waypoint)
    first_wp = pose_waypoints[0]
    parking_pos = (
        float(first_wp.xyz[0]),
        float(first_wp.xyz[1]),
        float(first_wp.xyz[2]) + float(parking_lift_m),
    )
    parking_quat = first_wp.quat_xyzw
    park_ids = _draw_frame_at_pose(parking_pos, parking_quat, axis_len=0.06, life_time=0.0)
    all_ids.extend(park_ids)
    info(f"Drew parking TCP frame at {[round(float(x), 4) for x in parking_pos]}")

    return all_ids


def update_tcp_frame(
    sim,
    ee_link_index: int,
    prev_ids: List[int],
) -> List[int]:
    """
    Redraw the live EE TCP frame at the current sim pose.
    Removes previous debug items first, then draws fresh axes.
    """
    for fid in prev_ids:
        p.removeUserDebugItem(fid)
    visual_ids: List[int] = []
    tcp_pos, tcp_quat = sim.get_tcp_pose(ee_link_index)
    visual_ids.extend(_draw_frame_at_pose(tcp_pos, tcp_quat, axis_len=0.08, life_time=0.0))
    link8_pos, link8_quat = _get_link_pose(sim, "fr3_link8")
    visual_ids.extend(_draw_frame_at_pose(link8_pos, link8_quat, axis_len=0.05, life_time=0.0))
    return visual_ids

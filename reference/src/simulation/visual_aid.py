#!/usr/bin/env python3
"""
visual_aid.py
by Victor Xia on 20260218

PyBullet debug drawing helpers: squares, polylines, link frames.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

# Attempt to import pybullet. If unavailable, set p = None.
try:
    import pybullet as p
except ImportError:
    p = None  # type: ignore


def draw_square_on_plane(
    center_xy: Tuple[float, float],
    side_len: float,
    z: float,
    life_time: float = 0.0,
    line_width: float = 2.0,
    ) -> List[int]:
    """
    Draw a 2D square at constant z in the world frame.
    Returns debug line ids.

    Args:
        center_xy: (x, y) tuple representing the center of the square in the plane.
        side_len: Length of the square's side.
        z: Z-height at which to draw the square.
        life_time: Duration (in seconds) for which the debug lines are visible. 0 = persistent.
        line_width: Width of the square's lines.

    Returns:
        List of debug line ids added to the PyBullet debug UI.
    """
    if p is None:
        # If pybullet isn't available, return an empty list.
        return []

    # Unpack the center coordinates and ensure side_len is a float
    cx, cy = center_xy
    L = float(side_len)

    # Compute the corner coordinates of the square
    x0, x1 = cx - 0.5 * L, cx + 0.5 * L
    y0, y1 = cy - 0.5 * L, cy + 0.5 * L

    # Define the square's four corners in 3D (all at z)
    p00 = (x0, y0, z)  # bottom-left
    p10 = (x1, y0, z)  # bottom-right
    p11 = (x1, y1, z)  # top-right
    p01 = (x0, y1, z)  # top-left

    ids = []
    # Draw each side of the square using yellow color [1, 1, 0]
    # Side 1: bottom edge
    ids.append(p.addUserDebugLine(p00, p10, [1, 1, 0], lineWidth=line_width, lifeTime=life_time))
    # Side 2: right edge
    ids.append(p.addUserDebugLine(p10, p11, [1, 1, 0], lineWidth=line_width, lifeTime=life_time))
    # Side 3: top edge
    ids.append(p.addUserDebugLine(p11, p01, [1, 1, 0], lineWidth=line_width, lifeTime=life_time))
    # Side 4: left edge
    ids.append(p.addUserDebugLine(p01, p00, [1, 1, 0], lineWidth=line_width, lifeTime=life_time))
    return ids

def draw_points(
    points_xyz: Iterable[Tuple[float, float, float]],
    life_time: float = 0.0,
    point_size: float = 2.0,
    color_rgb: Tuple[float, float, float] = (0.0, 1.0, 1.0),
    ) -> List[int]:
    if p is None:
        return []
    pts = list(points_xyz)
    if not pts:
        return []
    cols = [list(color_rgb)] * len(pts)
    uid = p.addUserDebugPoints(pts, cols, pointSize=point_size, lifeTime=life_time)
    return [uid]


def draw_normals(
    points_xyz: Iterable[Tuple[float, float, float]],
    normals_xyz: Iterable[Tuple[float, float, float]],
    axis_len: float,
    life_time: float,
    line_width: float,
    color_rgb: Tuple[float, float, float],
    ) -> List[int]:
    """
    Draw normal vectors as line segments from points.
    """
    if p is None:
        return []
    points = list(points_xyz)
    normals = list(normals_xyz)
    if len(points) != len(normals):
        return []
    ids: List[int] = []
    for idx in range(len(points)):
        src = points[idx]
        n = normals[idx]
        end = (
            src[0] + float(axis_len) * float(n[0]),
            src[1] + float(axis_len) * float(n[1]),
            src[2] + float(axis_len) * float(n[2]),
        )
        ids.append(
            p.addUserDebugLine(
                src,
                end,
                [float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2])],
                lineWidth=float(line_width),
                lifeTime=float(life_time),
            )
        )
    return ids


def draw_polyline(
    points_xyz: Iterable[Tuple[float, float, float]],
    life_time: float = 0.0,
    line_width: float = 1.0,
    ) -> List[int]:
    """
    Draw connected line segments between consecutive points.

    Args:
        points_xyz: An iterable of (x, y, z) tuples specifying the polyline vertices.
        life_time: Duration (in seconds) for which the polyline is visible. 0 = persistent.
        line_width: Width of the polyline segments.

    Returns:
        List of debug line ids for each segment created.
    """
    if p is None:
        # If pybullet isn't available, return an empty list.
        return []

    pts = list(points_xyz)  # Materialize the point list, as the input may be a generator

    ids: List[int] = []
    # Iterate over consecutive segments and add a debug line for each
    for i in range(len(pts) - 1):
        # Use cyan color [0, 1, 1] for visibility
        ids.append(p.addUserDebugLine(pts[i], pts[i + 1], [0, 1, 1], lineWidth=line_width, lifeTime=life_time))
    return ids


def draw_frame(
    body_id: int,
    link_index: int,
    axis_len: float = 0.08,
    life_time: float = 0.0,
    ) -> List[int]:
    """
    Draw XYZ axes of the link frame in world coordinates using three colored lines.

    Args:
        body_id: Integer specifying the unique id of the robot body in PyBullet.
        link_index: Index of the link for which to draw the frame.
        axis_len: Length of each axis line to draw (in meters).
        life_time: Duration (in seconds) for which the frame is visible. 0 = persistent.

    Returns:
        List of debug line ids, one for each axis (X/red, Y/green, Z/blue).
    """
    if p is None:
        # If pybullet isn't available, return an empty list.
        return []

    # Get the link's world position and quaternion orientation
    pos, orn = p.getLinkState(body_id, link_index, computeForwardKinematics=True)[:2]
    # Get the 3x3 rotation matrix from the quaternion (row-major order)
    rot = p.getMatrixFromQuaternion(orn)

    # Extract the X, Y, Z axes from rotation matrix columns
    x_axis = [rot[0], rot[3], rot[6]]
    y_axis = [rot[1], rot[4], rot[7]]
    z_axis = [rot[2], rot[5], rot[8]]

    def add_axis(vec: List[float], color: List[float]) -> int:
        """
        Helper to add a debug line from the origin (pos) in direction of vec, colored accordingly.

        Args:
            vec: Direction vector (should be normalized).
            color: RGB color as [r, g, b].

        Returns:
            id of the debug line.
        """
        end = [pos[0] + axis_len * vec[0], pos[1] + axis_len * vec[1], pos[2] + axis_len * vec[2]]
        return p.addUserDebugLine(pos, end, color, lineWidth=2, lifeTime=life_time)

    ids = []
    # Draw X axis in red
    ids.append(add_axis(x_axis, [1, 0, 0]))
    # Draw Y axis in green
    ids.append(add_axis(y_axis, [0, 1, 0]))
    # Draw Z axis in blue
    ids.append(add_axis(z_axis, [0, 0, 1]))
    return ids


def draw_frame_at_pose(
    pos_xyz: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float],
    axis_len: float = 0.08,
    life_time: float = 0.0,
) -> List[int]:
    """
    Draw XYZ axes of an arbitrary pose in world coordinates.
    """
    if p is None:
        return []
    pos = (float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
    orn = (
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
        float(quat_xyzw[3]),
    )
    rot = p.getMatrixFromQuaternion(orn)
    x_axis = [rot[0], rot[3], rot[6]]
    y_axis = [rot[1], rot[4], rot[7]]
    z_axis = [rot[2], rot[5], rot[8]]

    def add_axis(vec: List[float], color: List[float]) -> int:
        end = [pos[0] + axis_len * vec[0], pos[1] + axis_len * vec[1], pos[2] + axis_len * vec[2]]
        return p.addUserDebugLine(pos, end, color, lineWidth=2, lifeTime=life_time)

    ids: List[int] = []
    ids.append(add_axis(x_axis, [1, 0, 0]))
    ids.append(add_axis(y_axis, [0, 1, 0]))
    ids.append(add_axis(z_axis, [0, 0, 1]))
    return ids


def remove_debug_items(ids: List[int]) -> None:
    """
    Remove PyBullet user debug items by id.

    Args:
        ids: List of PyBullet debug item ids to remove.
    """
    if p is None:
        return
    # Remove all debug items with the given ids
    for fid in ids:
        p.removeUserDebugItem(fid)

#!/usr/bin/env python3
"""
trajectories.py
by Victor Xia on 20260218

Trajectory generation types and helpers, independent of actuation backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

class IKSolverSim(Protocol):
    """
    Protocol for a simulator that provides IK and current joint positions (e.g. BulletRobotSim).
    Any simulator implementing this protocol must provide:
      - get_current_joint_positions: returns a dictionary of joint name to position.
      - solve_ik: computes joint positions that achieve a given EE pose, optionally using a rest pose.
    """

    def get_current_joint_positions(self) -> Dict[str, float]: ...
    def solve_ik(
        self,
        ee_link_index: int,
        target_pos: Tuple[float, float, float],
        target_quat_xyzw: Tuple[float, float, float, float],
        rest_pose_by_name: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]: ...


class PoseWaypointLike(Protocol):
    """
    Protocol for point-cloud pose waypoints.
    Required attributes:
      - xyz: target position in world coordinates
      - quat_xyzw: target orientation quaternion (xyzw)
    """

    xyz: Tuple[float, float, float]
    quat_xyzw: Tuple[float, float, float, float]


class TrajectoryProvider:
    """
    Interface/base class for trajectory generation.
    Subclasses should implement sample(t) which returns a dictionary mapping joint names
    to desired joint positions at time t.
    """
    def sample(self, t: float) -> Dict[str, float]:
        raise NotImplementedError

@dataclass
class HoldStillTrajectory(TrajectoryProvider):
    """
    A trajectory provider that always returns the same set of joint positions (holds the robot still).
    Useful for testing: the robot should not move if this provider is used.
    """
    hold_positions: Dict[str, float]  # The fixed joint configuration to hold

    def sample(self, t: float) -> Dict[str, float]:
        # Always return the held configuration regardless of t.
        return self.hold_positions

@dataclass(frozen=True)
class RasterPlanSpec:
    """
    Describes a raster path to be traced on a plane parallel to the XY plane, at constant Z.
    All units are in meters.
      - center_xy: the center position of the raster square (in (x, y))
      - side_len: length of a side of the square raster (meters)
      - z: height above world XY (meters)
      - dx: point resolution (spacing) along the x direction (meters)
      - dy: spacing between raster lines in the y direction (meters)
      - tool_quat_wxyz: quaternion specifying constant tool orientation (xyzw order, e.g. for PyBullet)
    """
    center_xy: Tuple[float, float]
    side_len: float
    z: float
    dx: float
    dy: float
    tool_quat_wxyz: Tuple[float, float, float, float]

def generate_raster_waypoints(spec: RasterPlanSpec) -> List[Tuple[float, float, float]]:
    """
    Generate a zigzag raster path over a square grid at constant Z.
    Returns a list of (x, y, z) tuples in world coordinates.
    The raster alternates directions on each line for efficiency ("snake"-like).
    """
    # Extract and convert relevant parameters from the spec
    cx, cy = spec.center_xy
    L = float(spec.side_len)
    dx = float(spec.dx)
    dy = float(spec.dy)
    z = float(spec.z)

    # Check for parameter validity
    if L <= 0.0:
        raise ValueError("side_len must be > 0.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be > 0.")

    # Compute limits of the square region
    x0 = cx - 0.5 * L              # Left edge x
    x1 = cx + 0.5 * L              # Right edge x
    y0 = cy - 0.5 * L              # Bottom edge y
    y1 = cy + 0.5 * L              # Top edge y

    # Compute the number of raster rows (y direction) and points per line (x direction)
    ny = int(math.floor((y1 - y0) / dy)) + 1
    nx = int(math.floor((x1 - x0) / dx)) + 1

    pts: List[Tuple[float, float, float]] = []
    for iy in range(ny):
        # For each line (constant y), compute the y location
        y = y0 + iy * dy
        if y > y1:
            y = y1  # Clamp last line

        # Decide direction of this line (even = left-to-right, odd = right-to-left for zigzag)
        if iy % 2 == 0:
            xs = [x0 + i * dx for i in range(nx)]
        else:
            xs = [x1 - i * dx for i in range(nx)]

        for x in xs:
            # Clamp x within [x0, x1] bounds (to handle floating point rounding at edges)
            if x < x0:
                x = x0
            if x > x1:
                x = x1
            # Append this (x, y, z) point to the waypoint list (always use specified z)
            pts.append((float(x), float(y), float(z)))

    return pts

@dataclass
class IKRasterTrajectory(TrajectoryProvider):
    """
    Trajectory provider which uses IK to track a raster path with the end effector.
    Solves IK to follow the raster waypoints generated from the given RasterPlanSpec.
    This will only work if the provided sim implements the IKSolverSim protocol.
    """
    sim: IKSolverSim                               # Simulator supporting get_current_joint_positions and solve_ik
    ee_link_index: int                             # End effector link index (for IK)
    raster: RasterPlanSpec                         # The raster pattern to trace
    dt: float                                      # Discretization time step between waypoints (seconds)
    time_scale: float = 1.0                        # 1.0 means one waypoint per dt; <1 slows down; >1 speeds up
    rest_pose_by_name: Optional[Dict[str, float]] = None  # Optional rest pose (starting configuration for IK)

    def __post_init__(self) -> None:
        # Precompute the spatial waypoints for the raster trajectory
        self._pts = generate_raster_waypoints(self.raster)
        # Store the fixed tool orientation specified in spec (should remain constant along the path)
        self._orn = self.raster.tool_quat_wxyz
        # If a rest pose is not provided, default to the robot's current configuration
        if self.rest_pose_by_name is None:
            self.rest_pose_by_name = self.sim.get_current_joint_positions()

    def sample(self, t: float) -> Dict[str, float]:
        """
        For a given time t, select the corresponding raster waypoint and return joint positions to reach it.
        Uses the simulator's IK solver.
        If t exceeds the path duration, returns the final waypoint's joint positions.
        """
        # If there are no path points, just hold current robot pose (safety fallback)
        if not self._pts:
            return self.sim.get_current_joint_positions()

        # Determine the waypoint index to use based on time t, dt, and any time scaling
        k = int((t / self.dt) * float(self.time_scale))
        # Clamp index so it never falls outside the valid waypoint range
        if k < 0:
            k = 0
        if k >= len(self._pts):
            k = len(self._pts) - 1

        # Get the target waypoint
        pos = self._pts[k]
        # Re-seed IK from the current joint state at every sample to keep solving local and continuous.
        current_rest_pose = self.sim.get_current_joint_positions()
        q = self.sim.solve_ik(
            ee_link_index=self.ee_link_index,
            target_pos=pos,
            target_quat_xyzw=self._orn,
            rest_pose_by_name=current_rest_pose,
        )
        # Return the joint command dictionary for this time t
        return q


@dataclass
class IKPoseSequenceTrajectory(TrajectoryProvider):
    """
    Trajectory provider for per-waypoint poses (position + orientation),
    typically produced by point-cloud surface processing.
    """

    sim: IKSolverSim
    ee_link_index: int
    pose_waypoints: Sequence[PoseWaypointLike]
    dt: float
    time_scale: float
    rest_pose_by_name: Optional[Dict[str, float]]

    def __post_init__(self) -> None:
        self._poses = list(self.pose_waypoints)
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0.")
        if self.time_scale <= 0.0:
            raise ValueError("time_scale must be > 0.")
        if self.rest_pose_by_name is None:
            self.rest_pose_by_name = self.sim.get_current_joint_positions()
        self.last_index = 0
        self.last_target_pos = self._poses[0].xyz if self._poses else (0.0, 0.0, 0.0)

    def sample(self, t: float) -> Dict[str, float]:
        """
        Return IK joint command for pose waypoint at time t.
        """
        if not self._poses:
            return self.sim.get_current_joint_positions()
        k = int((t / self.dt) * float(self.time_scale))
        if k < 0:
            k = 0
        if k >= len(self._poses):
            k = len(self._poses) - 1
        wp = self._poses[k]
        self.last_index = k
        self.last_target_pos = wp.xyz
        current_rest_pose = self.sim.get_current_joint_positions()
        return self.sim.solve_ik(
            ee_link_index=self.ee_link_index,
            target_pos=wp.xyz,
            target_quat_xyzw=wp.quat_xyzw,
            rest_pose_by_name=current_rest_pose,
        )

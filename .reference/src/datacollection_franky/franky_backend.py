"""franky_backend.py

Hardware backend using Franky (franky-panda Python package).

This backend executes the precompiled joint waypoint list.

Notes:
- Joint waypoints use position; optional per-waypoint velocity hints (JointState) when provided.

Reference (API examples):
- franky-panda PyPI tutorial includes JointWaypointMotion and JointWaypoint usage.
"""

from __future__ import annotations

import gc
from typing import Dict, List

from .config import FrankyConfig, JOINT_NAMES


class FrankyRobot:
    def __init__(self, robot_ip: str, cfg: FrankyConfig):
        self.robot_ip = str(robot_ip)
        self.cfg = cfg
        self._robot = None

    def try_connect(self) -> bool:
        """Try connecting to the robot. Returns True on success, False otherwise."""

        try:
            import franky
            from franky import Robot
        except Exception as e:
            raise RuntimeError(
                "Failed to import franky. Install with: pip install franky-panda\n"
                f"Import error: {e}"
            ) from e

        try:
            robot = Robot(self.robot_ip)
            self._robot = robot
            # Use a conservative default for OTG on hardware with position-only waypoints.
            self.set_relative_dynamics_factor(float(self.cfg.relative_dynamics_factor))
            # Clear any existing error state.
            try:
                robot.recover_from_errors()
            except Exception:
                # Not fatal; some firmware/configs may not permit.
                pass
            return True
        except Exception:
            self._robot = None
            return False

    @property
    def is_connected(self) -> bool:
        return self._robot is not None

    def current_joint_positions_by_name(self) -> Dict[str, float]:
        if self._robot is None:
            raise RuntimeError("Robot not connected")

        q = list(self._robot.current_joint_positions)
        if len(q) < 7:
            raise RuntimeError(f"Unexpected joint position length: {len(q)}")

        return {JOINT_NAMES[i]: float(q[i]) for i in range(7)}

    def disconnect(self) -> None:
        """Release robot handle and associated runtime threads."""

        if self._robot is None:
            return

        robot = self._robot
        try:
            try:
                if bool(getattr(robot, "is_in_control", False)):
                    try:
                        robot.stop()
                    except Exception:
                        pass
                    try:
                        robot.join_motion(timeout=1.0)
                    except Exception:
                        pass
            except Exception:
                pass
        finally:
            self._robot = None
            del robot
            gc.collect()

    def set_relative_dynamics_factor(self, relative_dynamics_factor: float) -> None:
        """Update Franky OTG scaling for the next executed motion."""

        if self._robot is None:
            raise RuntimeError("Robot not connected")
        self._robot.relative_dynamics_factor = float(relative_dynamics_factor)

    def execute_joint_waypoints(
        self,
        trajectory_points: List[object],
        relative_dynamics_factor: float,
    ) -> None:
        """Execute joint waypoints using JointWaypointMotion.

        Args:
            trajectory_points: unified runtime trajectory points.
            relative_dynamics_factor: OTG scaling for this motion command.
        """

        if self._robot is None:
            raise RuntimeError("Robot not connected")

        try:
            from franky import JointState, JointWaypoint, JointWaypointMotion
        except Exception as e:
            raise RuntimeError(f"Franky API import failed: {e}") from e

        n = len(trajectory_points)
        if n < 2:
            raise ValueError("Need at least 2 trajectory points to execute")

        self.set_relative_dynamics_factor(float(relative_dynamics_factor))

        waypoints = []
        for k in range(n):
            point = trajectory_points[k]
            pos = [float(point.q_by_name[name]) for name in JOINT_NAMES]
            dq_hint = getattr(point, "dq_hint_by_name", None)
            if dq_hint is not None:
                vel = [float(dq_hint[name]) for name in JOINT_NAMES]
                js = JointState(position=pos, velocity=vel)
                waypoints.append(JointWaypoint(js))
            else:
                waypoints.append(JointWaypoint(pos))

        motion = JointWaypointMotion(waypoints)

        # Execute blocking.
        self._robot.move(motion)

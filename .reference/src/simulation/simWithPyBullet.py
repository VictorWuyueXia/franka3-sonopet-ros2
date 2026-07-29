#!/usr/bin/env python3
"""
simWithPyBullet.py
by Victor Xia on 20260218

PyBullet simulation backend module.
- Provides: JointSpec (joint metadata), BulletRobotSim (simulation class), and find_joint_index_by_name (utility).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from datacollection_franky.tcp_config import load_sonopet_tcp_geometry

# Try importing pybullet and its data path library.
# Raises a clear user-facing message if not installed.
try:
    import pybullet as p
    import pybullet_data
except ImportError as e:
    raise SystemExit(
        "pybullet is not installed in this environment.\n"
        "Install it in your sonopet venv with:\n"
        "  pip install pybullet\n"
    ) from e


def _debug_emit(hypothesis_id: str, location: str, message: str, data: Dict[str, object], run_id: str) -> None:
    payload = {
        "sessionId": "dd3fc0",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
    }
    log_path = os.path.join(_repo_root, ".cursor", "debug-dd3fc0.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "a", encoding="utf-8")
    f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    f.close()


def find_joint_index_by_name(body_id: int, joint_name: str) -> int:
    """
    Find the index of a joint in a robot by its name.
    Args:
        body_id: The unique id returned by PyBullet for the loaded robot.
        joint_name: The target joint name (string, must match exactly as in URDF).
    Returns:
        The index (integer) of the joint.
    Raises:
        RuntimeError: If the joint name is not found in the robot.
    """
    for ji in range(p.getNumJoints(body_id)):
        # getJointInfo returns a tuple, [1] is the joint name as bytes, so decode to str.
        name = p.getJointInfo(body_id, ji)[1].decode("utf-8")
        if name == joint_name:
            return ji
    raise RuntimeError(f"Joint not found: {joint_name}")


def find_last_link_index(body_id: int) -> int:
    """
    Return the last link index in the loaded chain.
    If no joints are present, return base link index -1.
    """
    num_joints = p.getNumJoints(body_id)
    if num_joints <= 0:
        return -1
    return int(num_joints - 1)


@dataclass
class JointSpec:
    """
    Metadata for a single robot joint as defined by the URDF.
    This is used to store information about controllable joints.
    
    Attributes:
        name: The name of the joint, as specified in the URDF.
        index: The joint's index in PyBullet.
        lower: Lower joint limit (if fixed, lower == upper).
        upper: Upper joint limit.
        max_force: Maximum force/torque specified for the joint.
        max_velocity: Maximum allowed velocity for the joint.
    """
    name: str
    index: int
    lower: float
    upper: float
    max_force: float
    max_velocity: float


class BulletRobotSim:
    """
    High-level PyBullet robot simulation interface.
    - Handles connecting to the simulator, loading robots, position control, stepping the sim, and getting/setting joint states.
    - All attributes are set on per-instance basis.
    """
    def __init__(self, gui: bool, time_step: float):
        """
        Initialize simulation interface.
        Args:
            gui: Whether to use PyBullet GUI (True) or headless (False).
            time_step: Simulation step time in seconds.
        """
        self.gui = gui
        self.time_step = time_step
        self.client_id: Optional[int] = None  # PyBullet client connection id
        self.robot_id: Optional[int] = None   # PyBullet id of the loaded robot
        self.joints: List[JointSpec] = []     # List of actuated joints found in URDF
        # Controller tuning: raise position gain for better convergence.
        self.position_control_gain = 0.1
        self.velocity_control_gain = 1.0
        self.small_nonzero_inertia = 1e-3
        tcp_geom = load_sonopet_tcp_geometry()
        self.tcp_nominal_pos_in_link8 = tcp_geom.nominal_link8_to_tcp_xyz_m
        self.tcp_nominal_quat_xyzw_in_link8 = tcp_geom.nominal_link8_to_tcp_quat_xyzw
        self.tcp_trim_pos_in_tcp = tcp_geom.tcp_offset_xyz_m
        self.tcp_trim_quat_xyzw_in_tcp = tcp_geom.tcp_offset_quat_xyzw
        self.tcp_offset_pos_in_link = tcp_geom.trimmed_link8_to_tcp_xyz_m
        self.tcp_offset_quat_xyzw_in_link = tcp_geom.trimmed_link8_to_tcp_quat_xyzw

    def connect(self) -> None:
        """
        Connect to the PyBullet simulation.
        - Opens GUI window if self.gui is True, otherwise does not create window.
        - Loads a flat ground plane.
        - Sets simulation parameters such as gravity and time step.
        """
        mode = "GUI" if self.gui else "DIRECT"
        #region agent log
        _debug_emit(
            "H2",
            "simulation/simWithPyBullet.py:BulletRobotSim.connect:before_connect",
            "Calling p.connect with selected PyBullet mode",
            {
                "mode": mode,
                "glx_vendor": os.environ.get("__GLX_VENDOR_LIBRARY_NAME"),
                "display": os.environ.get("DISPLAY"),
                "wayland_display": os.environ.get("WAYLAND_DISPLAY"),
                "xdg_session_type": os.environ.get("XDG_SESSION_TYPE"),
            },
            "pre-fix",
        )
        #endregion
        if self.gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        #region agent log
        _debug_emit(
            "H2",
            "simulation/simWithPyBullet.py:BulletRobotSim.connect:after_connect",
            "Returned from p.connect",
            {
                "mode": mode,
                "client_id": int(self.client_id),
            },
            "pre-fix",
        )
        #endregion
        if self.client_id < 0:
            raise RuntimeError("Failed to connect to PyBullet.")
        p.resetSimulation()
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSubSteps=4,
            numSolverIterations=150
        )
        p.setTimeStep(self.time_step)
        # p.setGravity(0.0, 0.0, -9.81)  # Disable gravity for current debugging phase
        p.setGravity(0.0, 0.0, 0.0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Ensures bundled plane.urdf is found
        p.loadURDF("plane.urdf")
        if self.gui:
            # Sets a suitable camera view for the GUI window.
            p.resetDebugVisualizerCamera(
                cameraDistance=1.6,
                cameraYaw=55,
                cameraPitch=-25,
                cameraTargetPosition=[0.0, 0.0, 0.6],
            )

    def disconnect(self) -> None:
        """Disconnect from PyBullet simulation (shutdown client)."""
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None

    def _urdf_load_hint(self) -> str:
        """
        Returns an informative error string explaining common URDF loading failures and fixes.
        Useful to append to exceptions from loadURDF.
        """
        return (
            "URDF load failed. Common causes:\n"
            "1) Mesh paths in the URDF use package:// which PyBullet does not resolve.\n"
            "2) Mesh files are not present relative to this folder.\n\n"
            "Fix options:\n"
            "- Replace package:// paths with relative paths and keep meshes beside the URDF.\n"
            "- Or generate a resolved URDF with absolute mesh paths.\n"
        )

    def load_robot(self, urdf_path: str, fixed_base: bool = True) -> None:
        """
        Loads a robot model from a URDF file into simulation.
        - Updates the search path for mesh resolution.
        - Will use certain PyBullet URDF flags for inertia and to ignore collisions.
        Args:
            urdf_path: Path to the robot's URDF file on disk.
            fixed_base: If True, the robot's base will not move.
        Raises:
            FileNotFoundError: If the urdf_path does not exist.
            RuntimeError: On PyBullet exceptions, with extra URDF diagnostics.
        """
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        p.setAdditionalSearchPath(urdf_dir)  # Add folder containing URDF to search path
        flags = (
            p.URDF_USE_INERTIA_FROM_FILE
            # | p.URDF_MERGE_FIXED_LINKS
            | p.URDF_IGNORE_COLLISION_SHAPES
        )
        base_pos = [0.0, 0.0, 0.0]
        base_orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        try:
            self.robot_id = p.loadURDF(
                urdf_path,
                basePosition=base_pos,
                baseOrientation=base_orn,
                useFixedBase=fixed_base,
                flags=flags,
            )
        except Exception as e:
            # Attach diagnostic hint about typical errors with mesh paths in URDFs.
            raise RuntimeError(str(e) + "\n\n" + self._urdf_load_hint())
        self._assign_small_inertia_for_zero_inertia_links()
        # After loading, find and store the actuated joints.
        self._introspect_joints()
        self._reset_to_idle_pose_on_load()

    def _assign_small_inertia_for_zero_inertia_links(self) -> None:
        """
        Ensure every loaded link has non-zero local inertia diagonal.
        This stabilizes dynamics when URDF provides zero inertia on helper/tool links.
        """
        assert self.robot_id is not None
        n = p.getNumJoints(self.robot_id)
        all_link_indices = [-1]
        for ji in range(n):
            all_link_indices.append(ji)
        eps = float(self.small_nonzero_inertia)
        updated_count = 0
        for link_idx in all_link_indices:
            dyn = p.getDynamicsInfo(self.robot_id, link_idx)
            inertia_diag = dyn[2]
            ix = float(inertia_diag[0])
            iy = float(inertia_diag[1])
            iz = float(inertia_diag[2])
            if ix <= 0.0 or iy <= 0.0 or iz <= 0.0:
                p.changeDynamics(
                    self.robot_id,
                    link_idx,
                    localInertiaDiagonal=[max(ix, eps), max(iy, eps), max(iz, eps)],
                )
                updated_count += 1
        print(
            "[BulletRobotSim] inertia sanitization:",
            f"updated_links={updated_count}, eps={eps}",
        )

    def _reset_to_idle_pose_on_load(self) -> None:
        """
        Place the robot directly at idle joint pose right after loading.
        """
        assert self.robot_id is not None
        idle_pose = self._compute_idle_pose_from_urdf()
        for js in self.joints:
            q = float(idle_pose.get(js.name, 0.0))
            if js.upper > js.lower:
                if q < float(js.lower):
                    q = float(js.lower)
                if q > float(js.upper):
                    q = float(js.upper)
            p.resetJointState(self.robot_id, js.index, targetValue=q, targetVelocity=0.0)
        print("[BulletRobotSim] initialized robot joint state at idle pose")

    def _link_name_by_index(self, link_index: int) -> str:
        """
        Return link name for the given index.
        Index -1 refers to the base link.
        """
        assert self.robot_id is not None
        if int(link_index) < 0:
            base_info = p.getBodyInfo(self.robot_id)
            return base_info[0].decode("utf-8")
        ji = int(link_index)
        if ji >= p.getNumJoints(self.robot_id):
            raise RuntimeError(f"Invalid link index: {link_index}")
        return p.getJointInfo(self.robot_id, ji)[12].decode("utf-8")

    def _link_has_native_tcp_frame(self, ee_link_index: int) -> bool:
        return self._link_name_by_index(ee_link_index) == "sonopet_tcp"

    def _tcp_target_to_link_target(
        self,
        ee_link_index: int,
        target_tcp_pos: Tuple[float, float, float],
        target_tcp_quat_xyzw: Tuple[float, float, float, float],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """
        Convert desired TCP pose to the pose for IK end-effector link.
        """
        if self._link_has_native_tcp_frame(ee_link_index):
            inv_trim_pos, inv_trim_quat = p.invertTransform(
                self.tcp_trim_pos_in_tcp,
                self.tcp_trim_quat_xyzw_in_tcp,
            )
            link_pos, link_quat = p.multiplyTransforms(
                target_tcp_pos,
                target_tcp_quat_xyzw,
                inv_trim_pos,
                inv_trim_quat,
            )
            return (
                (float(link_pos[0]), float(link_pos[1]), float(link_pos[2])),
                (float(link_quat[0]), float(link_quat[1]), float(link_quat[2]), float(link_quat[3])),
            )
        inv_pos, inv_quat = p.invertTransform(
            self.tcp_offset_pos_in_link,
            self.tcp_offset_quat_xyzw_in_link,
        )
        link_pos, link_quat = p.multiplyTransforms(
            target_tcp_pos,
            target_tcp_quat_xyzw,
            inv_pos,
            inv_quat,
        )
        return (
            (float(link_pos[0]), float(link_pos[1]), float(link_pos[2])),
            (float(link_quat[0]), float(link_quat[1]), float(link_quat[2]), float(link_quat[3])),
        )

    def get_tcp_pose(
        self,
        ee_link_index: int,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """
        Return current TCP pose in world frame.
        If IK link is sonopet_tcp, apply the configured trim after that native frame.
        Otherwise, apply the full nominal-plus-trim TCP offset from the selected link.
        """
        assert self.robot_id is not None
        ee_state = p.getLinkState(self.robot_id, ee_link_index, computeForwardKinematics=True)
        link_pos = ee_state[4]
        link_quat = ee_state[5]
        if self._link_has_native_tcp_frame(ee_link_index):
            tcp_pos, tcp_quat = p.multiplyTransforms(
                link_pos,
                link_quat,
                self.tcp_trim_pos_in_tcp,
                self.tcp_trim_quat_xyzw_in_tcp,
            )
            return (
                (float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2])),
                (float(tcp_quat[0]), float(tcp_quat[1]), float(tcp_quat[2]), float(tcp_quat[3])),
            )
        tcp_pos, tcp_quat = p.multiplyTransforms(
            link_pos,
            link_quat,
            self.tcp_offset_pos_in_link,
            self.tcp_offset_quat_xyzw_in_link,
        )
        return (
            (float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2])),
            (float(tcp_quat[0]), float(tcp_quat[1]), float(tcp_quat[2]), float(tcp_quat[3])),
        )

    def _introspect_joints(self) -> None:
        """
        Scan all joints of the loaded robot and collect actuated joints (revolute/prismatic).
        Stores them in self.joints as JointSpec objects.
        Raises:
            RuntimeError: If no controllable joints are found (bad/incomplete URDF).
        """
        assert self.robot_id is not None
        self.joints.clear()
        n = p.getNumJoints(self.robot_id)
        for ji in range(n):
            info = p.getJointInfo(self.robot_id, ji)
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            # Only actuated (i.e., not fixed/"mimic") joints can be controlled
            if joint_type not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                continue
            lower = float(info[8])
            upper = float(info[9])
            max_force = float(info[10])
            max_vel = float(info[11])
            # If URDF omits max_force, set a safe large default
            if max_force <= 0.0:
                max_force = 150.0
            self.joints.append(
                JointSpec(
                    name=joint_name,
                    index=ji,
                    lower=lower,
                    upper=upper,
                    max_force=max_force,
                    max_velocity=max_vel,
                )
            )
        if not self.joints:
            raise RuntimeError("No actuated joints found in URDF.")

    def get_current_joint_positions(self) -> Dict[str, float]:
        """
        Returns current joint positions (angles or translations) for controllable joints.
        Returns:
            Dictionary mapping joint names to current joint values.
        """
        assert self.robot_id is not None
        cur: Dict[str, float] = {}
        for js in self.joints:
            pos, vel, _, _ = p.getJointState(self.robot_id, js.index)
            cur[js.name] = float(pos)
        return cur

    def set_joint_positions(self, targets_by_name: Dict[str, float], 
        force_scale: float = 1.0,
        force_override: Optional[float] = None
        ) -> None:
        """
        Send joint position command to robot.
        - Will only act on joints in self.joints whose names match those in targets_by_name dict.
        - Clamps commanded positions inside joint limits.
        Args:
            targets_by_name: dict mapping joint names to desired joint positions (float).
        Raises:
            RuntimeError: If no joint names in targets_by_name match the robot.
        """
        assert self.robot_id is not None
        joint_indices: List[int] = []
        target_positions: List[float] = []
        forces: List[float] = []
        for js in self.joints:
            if js.name not in targets_by_name:
                continue
            q_des = float(targets_by_name[js.name])
            # Optional: clamp to joint limit if applicable
            if js.upper > js.lower:
                if q_des < js.lower:
                    q_des = js.lower
                if q_des > js.upper:
                    q_des = js.upper
            joint_indices.append(js.index)
            target_positions.append(float(q_des))
            if force_override is not None:
                f = float(force_override)
            else:
                f = float(js.max_force) * float(force_scale)
            forces.append(f)
        if not joint_indices:
            raise RuntimeError("No joint targets matched joint names in the robot model.")
        # p.setJointMotorControlArray(
        #     bodyUniqueId=self.robot_id,
        #     jointIndices=joint_indices,
        #     controlMode=p.POSITION_CONTROL,
        #     targetPositions=target_positions,
        #     forces=forces,
        # )     # This yeilds oscillation
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            targetVelocities=[0.0] * len(joint_indices),          # adds damping path
            positionGains=[float(self.position_control_gain)] * len(joint_indices),
            velocityGains=[float(self.velocity_control_gain)] * len(joint_indices),
            forces=forces,
        )


    def step(self) -> None:
        """
        Advance the simulation by one step of self.time_step seconds.
        Only changes simulation time; does not update control commands.
        """
        p.stepSimulation()

    def _get_actuated_joint_name_set(self) -> set:
        """
        Returns a set of the names of all actuated (controllable) joints on the robot.
        """
        return {js.name for js in self.joints}

    def _compute_idle_pose_from_urdf(self) -> Dict[str, float]:
        """
        Return a common known safe and stable FR3 "home/ready" pose.
        This is intentionally NOT the midpoint of joint limits.

        Units: radians.
        Note: This assumes actuated joint names fr3_joint1..fr3_joint7 exist.
        Any missing names will be ignored by set_joint_positions (safe).
        """
        return {
            "fr3_joint1": 0.0,
            "fr3_joint2": -0.78539816339,   # -45 deg
            "fr3_joint3": 0.0,
            "fr3_joint4": -2.35619449019,   # -135 deg
            "fr3_joint5": 0.0,
            "fr3_joint6": 1.57079632679,    # 90 deg
            "fr3_joint7": 0,
        }


    def _goto_joint_pose(
        self,
        q_target_by_name: Dict[str, float],
        settle_tol_rad: float = 1e-2,
        settle_max_steps: int = 2000,
        realtime_sleep: bool = False,
        ) -> None:
        """
        Move robot to specific joint positions with feedback/settling loop.
        Args:
            q_target_by_name: dict mapping joint names to command values.
            settle_tol_rad: Maximum error to declare settling successful (in radians/meters).
            settle_max_steps: Maximum simulation steps to settle.
            realtime_sleep: If True, insert real-time sleep for each sim step (for GUI syncing).
        """
        assert self.robot_id is not None
        # Start from current positions
        cur = self.get_current_joint_positions()
        cmd: Dict[str, float] = dict(cur)
        # Update command with requested joints
        for name, val in q_target_by_name.items():
            if name in cmd:
                cmd[name] = float(val)
        # Command loop with error feedback, up to settle_max_steps
        err_max = float("inf")
        for _ in range(int(settle_max_steps)):
            self.set_joint_positions(cmd)
            self.step()
            if realtime_sleep and self.gui:
                time.sleep(self.time_step)
            cur2 = self.get_current_joint_positions()
            # Compute max error (over all joints)
            err_max = max(abs(cur2[name] - qdes) for name, qdes in cmd.items())
            if err_max <= float(settle_tol_rad):
                break
        else:
            raise RuntimeError(
                f"Joint settling failed: max_err={err_max:.6f} rad, "
                f"tol={float(settle_tol_rad):.6f}, steps={int(settle_max_steps)}"
            )

    def goto_joint_pose_slow(
        self,
        q_target_by_name: Dict[str, float],
        move_time_sec: float = 3.0,
        settle_tol_rad: float = 2e-3,
        settle_max_steps: int = 2000,
        realtime_sleep: bool = False,
        ) -> None:
        """
        Move to q_target_by_name with a linear joint space ramp over move_time_sec,
        then run the usual settling loop.

        Uses only PyBullet POSITION_CONTROL defaults (same as set_joint_positions),
        so speed is controlled by the ramp duration, not motor gains.
        """
        if move_time_sec <= 0.0:
            # fall back to the original behavior
            return self._goto_joint_pose(q_target_by_name, settle_tol_rad, settle_max_steps, realtime_sleep)

        # start pose
        q0 = self.get_current_joint_positions()
        # build a full target dict over actuated joints (keep unspecified joints at current)
        q1 = dict(q0)
        for name, val in q_target_by_name.items():
            if name in q1:
                q1[name] = float(val)

        n_steps = max(1, int(move_time_sec / self.time_step))
        for k in range(n_steps):
            a = (k + 1) / float(n_steps)  # 0->1
            cmd = {}
            for name in q0.keys():
                cmd[name] = (1.0 - a) * q0[name] + a * q1[name]
            self.set_joint_positions(cmd)
            self.step()
            if realtime_sleep and self.gui:
                time.sleep(self.time_step)

        # final settle
        self._goto_joint_pose(q1, settle_tol_rad, settle_max_steps, realtime_sleep)


    def _init_idle_from_neutral(
        self,
        idle_pose_by_name: Optional[Dict[str, float]] = None,
        settle_tol_rad: float = 2e-3,
        settle_max_steps: int = 2000,
        realtime_sleep: bool = False,
        ) -> Dict[str, float]:
        """
        Moves robot to an idle configuration, optionally specified.
        - Default idle uses midpoints (from self._compute_idle_pose_from_urdf).
        - Waits until robot 'settles' at target position.
        """
        if idle_pose_by_name is None:
            idle_pose_by_name = self._compute_idle_pose_from_urdf()
        self._goto_joint_pose(
            q_target_by_name=idle_pose_by_name,
            settle_tol_rad=settle_tol_rad,
            settle_max_steps=settle_max_steps,
            realtime_sleep=realtime_sleep,
        )
        return idle_pose_by_name

    def _build_ik_arrays(self) -> Tuple[List[int], List[float], List[float], List[float]]:
        """
        Build IK arrays aligned with PyBullet IK ordering:
        movable joints only (revolute/prismatic), in increasing joint index.

        Returns:
            movable_joint_indices: [jointIndex0, jointIndex1, ...]
            lower, upper, rng: arrays aligned with movable_joint_indices
        """
        assert self.robot_id is not None
        n = p.getNumJoints(self.robot_id)

        movable_joint_indices: List[int] = []
        for ji in range(n):
            jtype = p.getJointInfo(self.robot_id, ji)[2]
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                movable_joint_indices.append(ji)

        spec_by_index = {js.index: js for js in self.joints}

        lower: List[float] = []
        upper: List[float] = []
        rng: List[float] = []
        for ji in movable_joint_indices:
            js = spec_by_index.get(ji, None)
            if js is not None and js.upper > js.lower:
                lo = float(js.lower)
                up = float(js.upper)
                lower.append(lo)
                upper.append(up)
                rng.append(up - lo)
            else:
                # If limits missing, pass zeros for that joint
                lower.append(0.0)
                upper.append(0.0)
                rng.append(0.0)

        return movable_joint_indices, lower, upper, rng

    def _solve_ik(
        self,
        ee_link_index: int,
        target_pos: Tuple[float, float, float],
        target_quat_xyzw: Tuple[float, float, float, float],
        rest_pose_by_name: Optional[Dict[str, float]] = None,
        max_iters: int = 120,
        residual_thresh: float = 5e-5,
        ) -> Dict[str, float]:
        """
        Solve inverse kinematics and return joint command dictionary.
        Args:
            ee_link_index: Index of end effector link (as seen by PyBullet).
            target_pos: World position [x, y, z] to move TCP to.
            target_quat_xyzw: Target orientation [x, y, z, w] quaternion.
            rest_pose_by_name: Dict of preferred joint rest positions (optional).
            max_iters: Max number of IK iterations to allow.
            residual_thresh: Final error threshold for IK convergence.
        Returns:
            Dictionary mapping joint names to calculated values.
        """
        assert self.robot_id is not None
        if rest_pose_by_name is None:
            # If not given, use current positions as rest bias
            rest_pose_by_name = self.get_current_joint_positions()
        # IK arrays aligned to PyBullet's movable-joint ordering
        movable_joint_indices, lower, upper, rng = self._build_ik_arrays()

        # Rest poses aligned to movable joints ordering
        rest_pose_by_index = {js.index: float(rest_pose_by_name.get(js.name, p.getJointState(self.robot_id, js.index)[0]))
                            for js in self.joints}

        rest: List[float] = []
        for ji in movable_joint_indices:
            rest.append(float(rest_pose_by_index.get(ji, p.getJointState(self.robot_id, ji)[0])))

        target_link_pos, target_link_quat = self._tcp_target_to_link_target(
            ee_link_index=ee_link_index,
            target_tcp_pos=target_pos,
            target_tcp_quat_xyzw=target_quat_xyzw,
        )

        q_all = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=int(ee_link_index),
            targetPosition=[float(target_link_pos[0]), float(target_link_pos[1]), float(target_link_pos[2])],
            targetOrientation=[float(target_link_quat[0]), float(target_link_quat[1]), float(target_link_quat[2]), float(target_link_quat[3])],
            lowerLimits=lower,
            upperLimits=upper,
            jointRanges=rng,
            restPoses=rest,
            maxNumIterations=int(max_iters),
            residualThreshold=float(residual_thresh),
        )

        # Map IK solution back to actuated joints using jointIndex->solutionIndex mapping
        sol_index_by_joint_index = {ji: k for k, ji in enumerate(movable_joint_indices)}

        cmd: Dict[str, float] = {}
        for js in self.joints:
            k = sol_index_by_joint_index.get(js.index)
            if k is None:
                continue
            if k < 0 or k >= len(q_all):
                continue
            cmd[js.name] = float(q_all[k])

        # Safety: if IK returned nothing usable, hold current pose so motors still run
        if not cmd:
            return self.get_current_joint_positions()

        return cmd



    @staticmethod
    def install_bullet_robot_sim_extensions() -> None:
        """
        Install the public API methods as attributes of BulletRobotSim.
        - This attaches wrapper functions with "canonical" naming for outside use.
        """
        BulletRobotSim.get_actuated_joint_name_set = _brs_get_actuated_joint_name_set
        BulletRobotSim.compute_idle_pose_from_urdf = _brs_compute_idle_pose_from_urdf
        BulletRobotSim.goto_joint_pose = _brs_goto_joint_pose
        BulletRobotSim.init_idle_from_neutral = _brs_init_idle_from_neutral
        # BulletRobotSim.build_ik_arrays = _brs__build_ik_arrays
        BulletRobotSim.solve_ik = _brs_solve_ik


# === Extension functions for BulletRobotSim (attached as public API) ===

def _brs_get_actuated_joint_name_set(self: BulletRobotSim) -> set:
    """
    Wrapper for BulletRobotSim._get_actuated_joint_name_set, for public API.
    Returns: set of actuated joint names.
    """
    return self._get_actuated_joint_name_set()


def _brs_compute_idle_pose_from_urdf(self: BulletRobotSim) -> Dict[str, float]:
    """
    Wrapper for BulletRobotSim._compute_idle_pose_from_urdf, for public API.
    Returns: dict of idle pose joint positions.
    """
    return self._compute_idle_pose_from_urdf()


def _brs_goto_joint_pose(
    self: BulletRobotSim,
    q_target_by_name: Dict[str, float],
    settle_tol_rad: float = 2e-3,
    settle_max_steps: int = 2000,
    realtime_sleep: bool = False,
) -> None:
    """
    Wrapper for BulletRobotSim._goto_joint_pose, for public API.
    Moves robot to desired joint pose until settled.
    """
    return self._goto_joint_pose(q_target_by_name, settle_tol_rad, settle_max_steps, realtime_sleep)


def _brs_init_idle_from_neutral(
    self: BulletRobotSim,
    idle_pose_by_name: Optional[Dict[str, float]] = None,
    settle_tol_rad: float = 2e-3,
    settle_max_steps: int = 2000,
    realtime_sleep: bool = False,
) -> Dict[str, float]:
    """
    Wrapper for BulletRobotSim._init_idle_from_neutral, for public API.
    Moves robot to idle pose and returns it.
    """
    return self._init_idle_from_neutral(idle_pose_by_name, settle_tol_rad, settle_max_steps, realtime_sleep)


def _brs__build_ik_arrays(self: BulletRobotSim) -> Tuple[List[float], List[float], List[float]]:
    """
    Wrapper for BulletRobotSim._build_ik_arrays, for public API.
    Returns IK arrays required by PyBullet (lower, upper, range).
    """
    return self._build_ik_arrays()


def _brs_solve_ik(
    self: BulletRobotSim,
    ee_link_index: int,
    target_pos: Tuple[float, float, float],
    target_quat_xyzw: Tuple[float, float, float, float],
    rest_pose_by_name: Optional[Dict[str, float]] = None,
    max_iters: int = 120,
    residual_thresh: float = 5e-5,
) -> Dict[str, float]:
    """
    Wrapper for BulletRobotSim._solve_ik, for public API.
    Solves IK for a pose and returns dict of joint positions.
    """
    return self._solve_ik(ee_link_index, target_pos, target_quat_xyzw, rest_pose_by_name, max_iters, residual_thresh)


# On module load: install public API wrappers for canonical method names
BulletRobotSim.install_bullet_robot_sim_extensions()

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from datacollection_franky.path_utils import find_fr3_urdf, find_simulation_dir
from datacollection_franky.pipeline_runtime import run_probe_robot_subprocess
from datacollection_franky.preview import reset_robot_joints
from datacollection_franky.sim_runtime import find_link_index_by_name, validate_joint_dict

from .constants import BASE_FRAME_NAME, ROS_TF_PROBE_PATH, SYSTEM_PYTHON, TCP_FRAME_NAME
from .pointcloud.transforms import TransformSpec, build_transform_spec, compose_transform_specs, invert_transform_spec
from .ros_env import build_franka_tf_helper_command, build_source_chain_command, discover_ros_setup_scripts
from .stream import log_status

FK_LINK_FRAME_NAME = "fr3_link8"


@dataclass(frozen=True)
class RobotCapturePose:
    """Robot TCP pose captured at the same moment as the point-cloud snapshot."""

    robot_ip: str
    q_current: dict
    base_to_tcp: TransformSpec


class LocalRobotLinkFkResolver:
    """Resolve one robot link pose from live joint values using local PyBullet FK."""

    def __init__(self, link_name: str) -> None:
        self.link_name = str(link_name)
        self._sim = None
        self._pybullet = None
        self._link_index = None

    def close(self) -> None:
        if self._sim is None:
            return
        self._sim.disconnect()
        self._sim = None
        self._pybullet = None
        self._link_index = None

    def capture_base_to_link(self, q_by_name: dict) -> TransformSpec | None:
        if not q_by_name:
            return None
        try:
            self._ensure_backend_ready()
            validate_joint_dict(self._sim, q_by_name, name="capture_q_current")
            reset_robot_joints(self._sim, q_by_name)
        except Exception as exc:
            log_status(f"{self.link_name}: local FK unavailable ({exc})")
            return None
   

        link_state = self._pybullet.getLinkState(self._sim.robot_id, int(self._link_index), computeForwardKinematics=True)
        link_pos = link_state[4]
        link_quat = link_state[5]
        return build_transform_spec(
            parent_frame=BASE_FRAME_NAME,
            child_frame=str(self.link_name),
            translation_xyz=link_pos,
            quaternion_xyzw=link_quat,
        )

    def _ensure_backend_ready(self) -> None:
        if self._sim is not None:
            return
        self._build_backend()

    def _build_backend(self) -> None:
        from simulation.simWithPyBullet import BulletRobotSim
        import pybullet as p

        module_dir = Path(__file__).resolve().parent
        sim_dir = find_simulation_dir(str(module_dir), 6)
        if sim_dir is None:
            raise RuntimeError("Could not locate simulation directory for local FK")
        urdf_path = find_fr3_urdf(str(sim_dir))
        sim = BulletRobotSim(gui=False, time_step=1.0 / 240.0)
        sim.connect()
        sim.load_robot(str(urdf_path), fixed_base=True)
        self._sim = sim
        self._pybullet = p
        self._link_index = int(find_link_index_by_name(sim.robot_id, str(self.link_name)))
        log_status(f"{self.link_name}: local FK backend ready with {urdf_path}")


class FrankaTfHelperThread:
    """Background helper that launches the minimal Franka TF publishing chain."""

    def __init__(self, robot_ip: str | None) -> None:
        self.robot_ip = str(robot_ip) if robot_ip else None
        self._thread = None
        self._process = None
        self._lock = threading.Lock()
        self._stop_requested = False
        self._launch_attempted = False

    def ensure_started(self) -> bool:
        if not self.robot_ip:
            return False
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return True
            if self._thread is not None and self._thread.is_alive():
                return True
            if self._launch_attempted:
                return False
            self._launch_attempted = True
            self._stop_requested = False
            self._thread = threading.Thread(target=self._run, name="franka_tf_helper", daemon=True)
            self._thread.start()
        log_status("Started minimal Franka TF helper thread.")
        return True

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._process is not None and self._process.poll() is None)

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True
            process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
        with self._lock:
            self._process = None

    def _run(self) -> None:
        setup_paths = discover_ros_setup_scripts()
        if not setup_paths:
            log_status("Franka TF helper could not start because ROS setup scripts were not found.")
            return
        command = build_franka_tf_helper_command(str(self.robot_ip), setup_paths)
        log_status(f"Launching minimal Franka TF helper with robot_ip={self.robot_ip}")
        process = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
        with self._lock:
            self._process = process
        returncode = int(process.wait())
        with self._lock:
            self._process = None
            if not self._stop_requested:
                self._launch_attempted = False
        if not self._stop_requested:
            log_status(f"Franka TF helper exited with returncode={returncode}")


class RobotTcpPoseResolver:
    """Resolve live base->parent-frame pose for eye-in-hand stitching via ROS TF."""

    def __init__(self, robot_ip: str | None) -> None:
        self.robot_ip = str(robot_ip) if robot_ip else None
        self.tf_helper = FrankaTfHelperThread(self.robot_ip)
        self.fk_link_resolver = LocalRobotLinkFkResolver(FK_LINK_FRAME_NAME)
        self.link8_to_tcp = None

    def capture_base_to_frame(self, frame_name: str) -> RobotCapturePose | None:
        q_by_name = self._probe_joint_state_if_available()
        base_to_frame = self._lookup_base_to_frame_transform(str(frame_name), q_by_name)
        if base_to_frame is None:
            return None
        return RobotCapturePose(
            robot_ip=str(self.robot_ip or ""),
            q_current=q_by_name,
            base_to_tcp=base_to_frame,
        )

    def close(self) -> None:
        self.tf_helper.stop()
        self.fk_link_resolver.close()

    def _lookup_base_to_frame_transform(self, frame_name: str, q_by_name: dict) -> TransformSpec | None:
        fk_transform = self._lookup_base_to_frame_transform_from_fk(str(frame_name), q_by_name)
        if fk_transform is not None:
            return fk_transform
        return self._lookup_base_to_frame_transform_from_tf(str(frame_name))

    def _lookup_base_to_frame_transform_from_fk(self, frame_name: str, q_by_name: dict) -> TransformSpec | None:
        if str(frame_name) != TCP_FRAME_NAME:
            return None
      
        base_to_link8 = self.fk_link_resolver.capture_base_to_link(q_by_name)
        if base_to_link8 is None:
            return None
        link8_to_tcp = self._resolve_link8_to_tcp_transform()
        if link8_to_tcp is None:
            return None
        log_status(f"{frame_name}: using local FK from live q_current")
        return compose_transform_specs(base_to_link8, link8_to_tcp)

    def _resolve_link8_to_tcp_transform(self) -> TransformSpec | None:
        if self.link8_to_tcp is not None:
            return self.link8_to_tcp
        base_to_link8 = self._lookup_base_to_frame_transform_from_tf(FK_LINK_FRAME_NAME)
        if base_to_link8 is None:
            return None
        base_to_tcp = self._lookup_base_to_frame_transform_from_tf(TCP_FRAME_NAME)
        if base_to_tcp is None:
            return None
        self.link8_to_tcp = compose_transform_specs(invert_transform_spec(base_to_link8), base_to_tcp)
        log_status(f"{TCP_FRAME_NAME}: cached {FK_LINK_FRAME_NAME} -> {TCP_FRAME_NAME} transform from ROS TF")
        return self.link8_to_tcp

    def _lookup_base_to_frame_transform_from_tf(self, frame_name: str) -> TransformSpec | None:
        tf_payload = self._run_ros_tf_probe(str(frame_name))
        if not bool(tf_payload.get("ok")) and self._should_launch_tf_helper(tf_payload):
            if self.tf_helper.ensure_started():
                tf_payload = self._retry_ros_tf_probe_after_helper(str(frame_name))
        if not bool(tf_payload.get("ok")):
            self._log_probe_failure(frame_name, tf_payload)
            return None
        return self._build_transform_from_payload(frame_name, tf_payload)

    def _log_probe_failure(self, frame_name: str, tf_payload: dict) -> None:
        log_status(
            f"{frame_name}: failed to read ROS TF {BASE_FRAME_NAME} -> {frame_name} "
            f"({tf_payload.get('error')})"
        )
        self._log_ros_graph_probe_summary(frame_name, tf_payload)

    def _build_transform_from_payload(self, frame_name: str, tf_payload: dict) -> TransformSpec | None:
        translation_xyz = tf_payload.get("translation_xyz")
        quaternion_xyzw = tf_payload.get("quaternion_xyzw")
        if not isinstance(translation_xyz, list) or not isinstance(quaternion_xyzw, list):
            log_status(f"{frame_name}: ROS TF probe returned malformed payload")
            return None
        return build_transform_spec(
            parent_frame=BASE_FRAME_NAME,
            child_frame=str(frame_name),
            translation_xyz=translation_xyz,
            quaternion_xyzw=quaternion_xyzw,
        )

    def _run_ros_tf_probe(self, frame_name: str) -> dict:
        setup_paths = discover_ros_setup_scripts()
        if not setup_paths:
            return {
                "ok": False,
                "error": "No ROS setup.bash scripts were found for TF probe.",
            }
        source_chain = build_source_chain_command(setup_paths)
        command = (
            f"{source_chain} && "
            f'"{SYSTEM_PYTHON}" "{ROS_TF_PROBE_PATH}" '
            f'--base-frame "{BASE_FRAME_NAME}" '
            f'--target-frame "{frame_name}" '
            f'--timeout-sec "3.0"'
        )
        result = subprocess.run(
            ["/bin/bash", "-lc", command],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        return self._parse_probe_result(result)

    def _parse_probe_result(self, result: subprocess.CompletedProcess) -> dict:
        stdout = str(result.stdout or "").strip()
        if int(result.returncode) != 0:
            return {
                "ok": False,
                "error": f"probe subprocess failed: returncode={int(result.returncode)} stderr={str(result.stderr or '').strip()}",
            }
        if not stdout:
            return {"ok": False, "error": "probe subprocess returned empty stdout"}
        try:
            return json.loads(stdout)
        except Exception as exc:
            return {"ok": False, "error": f"failed to parse probe stdout: {exc}; stdout={stdout}"}

    def _retry_ros_tf_probe_after_helper(self, frame_name: str) -> dict:
        log_status(f"{frame_name}: waiting for minimal Franka TF helper to publish ROS TF")
        for _ in range(6):
            time.sleep(1.0)
            tf_payload = self._run_ros_tf_probe(str(frame_name))
            if bool(tf_payload.get("ok")):
                return tf_payload
            if not self._should_keep_waiting_for_helper():
                return tf_payload
        return self._run_ros_tf_probe(str(frame_name))

    def _should_launch_tf_helper(self, tf_payload: dict) -> bool:
        if not self.robot_ip:
            return False
        publishers = tf_payload.get("required_topic_publishers")
        if not isinstance(publishers, dict):
            return False
        tf_publishers = int(publishers.get("/tf", 0))
        tf_static_publishers = int(publishers.get("/tf_static", 0))
        joint_state_publishers = int(publishers.get("/joint_states", 0))
        if tf_publishers <= 0 or tf_static_publishers <= 0 or joint_state_publishers <= 0:
            log_status("ROS graph is missing TF or joint-state publishers; starting minimal Franka TF helper.")
            return True
        error_text = str(tf_payload.get("error") or "").lower()
        if "unconnected" in error_text or "not part of the same tree" in error_text:
            log_status("ROS TF tree is disconnected; starting minimal Franka TF helper.")
            return True
        return False

    def _should_keep_waiting_for_helper(self) -> bool:
        return self.tf_helper.is_running()

    def _log_ros_graph_probe_summary(self, frame_name: str, tf_payload: dict) -> None:
        required = tf_payload.get("required_topics_visible")
        publishers = tf_payload.get("required_topic_publishers")
        if isinstance(required, dict):
            log_status(
                f"{frame_name}: ROS topic visibility "
                f"/tf={bool(required.get('/tf'))}, "
                f"/tf_static={bool(required.get('/tf_static'))}, "
                f"/joint_states={bool(required.get('/joint_states'))}"
            )
        if isinstance(publishers, dict):
            log_status(
                f"{frame_name}: ROS publisher counts "
                f"/tf={int(publishers.get('/tf', 0))}, "
                f"/tf_static={int(publishers.get('/tf_static', 0))}, "
                f"/joint_states={int(publishers.get('/joint_states', 0))}"
            )
        visible_nodes = tf_payload.get("visible_nodes")
        if isinstance(visible_nodes, list) and visible_nodes:
            preview = ", ".join(str(name) for name in visible_nodes[:8])
            suffix = " ..." if len(visible_nodes) > 8 else ""
            log_status(f"{frame_name}: ROS visible nodes: {preview}{suffix}")

    def _probe_joint_state_if_available(self) -> dict:
        if not self.robot_ip:
            return {}
        try:
            payload = run_probe_robot_subprocess(self.robot_ip, 80)
        except Exception as exc:
            log_status(f"robot probe: failed to read current joint state ({exc})")
            return {}
        if not bool(payload.get("connected")):
            error = payload.get("error")
            log_status(f"robot probe: connection failed while reading joint state ({error})")
            return {}
        q_current = payload.get("q_current")
        if not isinstance(q_current, dict):
            return {}
        return {str(name): float(value) for name, value in q_current.items()}

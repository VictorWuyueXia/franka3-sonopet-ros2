"""Simulation runtime helpers kept separate from pipeline orchestration."""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import pybullet as p

from .config import JOINT_NAMES
from .prompts import info, warn


def find_link_index_by_name(robot_id: int, link_name: str) -> int:
    """Return the PyBullet link index whose link name matches exactly."""

    for joint_index in range(p.getNumJoints(robot_id)):
        current_link_name = p.getJointInfo(robot_id, joint_index)[12].decode("utf-8")
        if current_link_name == str(link_name):
            return int(joint_index)
    raise RuntimeError(f"Link not found: {link_name}")


def select_ee_link_index(robot_id: int, find_joint_index_by_name, find_last_link_index) -> int:
    """Prefer native Sonopet TCP link and fall back to terminal link."""

    try:
        ee_link_index = int(find_link_index_by_name(robot_id, "sonopet_tcp"))
        info(f"Using ee_link_index={ee_link_index} (link=sonopet_tcp)")
        return ee_link_index
    except RuntimeError:
        ee_link_index = int(find_last_link_index(robot_id))
        info(f"Using ee_link_index={ee_link_index} (fallback=last_link)")
        return ee_link_index


def validate_joint_dict(sim, q: Dict[str, float], name: str) -> None:
    """Validate a joint dictionary against robot joint names and limits."""

    spec = {js.name: js for js in sim.joints}
    for jn in JOINT_NAMES:
        if jn not in q:
            raise RuntimeError(f"{name}: missing joint {jn}")
        if jn not in spec:
            raise RuntimeError(f"{name}: joint {jn} not found in URDF")
        js = spec[jn]
        if js.upper > js.lower:
            qv = float(q[jn])
            if qv < float(js.lower) or qv > float(js.upper):
                raise RuntimeError(
                    f"{name}: joint {jn} target {qv:.6f} out of limits ({float(js.lower):.6f}, {float(js.upper):.6f})"
                )


def prepare_open3d_picker_environment(square_center_override) -> Optional[str]:
    """Temporarily clear GLX vendor override for Open3D picking if needed."""

    if square_center_override is not None:
        return None
    glx_vendor = os.environ.get("__GLX_VENDOR_LIBRARY_NAME")
    if glx_vendor is None:
        return None
    if str(glx_vendor).strip() == "":
        return None
    warn(
        "Detected forced GLX vendor override. "
        "Temporarily clearing __GLX_VENDOR_LIBRARY_NAME for Open3D point-picking window."
    )
    os.environ.pop("__GLX_VENDOR_LIBRARY_NAME")
    return str(glx_vendor)


def restore_open3d_picker_environment(glx_vendor_backup: Optional[str]) -> None:
    """Restore GLX vendor override after Open3D picking stage."""

    if glx_vendor_backup is None:
        return
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = str(glx_vendor_backup)


def gl_env_snapshot(source_env: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Capture relevant OpenGL-related environment keys."""

    return {
        "display": source_env.get("DISPLAY"),
        "wayland_display": source_env.get("WAYLAND_DISPLAY"),
        "xdg_session_type": source_env.get("XDG_SESSION_TYPE"),
        "__GLX_VENDOR_LIBRARY_NAME": source_env.get("__GLX_VENDOR_LIBRARY_NAME"),
        "__NV_PRIME_RENDER_OFFLOAD": source_env.get("__NV_PRIME_RENDER_OFFLOAD"),
        "DRI_PRIME": source_env.get("DRI_PRIME"),
        "LIBGL_ALWAYS_SOFTWARE": source_env.get("LIBGL_ALWAYS_SOFTWARE"),
        "MESA_LOADER_DRIVER_OVERRIDE": source_env.get("MESA_LOADER_DRIVER_OVERRIDE"),
    }


def build_pybullet_gui_env_overrides(prefer_gpu_renderer: bool) -> Dict[str, Optional[str]]:
    """Build env overrides for optional PyBullet GUI GPU preference."""

    if not prefer_gpu_renderer:
        return {}
    return {
        "__NV_PRIME_RENDER_OFFLOAD": "1",
        "DRI_PRIME": "1",
        "LIBGL_ALWAYS_SOFTWARE": None,
        "MESA_LOADER_DRIVER_OVERRIDE": None,
    }


@contextlib.contextmanager
def temporary_env(overrides: Dict[str, Optional[str]]):
    """Apply temporary environment overrides and then restore values."""

    if not overrides:
        yield
        return

    previous = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[str(key)] = str(value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[str(key)] = str(value)


def probe_glx_renderer(env: Dict[str, str]) -> Dict[str, object]:
    """Probe GL renderer information through glxinfo -B."""

    glxinfo_path = shutil.which("glxinfo")
    if glxinfo_path is None:
        return {"available": False, "reason": "glxinfo not found on PATH"}

    result = subprocess.run([glxinfo_path, "-B"], capture_output=True, text=True, env=env)
    renderer = None
    vendor = None
    version = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("opengl vendor string:"):
            vendor = stripped.split(":", 1)[1].strip()
        elif lower.startswith("opengl renderer string:"):
            renderer = stripped.split(":", 1)[1].strip()
        elif lower.startswith("opengl version string:"):
            version = stripped.split(":", 1)[1].strip()
    return {
        "available": True,
        "returncode": int(result.returncode),
        "renderer": renderer,
        "vendor": vendor,
        "version": version,
        "stdout_tail": str(result.stdout[-500:]),
        "stderr_tail": str(result.stderr[-500:]),
    }


def renderer_looks_software(glx_probe: Dict[str, object]) -> bool:
    """Return True when the renderer appears software-backed."""

    renderer = str(glx_probe.get("renderer") or "").lower()
    return ("llvmpipe" in renderer) or ("soft" in renderer) or ("software rasterizer" in renderer)


def pybullet_probe_requires_env_fallback(probe_payload: Dict[str, object]) -> bool:
    """Check whether PyBullet probe indicates env override fallback."""

    if int(probe_payload.get("returncode", 0)) != 0:
        return True
    stderr_tail = str(probe_payload.get("stderr_tail") or "").lower()
    stdout_tail = str(probe_payload.get("stdout_tail") or "").lower()
    bad_tokens = ["badwindow", "nv-glx", "glx", "failed request"]
    combined = stderr_tail + "\n" + stdout_tail
    return any(token in combined for token in bad_tokens)


def probe_pybullet_gui_subprocess(stage: str, enabled: bool, env_overrides: Dict[str, Optional[str]]) -> Dict[str, object]:
    """Probe PyBullet GUI availability in a short-lived subprocess."""

    if not enabled:
        return {"enabled": False, "stage": stage}
    env = os.environ.copy()
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[str(key)] = str(value)

    glx_probe = probe_glx_renderer(env)
    cmd = [
        str(sys.executable),
        "-c",
        "import pybullet as p; cid=p.connect(p.GUI); print(f'probe_client_id={cid}'); p.disconnect(cid)",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    payload = {
        "enabled": True,
        "stage": stage,
        "returncode": int(result.returncode),
        "stdout_tail": str(result.stdout[-500:]),
        "stderr_tail": str(result.stderr[-500:]),
        "effective_env": gl_env_snapshot(env),
        "glx_probe": glx_probe,
    }
    if renderer_looks_software(glx_probe):
        warn(
            "PyBullet GUI probe still appears to be using software rendering "
            f"(renderer={glx_probe.get('renderer')})."
        )
    return payload


def connect_bullet_sim_with_env(sim, env_overrides: Dict[str, Optional[str]]) -> None:
    """Connect Bullet sim with optional temporary env overrides."""

    if not env_overrides:
        sim.connect()
        return
    try:
        with temporary_env(env_overrides):
            sim.connect()
    except Exception as exc:
        warn(
            "PyBullet GUI connect failed under preferred GPU env. "
            f"Retrying without overrides. error={exc}"
        )
        sim.connect()


def prompt_preview_action(gui_enabled: bool) -> str:
    """Return preview action in GUI mode, and auto-skip in headless mode."""

    if not bool(gui_enabled):
        return "skip"
    print("\n" + "=" * 80)
    print("About to run PyBullet preview of the compiled joint trajectory.")
    print("Type <y> to run preview, <s> to skip preview. Anything else aborts.")
    print("=" * 80)
    ans = input("> ").strip().lower()
    if ans == "y":
        return "run"
    if ans == "s":
        return "skip"
    raise SystemExit("Aborted by user.")


def quat_normalize_xyzw(q_xyzw: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Normalize quaternion tuple and reject near-zero norm."""

    qx, qy, qz, qw = q_xyzw
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n <= 1e-12:
        raise ValueError("Quaternion normalization failed: near-zero norm")
    return (qx / n, qy / n, qz / n, qw / n)


def quat_from_rpy_deg(rpy_deg: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    """Convert Euler degrees tuple to normalized quaternion."""

    roll = math.radians(float(rpy_deg[0]))
    pitch = math.radians(float(rpy_deg[1]))
    yaw = math.radians(float(rpy_deg[2]))
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return quat_normalize_xyzw((float(qx), float(qy), float(qz), float(qw)))


def resolve_target_quaternion(
    args,
    current_quat_xyzw: Tuple[float, float, float, float],
    idle_quat_xyzw: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Resolve target orientation quaternion based on selected mode."""

    mode = str(args.orientation_mode)
    if mode == "surface_normal":
        return quat_normalize_xyzw(current_quat_xyzw)
    if mode == "current":
        return quat_normalize_xyzw(current_quat_xyzw)
    if mode == "preset":
        has_named_preset = getattr(args, "preset_orientation", None) is not None
        has_rpy = args.preset_rpy_deg is not None
        has_quat = args.preset_quat_xyzw is not None
        preset_count = sum([has_named_preset, has_rpy, has_quat])
        if preset_count != 1:
            raise ValueError(
                "Preset mode requires exactly one of --preset-orientation idle, "
                "--preset-rpy-deg, or --preset-quat-xyzw"
            )
        if has_named_preset:
            if str(args.preset_orientation) != "idle":
                raise ValueError(f"Unknown preset orientation: {args.preset_orientation}")
            return quat_normalize_xyzw(idle_quat_xyzw)
        if has_quat:
            q = args.preset_quat_xyzw
            return quat_normalize_xyzw((float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        rpy = args.preset_rpy_deg
        return quat_from_rpy_deg((float(rpy[0]), float(rpy[1]), float(rpy[2])))
    raise ValueError(f"Unknown orientation mode: {mode}")


def apply_orientation_mode(
    pose_waypoints: List[object],
    orientation_mode: str,
    target_quat_xyzw: Tuple[float, float, float, float],
) -> List[object]:
    """Apply shared-orientation policy over pose waypoint sequence."""

    if orientation_mode == "surface_normal":
        return pose_waypoints
    out: List[object] = []
    q = (
        float(target_quat_xyzw[0]),
        float(target_quat_xyzw[1]),
        float(target_quat_xyzw[2]),
        float(target_quat_xyzw[3]),
    )
    for wp in pose_waypoints:
        out.append(type(wp)(xyz=wp.xyz, quat_xyzw=q, normal_xyz=wp.normal_xyz))
    return out


def extract_json_from_stdout(stdout_text: str) -> Dict[str, object]:
    """Decode the last valid JSON object from subprocess stdout."""

    lines = [line.strip() for line in str(stdout_text).splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Expected JSON output from subprocess, but stdout was empty")
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("Failed to decode JSON payload from subprocess stdout")


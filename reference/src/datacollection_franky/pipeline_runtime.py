"""Runtime helpers for process state, CPU partitioning, and robot probe."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import PipelineConfig
from .prompts import confirm_or_abort, info, warn
from .sim_runtime import extract_json_from_stdout
from .util.cpu_partition import format_cpu_set, get_cpu_partition
from .util.thread_affinity import pin_current_thread_to_cpus


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_PROBE_MODULE = "datacollection_franky.robot_probe"


def log_runtime_snapshot(tag: str, enabled: bool) -> None:
    """Log process, thread, memory, and affinity snapshot for one stage."""

    if not enabled:
        return
    pid = int(os.getpid())
    tid = int(threading.get_native_id())
    affinity = sorted(int(v) for v in os.sched_getaffinity(0))
    threads = -1
    vmrss_kb = -1
    if os.path.isfile("/proc/self/status"):
        status_file = open("/proc/self/status", "r", encoding="utf-8")
        status_lines = status_file.readlines()
        status_file.close()
        for line in status_lines:
            if line.startswith("Threads:"):
                threads = int(line.split(":", 1)[1].strip())
            elif line.startswith("VmRSS:"):
                vmrss_kb = int(line.split(":", 1)[1].strip().split()[0])
    info(
        f"[DEBUG:{tag}] pid={pid} tid={tid} "
        f"threads={threads} vmrss_kb={vmrss_kb} affinity={affinity}"
    )


def log_cpu_partition(rt_cpu_set, ik_cpu_set) -> None:
    """Log runtime CPU partition selected for RT and IK workloads."""

    if rt_cpu_set == ik_cpu_set:
        warn(
            "CPU partition fallback: fewer than 3 allowed CPU cores. "
            f"Using shared set={format_cpu_set(rt_cpu_set)} for RT and IK."
        )
        return
    info(
        f"CPU partition active: RT cores={format_cpu_set(rt_cpu_set)}; "
        f"IK/PyBullet cores={format_cpu_set(ik_cpu_set)}"
    )


def repin_main_thread_to_ik(ik_cpu_set) -> None:
    """Pin main orchestrator thread to IK/PyBullet CPU set."""

    main_tid = pin_current_thread_to_cpus(ik_cpu_set)
    info(
        "Pinned pipeline main thread "
        f"tid={main_tid} to IK/PyBullet cores={format_cpu_set(ik_cpu_set)}"
    )


def initialize_runtime_state(cfg: PipelineConfig):
    """Resolve CPU partition and apply main-thread pinning."""

    rt_cpu_set, ik_cpu_set = get_cpu_partition()
    log_cpu_partition(rt_cpu_set, ik_cpu_set)
    repin_main_thread_to_ik(ik_cpu_set)
    log_runtime_snapshot("after_main_thread_pin", enabled=bool(cfg.runtime_policy.debug_log))
    return rt_cpu_set, ik_cpu_set


def run_probe_robot_subprocess(robot_ip: str, rt_priority_cap: int) -> Dict[str, object]:
    """Execute robot probe subprocess and return its JSON payload."""

    env = os.environ.copy()
    existing_pythonpath = str(env.get("PYTHONPATH") or "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    cmd = [
        str(sys.executable),
        "-m",
        ROBOT_PROBE_MODULE,
        "--robot-ip",
        str(robot_ip),
        "--rt-priority-cap",
        str(int(rt_priority_cap)),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    try:
        payload = extract_json_from_stdout(result.stdout)
    except Exception as exc:
        raise RuntimeError(
            "Failed to decode robot probe subprocess payload: "
            f"{exc}. stderr_tail={str(result.stderr[-500:])}"
        ) from exc
    if int(result.returncode) != 0:
        raise RuntimeError(
            "Robot probe subprocess failed: "
            f"returncode={int(result.returncode)} stderr_tail={str(result.stderr[-500:])}"
        )
    return payload


def preview_fallback_pose(cfg: PipelineConfig) -> Dict[str, float]:
    """Return fixed preview fallback pose from configuration."""

    return dict(cfg.preview_fallback.current_joint_pose_by_name)


def acquire_initial_robot_state(
    args,
    cfg: PipelineConfig,
    ik_cpu_set,
) -> Tuple[Dict[str, float], bool, Optional[str]]:
    """Acquire initial joint state from probe or return preview fallback."""

    if not args.robot_ip or bool(args.preview_only):
        warn("No --robot-ip provided or --preview-only enabled. Running preview-only.")
        confirm_or_abort(
            "Preview-only mode will run using a fixed non-idle current pose.",
            token="y",
        )
        return preview_fallback_pose(cfg), False, None

    t_conn0 = float(time.monotonic())
    payload = run_probe_robot_subprocess(
        str(args.robot_ip),
        int(cfg.runtime_policy.rt_priority_cap),
    )
    info(f"Robot try_connect elapsed_s={float(time.monotonic()) - t_conn0:.3f}")
    info(
        "Probe mitigation: "
        f"pinned_rt_threads={int(payload.get('pinned_rt_threads', 0))} "
        f"priority_updates={int(payload.get('priority_updates', 0))}"
    )
    repin_main_thread_to_ik(ik_cpu_set)
    log_runtime_snapshot("after_probe_subprocess", enabled=bool(cfg.runtime_policy.debug_log))
    if bool(payload.get("connected")):
        q_current = payload.get("q_current")
        if not isinstance(q_current, dict):
            raise RuntimeError("Probe subprocess returned no joint state despite reporting connected=True")
        info(f"Connected to robot at {args.robot_ip} via short-lived subprocess probe")
        q_by_name = {str(name): float(value) for name, value in q_current.items()}
        return q_by_name, True, str(args.robot_ip)

    warn(
        f"Failed to connect to robot at {args.robot_ip} in subprocess probe. "
        f"Falling back to preview-only. error={payload.get('error')}"
    )
    confirm_or_abort(
        "No physical robot connected. Preview-only mode will run using a fixed non-idle current pose.",
        token="y",
    )
    return preview_fallback_pose(cfg), False, None


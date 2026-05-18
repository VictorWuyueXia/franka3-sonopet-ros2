"""Short-lived robot probe entry for subprocess pose capture."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict

from .config import PipelineConfig
from .franky_backend import FrankyRobot
from .util.cpu_partition import get_cpu_partition
from .util.thread_affinity import apply_rt_thread_policy


def parse_probe_args(cfg: PipelineConfig) -> argparse.Namespace:
    """Parse subprocess probe arguments."""

    p = argparse.ArgumentParser()
    p.add_argument(
        "--robot-ip",
        type=str,
        default=str(cfg.input_defaults.robot_ip),
        help="FCI IP address for subprocess probe",
    )
    p.add_argument(
        "--rt-priority-cap",
        type=int,
        default=int(cfg.runtime_policy.rt_priority_cap),
        help="Fixed RT-priority cap applied during probe mitigation",
    )
    return p.parse_args()


def run_robot_probe(robot_ip: str, cfg: PipelineConfig, rt_priority_cap: int) -> Dict[str, object]:
    """Run a one-shot robot probe and return JSON-serializable payload."""

    rt_cpu_set, _ = get_cpu_partition()
    payload: Dict[str, object] = {
        "connected": False,
        "q_current": None,
        "error": None,
        "rt_cpu_set": sorted(int(v) for v in rt_cpu_set),
        "pinned_rt_threads": 0,
        "priority_updates": 0,
    }
    robot = None
    try:
        robot = FrankyRobot(str(robot_ip), cfg.franky)
        connected = bool(robot.try_connect())
        payload["connected"] = bool(connected)
        if connected:
            policy_result = apply_rt_thread_policy(rt_cpu_set, int(rt_priority_cap))
            payload["pinned_rt_threads"] = int(policy_result["pinned_rt_threads"])
            payload["priority_updates"] = int(policy_result["priority_updates"])
            payload["q_current"] = robot.current_joint_positions_by_name()
    except Exception as exc:
        payload["error"] = str(exc)
    finally:
        if robot is not None:
            robot.disconnect()
    return payload


def main() -> None:
    """Subprocess entrypoint used by pipeline orchestration."""

    cfg = PipelineConfig()
    args = parse_probe_args(cfg)
    payload = run_robot_probe(str(args.robot_ip), cfg, int(args.rt_priority_cap))
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()


from __future__ import annotations

import os
from pathlib import Path

from .constants import HOME_DIR, PATH_SANITIZE_COMMAND, ROS_SETUP_BASH


def _append_unique_script(setup_paths: list[Path], seen_paths: set[Path], script_path: Path) -> None:
    resolved_path = script_path.expanduser().resolve()
    if not resolved_path.is_file():
        return
    if resolved_path in seen_paths:
        return
    setup_paths.append(resolved_path)
    seen_paths.add(resolved_path)


def _append_install_script_candidates(setup_paths: list[Path], seen_paths: set[Path], install_dir: Path) -> None:
    install_root = install_dir.expanduser().resolve()
    for script_name in ("setup.bash", "local_setup.bash"):
        _append_unique_script(setup_paths, seen_paths, install_root / script_name)


def _append_workspace_candidate(setup_paths: list[Path], seen_paths: set[Path], workspace_path: Path) -> None:
    workspace_root = workspace_path.expanduser().resolve()
    install_dir = workspace_root / "install"
    if install_dir.is_dir():
        _append_install_script_candidates(setup_paths, seen_paths, install_dir)
        return
    if workspace_root.name == "install":
        _append_install_script_candidates(setup_paths, seen_paths, workspace_root)


def _append_env_workspace_candidates(setup_paths: list[Path], seen_paths: set[Path]) -> None:
    env_keys = (
        "FRANKA_ROS_WS",
        "FRANKA_WS",
        "ROS2_WS",
        "MOVEIT_CALIBRATION_WS",
        "COLCON_CURRENT_PREFIX",
    )
    for env_key in env_keys:
        env_value = os.environ.get(env_key)
        if env_value:
            _append_workspace_candidate(setup_paths, seen_paths, Path(env_value))

    ament_prefix_path = os.environ.get("AMENT_PREFIX_PATH", "")
    for prefix_entry in str(ament_prefix_path).split(":"):
        if not prefix_entry.strip():
            continue
        prefix_path = Path(prefix_entry)
        _append_workspace_candidate(setup_paths, seen_paths, prefix_path)


def discover_ros_setup_scripts() -> list[Path]:
    setup_paths: list[Path] = []
    seen_paths: set[Path] = set()
    ros_setup = Path(ROS_SETUP_BASH)
    if ros_setup.is_file():
        _append_unique_script(setup_paths, seen_paths, ros_setup)

    _append_env_workspace_candidates(setup_paths, seen_paths)

    workspace_candidates = (
        HOME_DIR / "franka_ros2_ws",
        HOME_DIR / "franka_ws",
        HOME_DIR / "ros2_ws",
        HOME_DIR / "moveit_calibration_ws",
        HOME_DIR / "ws_moveit",
    )
    for workspace_path in workspace_candidates:
        _append_workspace_candidate(setup_paths, seen_paths, workspace_path)
    return setup_paths


def build_source_chain_command(setup_paths: list[Path]) -> str:
    source_parts = [PATH_SANITIZE_COMMAND]
    for setup_path in setup_paths:
        source_parts.append(f'source "{setup_path}"')
    return " && ".join(source_parts)


def build_franka_tf_helper_command(robot_ip: str, setup_paths: list[Path]) -> str:
    source_chain = build_source_chain_command(setup_paths)
    return (
        f"{source_chain} && "
        'ros2 launch franka_bringup franka.launch.py '
        'robot_type:=fr3 '
        f'robot_ip:={robot_ip} '
        'load_gripper:=true '
        'use_fake_hardware:=false '
        'fake_sensor_commands:=false '
        'joint_state_rate:=30'
    )

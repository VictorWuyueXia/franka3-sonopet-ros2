from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _camera_config():
    path = Path(get_package_share_directory("fr3_sonopet_bringup")) / "config" / "cameras.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _realsense_launch(config: dict, camera_key: str) -> IncludeLaunchDescription:
    camera = config["cameras"][camera_key]
    realsense = config["realsense"]
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"])
        ),
        launch_arguments={
            "serial_no": f"_{camera['serial']}",
            "camera_namespace": camera["namespace"],
            "camera_name": camera["camera_name"],
            "initial_reset": "false",
            "rgb_camera.color_profile": realsense["rgb_camera_profile"],
            "depth_module.depth_profile": realsense["depth_module_profile"],
            "align_depth.enable": "true",
            "pointcloud.enable": "false",
        }.items(),
    )


def generate_launch_description():
    config = _camera_config()
    return LaunchDescription(
        [
            _realsense_launch(config, "in_hand"),
            _realsense_launch(config, "fixed"),
        ]
    )

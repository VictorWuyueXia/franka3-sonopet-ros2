from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _include(name: str):
    # Local includes keep each hardware boundary readable in isolation.
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("fr3_sonopet_bringup"), "launch", name])
        )
    )


def _include_cameras():
    # RViz point picking needs live PointCloud2; non-RViz experiment launches keep it off.
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("fr3_sonopet_bringup"), "launch", "cameras.launch.py"]
            )
        ),
        launch_arguments={"pointcloud_enable": LaunchConfiguration("rviz")}.items(),
    )


def _description_include(name: str):
    # Description includes publish experiment-specific fixed frames outside vendor packages.
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("fr3_sonopet_description"), "launch", name])
        )
    )


def _raster_config():
    path = Path(get_package_share_directory("fr3_sonopet_bringup")) / "config" / "raster.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["raster"]


def generate_launch_description():
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("fr3_sonopet_bringup"), "rviz", "experiment.rviz"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="false"),
            _include("franka.launch.py"),
            _description_include("sonopet_tcp.launch.py"),
            _include_cameras(),
            _include("microphone.launch.py"),
            _include("recording.launch.py"),
            Node(
                package="fr3_sonopet_trajectory",
                executable="raster_planner_node",
                name="raster_planner_node",
                output="screen",
                parameters=[_raster_config()],
            ),
            Node(
                package="fr3_sonopet_motion",
                executable="motion_runner_node",
                name="motion_runner_node",
                output="screen",
            ),
            Node(
                package="fr3_sonopet_supervisor",
                executable="experiment_supervisor_node",
                name="experiment_supervisor_node",
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                condition=IfCondition(LaunchConfiguration("rviz")),
                output="screen",
            ),
        ]
    )

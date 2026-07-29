from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

INFO_LOG_ARGS = ["--ros-args", "--log-level", "info"]


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


def generate_launch_description():
    bringup_share = Path(get_package_share_directory("fr3_sonopet_bringup"))
    repo_root = bringup_share.parents[4]
    raster_config = yaml.safe_load(
        (bringup_share / "config" / "raster.yaml").read_text(encoding="utf-8")
    )["raster"]
    motion_config = yaml.safe_load(
        (bringup_share / "config" / "motion.yaml").read_text(encoding="utf-8")
    )["motion"]
    supervisor_config = yaml.safe_load(
        (bringup_share / "config" / "supervisor.yaml").read_text(encoding="utf-8")
    )["supervisor"]
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("fr3_sonopet_bringup"), "rviz", "experiment.rviz"]
    )
    sensor_stack = TimerAction(
        period=5.0,
        actions=[
            _description_include("sonopet_tcp.launch.py"),
            _include_cameras(),
            _include("microphone.launch.py"),
            _include("recording.launch.py"),
            Node(
                package="fr3_sonopet_trajectory",
                executable="raster_planner_node",
                name="raster_planner_node",
                output="screen",
                parameters=[raster_config],
                arguments=INFO_LOG_ARGS,
            ),
            Node(
                package="fr3_sonopet_motion",
                executable="motion_runner_node",
                name="motion_runner_node",
                output="screen",
                parameters=[motion_config],
                arguments=INFO_LOG_ARGS,
            ),
            Node(
                package="fr3_sonopet_supervisor",
                executable="experiment_supervisor_node",
                name="experiment_supervisor_node",
                output="screen",
                parameters=[supervisor_config],
            ),
            Node(
                package="fr3_sonopet_bringup",
                executable="shutdown_manager_node",
                name="shutdown_manager_node",
                output="screen",
                arguments=INFO_LOG_ARGS,
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config, *INFO_LOG_ARGS],
                condition=IfCondition(LaunchConfiguration("rviz")),
                output="screen",
            ),
        ],
    )
    return LaunchDescription(
        [
            SetEnvironmentVariable("FR3_SONOPET_REPO", str(repo_root)),
            DeclareLaunchArgument("rviz", default_value="false"),
            _include("franka.launch.py"),
            sensor_stack,
        ]
    )

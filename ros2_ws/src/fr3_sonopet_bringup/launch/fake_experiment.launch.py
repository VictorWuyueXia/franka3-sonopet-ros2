from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

INFO_LOG_ARGS = ["--ros-args", "--log-level", "info"]


def _description_include(name: str):
    # Fake launch still publishes the tool frame so TF consumers see the same graph.
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("fr3_sonopet_description"), "launch", name])
        )
    )


def generate_launch_description():
    bringup_share = Path(get_package_share_directory("fr3_sonopet_bringup"))
    raster_config = yaml.safe_load(
        (bringup_share / "config" / "raster.yaml").read_text(encoding="utf-8")
    )["raster"]
    motion_config = yaml.safe_load(
        (bringup_share / "config" / "motion.yaml").read_text(encoding="utf-8")
    )["motion"]
    supervisor_config = yaml.safe_load(
        (bringup_share / "config" / "supervisor.yaml").read_text(encoding="utf-8")
    )["supervisor"]
    # Fake hardware uses the same upstream Franka launch with non-hardware arguments.
    fake_franka = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("fr3_sonopet_bringup"), "launch", "franka.launch.py"]
            )
        ),
        launch_arguments={"robot_ip": "0.0.0.0", "use_fake_hardware": "true"}.items(),
    )

    return LaunchDescription(
        [
            fake_franka,
            _description_include("sonopet_tcp.launch.py"),
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
                parameters=[supervisor_config, {"fake_run": True}],
            ),
        ]
    )

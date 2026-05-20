from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _description_include(name: str):
    # Fake launch still publishes the tool frame so TF consumers see the same graph.
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("fr3_sonopet_description"), "launch", name])
        )
    )


def _raster_config():
    path = Path(get_package_share_directory("fr3_sonopet_bringup")) / "config" / "raster.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["raster"]


def generate_launch_description():
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
                parameters=[_raster_config()],
            ),
            Node(
                package="fr3_sonopet_motion",
                executable="motion_runner_node",
                name="motion_runner_node",
                output="screen",
                parameters=[{"fake_execution": True}],
            ),
            Node(
                package="fr3_sonopet_supervisor",
                executable="experiment_supervisor_node",
                name="experiment_supervisor_node",
                output="screen",
                parameters=[{"fake_run": True}],
            ),
        ]
    )

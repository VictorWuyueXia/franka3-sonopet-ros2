from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _include(name: str):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("fr3_sonopet_bringup"), "launch", name])
        )
    )


def generate_launch_description():
    return LaunchDescription(
        [
            _include("franka.launch.py"),
            _include("cameras.launch.py"),
            _include("microphone.launch.py"),
            _include("recording.launch.py"),
            Node(
                package="fr3_sonopet_trajectory",
                executable="raster_planner_node",
                name="raster_planner_node",
                output="screen",
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
        ]
    )


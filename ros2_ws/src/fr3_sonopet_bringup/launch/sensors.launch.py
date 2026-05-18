from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _include_launch(filename: str) -> IncludeLaunchDescription:
    # Keep sensor ownership in the individual launch files while exposing one entry point.
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("fr3_sonopet_bringup"), "launch", filename])
        )
    )


def generate_launch_description():
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("fr3_sonopet_bringup"), "rviz", "experiment.rviz"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="false"),
            _include_launch("cameras.launch.py"),
            _include_launch("microphone.launch.py"),
            _include_launch("recording.launch.py"),
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

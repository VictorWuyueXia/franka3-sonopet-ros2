from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

INFO_LOG_ARGS = ["--ros-args", "--log-level", "info"]


def generate_launch_description():
    config_path = PathJoinSubstitution(
        [FindPackageShare("fr3_sonopet_bringup"), "config", "microphone.yaml"]
    )
    return LaunchDescription(
        [
            Node(
                package="fr3_sonopet_microphone",
                executable="microphone_node",
                name="microphone_node",
                output="screen",
                parameters=[config_path],
                arguments=INFO_LOG_ARGS,
            )
        ]
    )

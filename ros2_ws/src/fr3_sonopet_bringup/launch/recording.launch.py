from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("artifact_root", default_value="artifacts/experiments"),
            Node(
                package="fr3_sonopet_recording",
                executable="recording_node",
                name="recording_node",
                output="screen",
                parameters=[{"artifact_root": "artifacts/experiments"}],
            ),
        ]
    )


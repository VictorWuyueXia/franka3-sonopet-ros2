from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_path = PathJoinSubstitution(
        [FindPackageShare("fr3_sonopet_bringup"), "config", "recording_topics.yaml"]
    )
    return LaunchDescription(
        [
            Node(
                package="fr3_sonopet_recording",
                executable="recording_node",
                name="recording_node",
                output="screen",
                parameters=[{"recording_config": config_path}],
            ),
        ]
    )

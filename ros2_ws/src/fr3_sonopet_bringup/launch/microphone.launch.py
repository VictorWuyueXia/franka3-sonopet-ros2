from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="fr3_sonopet_microphone",
                executable="microphone_node",
                name="microphone_node",
                output="screen",
            )
        ]
    )


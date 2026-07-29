from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="fr3_sonopet_bringup",
                executable="d405_intrinsics_publisher",
                name="d405_intrinsics_publisher",
                output="screen",
            )
        ]
    )

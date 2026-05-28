from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    recording_config = PathJoinSubstitution(
        [FindPackageShare("fr3_sonopet_bringup"), "config", "recording_topics.yaml"]
    )
    camera_config = PathJoinSubstitution(
        [FindPackageShare("fr3_sonopet_bringup"), "config", "cameras.yaml"]
    )
    pointcloud_config = PathJoinSubstitution(
        [FindPackageShare("fr3_sonopet_bringup"), "config", "pointcloud.yaml"]
    )
    bringup_config_dir = PathJoinSubstitution(
        [FindPackageShare("fr3_sonopet_bringup"), "config"]
    )
    description_config_dir = PathJoinSubstitution(
        [FindPackageShare("fr3_sonopet_description"), "config"]
    )
    return LaunchDescription(
        [
            Node(
                package="fr3_sonopet_recording",
                executable="recording_node",
                name="recording_node",
                output="screen",
                parameters=[
                    {
                        "recording_config": recording_config,
                        "camera_config": camera_config,
                        "bringup_config_dir": bringup_config_dir,
                        "description_config_dir": description_config_dir,
                    }
                ],
            ),
            Node(
                package="fr3_sonopet_pointcloud",
                executable="pointcloud_node",
                name="pointcloud_node",
                output="screen",
                parameters=[
                    {
                        "recording_config": recording_config,
                        "camera_config": camera_config,
                        "pointcloud_config": pointcloud_config,
                    }
                ],
            ),
        ]
    )

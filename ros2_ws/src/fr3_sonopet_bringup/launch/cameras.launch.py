from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare


def _realsense_launch(camera_key: str):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"])
        ),
        launch_arguments={
            "serial_no": [TextSubstitution(text="_"), LaunchConfiguration(f"{camera_key}_serial")],
            "camera_namespace": LaunchConfiguration(f"{camera_key}_namespace"),
            "camera_name": LaunchConfiguration(f"{camera_key}_name"),
            "align_depth.enable": "true",
            "pointcloud.enable": "true",
        }.items(),
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("in_hand_serial", default_value="323622273258"),
            DeclareLaunchArgument("in_hand_namespace", default_value="in_hand_d405"),
            DeclareLaunchArgument("in_hand_name", default_value="d405_in_hand"),
            DeclareLaunchArgument("fixed_serial", default_value="427622272709"),
            DeclareLaunchArgument("fixed_namespace", default_value="fixed_d405"),
            DeclareLaunchArgument("fixed_name", default_value="d405_fixed"),
            _realsense_launch("in_hand"),
            _realsense_launch("fixed"),
        ]
    )


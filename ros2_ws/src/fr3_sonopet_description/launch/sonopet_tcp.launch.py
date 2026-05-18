from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Static TF publishes the calibrated Sonopet TCP for robot and camera-frame consumers.
    tool_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="sonopet_tcp_static_tf",
        output="log",
        arguments=[
            "--frame-id",
            LaunchConfiguration("parent_frame"),
            "--child-frame-id",
            LaunchConfiguration("tool_frame"),
            "--x",
            LaunchConfiguration("tool_x_m"),
            "--y",
            LaunchConfiguration("tool_y_m"),
            "--z",
            LaunchConfiguration("tool_z_m"),
            "--roll",
            LaunchConfiguration("tool_roll_rad"),
            "--pitch",
            LaunchConfiguration("tool_pitch_rad"),
            "--yaw",
            LaunchConfiguration("tool_yaw_rad"),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("parent_frame", default_value="fr3_link8"),
            DeclareLaunchArgument("tool_frame", default_value="sonopet_tcp"),
            DeclareLaunchArgument("tool_x_m", default_value="0.005"),
            DeclareLaunchArgument("tool_y_m", default_value="0.010"),
            DeclareLaunchArgument("tool_z_m", default_value="0.026"),
            DeclareLaunchArgument("tool_roll_rad", default_value="0.0"),
            DeclareLaunchArgument("tool_pitch_rad", default_value="0.0"),
            DeclareLaunchArgument("tool_yaw_rad", default_value="0.0"),
            tool_tf,
        ]
    )

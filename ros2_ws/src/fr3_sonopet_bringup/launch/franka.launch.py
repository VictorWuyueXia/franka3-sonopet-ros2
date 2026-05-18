from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("franka_fr3_moveit_config"), "launch", "moveit.launch.py"]
            )
        ),
        launch_arguments={
            "robot_ip": LaunchConfiguration("robot_ip"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="172.16.0.2"),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            moveit_launch,
        ]
    )


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Upstream Franka MoveIt owns the arm controller, robot state publisher, and planning scene.
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("franka_fr3_moveit_config"),
                    "launch",
                    "moveit.launch.py",
                ]
            )
        ),
        launch_arguments={
            "robot_ip": LaunchConfiguration("robot_ip"),
            "namespace": LaunchConfiguration("namespace"),
            "load_gripper": LaunchConfiguration("load_gripper"),
            "ee_id": LaunchConfiguration("ee_id"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "fake_sensor_commands": LaunchConfiguration("fake_sensor_commands"),
            "db": LaunchConfiguration("db"),
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="172.16.0.2"),
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument("load_gripper", default_value="false"),
            DeclareLaunchArgument("ee_id", default_value="none"),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            DeclareLaunchArgument("fake_sensor_commands", default_value="false"),
            DeclareLaunchArgument("db", default_value="False"),
            moveit_launch,
        ]
    )

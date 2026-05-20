from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = yaml.safe_load(
        (Path(get_package_share_directory("fr3_sonopet_bringup")) / "config" / "cameras.yaml")
        .read_text(encoding="utf-8")
    )
    realsense = config["realsense"]
    rs_launch_path = PathJoinSubstitution(
        [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
    )
    pointcloud_enable = LaunchConfiguration("pointcloud_enable")

    actions = [DeclareLaunchArgument("pointcloud_enable", default_value="false")]
    for camera in config["cameras"].values():
        xyz = camera["translation_xyz"]
        quat = camera["quaternion_xyzw"]
        # D405 color and depth optical frames coincide at sub-millimeter for our 0.5 m crop range.
        for child_frame, suffix in (
            (camera["color_optical_frame"], "color_optical"),
            (camera["depth_optical_frame"], "depth_optical"),
        ):
            actions.append(
                Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    name=f"{camera['camera_name']}_{suffix}_static_tf",
                    output="log",
                    arguments=[
                        "--frame-id", camera["parent_frame"],
                        "--child-frame-id", child_frame,
                        "--x", str(xyz[0]), "--y", str(xyz[1]), "--z", str(xyz[2]),
                        "--qx", str(quat[0]), "--qy", str(quat[1]),
                        "--qz", str(quat[2]), "--qw", str(quat[3]),
                    ],
                )
            )
        # publish_tf:=false stops the driver from publishing a competing optical-frame chain.
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch_path),
                launch_arguments={
                    "serial_no": f"_{camera['serial']}",
                    "camera_namespace": camera["namespace"],
                    "camera_name": camera["camera_name"],
                    "initial_reset": "false",
                    "publish_tf": "false",
                    "enable_color": "true",
                    "enable_depth": "true",
                    "enable_sync": "true",
                    "rgb_camera.color_profile": realsense["rgb_camera_profile"],
                    "depth_module.depth_profile": realsense["depth_module_profile"],
                    "align_depth.enable": "true",
                    "pointcloud.enable": pointcloud_enable,
                    "pointcloud.stream_filter": "2",
                    "pointcloud.stream_index_filter": "0",
                    "pointcloud.ordered_pc": "true",
                    "pointcloud.allow_no_texture_points": "false",
                }.items(),
            )
        )

    return LaunchDescription(actions)

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

CAMERA_NAMESPACE = "RealSense_D405"
COLOR_PROFILE = "848x480x30"
DEPTH_PROFILE = "848x480x30"


def _realsense_launch(camera_name: str, serial: str) -> IncludeLaunchDescription:
    # RealSense owns sensor transport; this launch file fixes our experiment naming policy.
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"])
        ),
        launch_arguments={
            "serial_no": f"_{serial}",
            "camera_namespace": CAMERA_NAMESPACE,
            "camera_name": camera_name,
            "initial_reset": "false",
            "rgb_camera.color_profile": COLOR_PROFILE,
            "depth_module.depth_profile": DEPTH_PROFILE,
            "align_depth.enable": "true",
            "pointcloud.enable": "false",
        }.items(),
    )


def generate_launch_description():
    # Both D405 nodes share the device-class namespace and differ only by experiment role.
    return LaunchDescription(
        [
            _realsense_launch("in_hand", "323622273258"),
            _realsense_launch("fixed", "427622272709"),
        ]
    )

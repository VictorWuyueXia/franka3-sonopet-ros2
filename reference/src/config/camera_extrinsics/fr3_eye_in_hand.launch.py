""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-IN-HAND: fr3_hand_tcp -> camera_color_optical_frame """
import json
from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_eye_in_hand_transform() -> dict:
    config_path = _repo_root() / "config" / "camera_extrinsics" / "fr3_eye_in_hand.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def generate_launch_description() -> LaunchDescription:
    transform = load_eye_in_hand_transform()
    translation_xyz = transform["translation_xyz"]
    quaternion_xyzw = transform["quaternion_xyzw"]
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                transform["parent_frame"],
                "--child-frame-id",
                transform["child_frame"],
                "--x",
                str(translation_xyz[0]),
                "--y",
                str(translation_xyz[1]),
                "--z",
                str(translation_xyz[2]),
                "--qx",
                str(quaternion_xyzw[0]),
                "--qy",
                str(quaternion_xyzw[1]),
                "--qz",
                str(quaternion_xyzw[2]),
                "--qw",
                str(quaternion_xyzw[3]),
            ],
        ),
    ]
    return LaunchDescription(nodes)

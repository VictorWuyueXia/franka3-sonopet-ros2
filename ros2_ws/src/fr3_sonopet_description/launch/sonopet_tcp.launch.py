from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="fr3_sonopet_tool">
  <link name="fr3_link8"/>

  <joint name="EE_base_joint" type="fixed">
    <parent link="fr3_link8"/>
    <child link="sonopet_link"/>
    <origin xyz="{ee_base_xyz}" rpy="{ee_base_rpy}"/>
  </joint>

  <link name="sonopet_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://fr3_sonopet_description/meshes/sonopet.dae"/>
      </geometry>
    </visual>
  </link>

  <link name="sonopet_tcp_nominal"/>
  <joint name="sonopet_tcp_nominal_joint" type="fixed">
    <parent link="sonopet_link"/>
    <child link="sonopet_tcp_nominal"/>
    <origin xyz="{nominal_tcp_xyz}" rpy="{nominal_tcp_rpy}"/>
  </joint>

  <link name="sonopet_tcp"/>
  <joint name="sonopet_tcp_trim_joint" type="fixed">
    <parent link="sonopet_tcp_nominal"/>
    <child link="sonopet_tcp"/>
    <origin xyz="{tcp_trim_xyz}" rpy="{tcp_trim_rpy}"/>
  </joint>
</robot>
"""


def generate_launch_description():
    # The Sonopet chain stays in one URDF string so robot_state_publisher owns its TF subtree.
    package_share = Path(get_package_share_directory("fr3_sonopet_description"))
    offsets = yaml.safe_load(
        (package_share / "config" / "sonopet_tcp_offsets.yaml").read_text(encoding="utf-8")
    )["sonopet_tcp"]
    urdf = _URDF_TEMPLATE.format(
        ee_base_xyz=" ".join(f"{v:.12g}" for v in offsets["ee_base_xyz_m"]),
        ee_base_rpy=" ".join(f"{v:.12g}" for v in offsets["ee_base_rpy_rad"]),
        nominal_tcp_xyz=" ".join(f"{v:.12g}" for v in offsets["nominal_tcp_xyz_m"]),
        nominal_tcp_rpy=" ".join(f"{v:.12g}" for v in offsets["nominal_tcp_rpy_rad"]),
        tcp_trim_xyz=" ".join(f"{v:.12g}" for v in offsets["tcp_trim_xyz_m"]),
        tcp_trim_rpy=" ".join(f"{v:.12g}" for v in offsets["tcp_trim_rpy_rad"]),
    )
    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace="sonopet",
                name="robot_state_publisher",
                output="log",
                parameters=[{"robot_description": urdf}],
            )
        ]
    )

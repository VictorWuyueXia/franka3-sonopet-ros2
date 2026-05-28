import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    Shutdown,
)
from launch.conditions import UnlessCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

FR3_ARM_JOINTS = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]


def _load_yaml(package_name: str, file_path: str):
    absolute_path = os.path.join(get_package_share_directory(package_name), file_path)
    with open(absolute_path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def generate_launch_description():
    # This is a slim copy of franka_fr3_moveit_config/launch/moveit.launch.py with the vendor
    # rviz node removed, so our own RViz is the only window the operator sees.
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    namespace = LaunchConfiguration("namespace")

    franka_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots", "fr3", "fr3.urdf.xacro",
    )
    robot_description_config = Command(
        [
            FindExecutable(name="xacro"), " ", franka_xacro_file,
            " hand:=false",
            " robot_ip:=", robot_ip,
            " ee_id:=none",
            " use_fake_hardware:=", use_fake_hardware,
            " fake_sensor_commands:=", fake_sensor_commands,
            " ros2_control:=true",
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_config, value_type=str)
    }

    franka_semantic_xacro_file = os.path.join(
        get_package_share_directory("franka_description"),
        "robots", "fr3", "fr3.srdf.xacro",
    )
    robot_description_semantic_config = Command(
        [
            FindExecutable(name="xacro"), " ", franka_semantic_xacro_file,
            " hand:=false", " ee_id:=none",
        ]
    )
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            robot_description_semantic_config, value_type=str
        )
    }

    kinematics_config = {
        "robot_description_kinematics": _load_yaml(
            "franka_fr3_moveit_config", "config/kinematics.yaml"
        )
    }
    joint_limits_config = {
        "robot_description_planning": _load_yaml(
            "franka_fr3_moveit_config", "config/fr3_joint_limits.yaml"
        )
    }

    # MoveIt planning pipeline (OMPL) is the same baseline used by the vendor launch.
    ompl_planning_pipeline_config = {
        "move_group": {
            "planning_plugins": ["ompl_interface/OMPLPlanner"],
            "request_adapters": [
                "default_planning_request_adapters/ResolveConstraintFrames",
                "default_planning_request_adapters/ValidateWorkspaceBounds",
                "default_planning_request_adapters/CheckStartStateBounds",
                "default_planning_request_adapters/CheckStartStateCollision",
            ],
            "response_adapters": [
                "default_planning_response_adapters/AddTimeOptimalParameterization",
                "default_planning_response_adapters/ValidateSolution",
                "default_planning_response_adapters/DisplayMotionPath",
            ],
            "start_state_max_bounds_error": 0.1,
        }
    }
    ompl_planning_pipeline_config["move_group"].update(
        _load_yaml("franka_fr3_moveit_config", "config/ompl_planning.yaml")
    )

    moveit_controller_config = _load_yaml(
        "franka_fr3_moveit_config", "config/fr3_controllers.yaml"
    )
    moveit_controller_config["controller_names"] = ["fr3_arm_controller"]
    moveit_controller_config.pop("fr3_gripper", None)
    moveit_controllers = {
        "moveit_simple_controller_manager": moveit_controller_config,
        "moveit_controller_manager":
            "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }
    trajectory_execution = {
        "moveit_manage_controllers": True,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }
    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace=namespace,
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_config,
            joint_limits_config,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
        ],
        remappings=[("/compute_ik", "/fr3/compute_ik")],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        namespace=namespace,
        output="both",
        parameters=[robot_description],
    )
    preview_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="preview_robot_state_publisher",
        namespace=namespace,
        output="log",
        parameters=[robot_description, {"frame_prefix": "preview/"}],
        remappings=[("joint_states", "/sonopet/preview/joint_states")],
    )
    preview_root_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="preview_root_static_transform_publisher",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
            "--frame-id", "base",
            "--child-frame-id", "preview/base",
        ],
        output="log",
    )

    ros2_controllers_path = os.path.join(
        get_package_share_directory("franka_fr3_moveit_config"),
        "config", "fr3_ros_controllers.yaml",
    )
    joint_state_broadcaster_config = {
        "joint_state_broadcaster": {
            "ros__parameters": {
                "joints": FR3_ARM_JOINTS,
                "interfaces": ["position", "velocity", "effort"],
            }
        }
    }
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        namespace=namespace,
        parameters=[robot_description, ros2_controllers_path, joint_state_broadcaster_config],
        output={"stdout": "screen", "stderr": "screen"},
        on_exit=Shutdown(),
    )

    load_controllers = [
        ExecuteProcess(
            cmd=[
                "ros2", "run", "controller_manager", "spawner", controller,
                "--controller-manager-timeout", "60",
                "--controller-manager",
                PathJoinSubstitution([namespace, "controller_manager"]),
            ],
            output="screen",
        )
        for controller in ("fr3_arm_controller", "joint_state_broadcaster")
    ]

    franka_robot_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        namespace=namespace,
        arguments=["franka_robot_state_broadcaster"],
        output="screen",
        condition=UnlessCondition(use_fake_hardware),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="172.16.0.2"),
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            DeclareLaunchArgument("fake_sensor_commands", default_value="false"),
            move_group_node,
            robot_state_publisher,
            preview_robot_state_publisher,
            preview_root_tf,
            ros2_control_node,
            franka_robot_state_broadcaster,
        ]
        + load_controllers
    )

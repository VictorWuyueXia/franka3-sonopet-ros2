from pathlib import Path


def test_expected_launch_files_exist():
    package_root = Path(__file__).resolve().parents[1]
    expected = {
        "experiment.launch.py",
        "fake_experiment.launch.py",
        "franka.launch.py",
        "cameras.launch.py",
        "capture_d405_intrinsics.launch.py",
        "d405_intrinsics.launch.py",
        "microphone.launch.py",
        "recording.launch.py",
        "sensors.launch.py",
    }
    actual = {path.name for path in (package_root / "launch").glob("*.launch.py")}
    assert expected <= actual


def test_sensor_configs_encode_snapshot_and_microphone_policy():
    package_root = Path(__file__).resolve().parents[1]
    cameras = (package_root / "config" / "cameras.yaml").read_text(encoding="utf-8")
    recording = (package_root / "config" / "recording_topics.yaml").read_text(encoding="utf-8")
    raster = (package_root / "config" / "raster.yaml").read_text(encoding="utf-8")
    motion = (package_root / "config" / "motion.yaml").read_text(encoding="utf-8")
    microphone = (package_root / "config" / "microphone.yaml").read_text(encoding="utf-8")

    assert not any(line.lstrip().startswith("#") for line in cameras.splitlines())
    assert not any(line.lstrip().startswith("#") for line in recording.splitlines())
    assert not any(line.lstrip().startswith("#") for line in raster.splitlines())
    assert not any(line.lstrip().startswith("#") for line in motion.splitlines())
    assert "rgb_camera_profile: 848x480x30" in cameras
    assert "translation_xyz:" in cameras
    assert "quaternion_xyzw:" in cameras
    assert "parent_frame: fr3_link8" in cameras
    assert "parent_frame: fr3_link0" in cameras
    assert "color_optical_frame:" in cameras
    assert "depth_optical_frame:" in cameras
    assert "pointcloud_snapshots:" in cameras
    assert "/RealSense_D405/in_hand/color/image_rect_raw" in recording
    assert "/RealSense_D405/fixed/color/image_rect_raw" in recording
    assert "planning_cloud_topic: /sonopet/captured_planning_cloud" in raster
    assert "planning_cloud_display_topic: /sonopet/planning_cloud" in raster
    assert "clicked_point_topic: /clicked_point" in raster
    assert "idle_joint_positions:" in motion
    assert "motion_speed_m_s: 0.03" in motion
    assert "raster_speed_m_s:" in motion
    assert "parking_lift_m: 0.05" in motion
    assert "/microphone/audio" in recording
    assert "ros__parameters:" in microphone
    assert "iMM-6C" in microphone
    assert "imm6c" in microphone
    assert "fail_if_preferred_not_found" not in microphone


def test_launch_files_use_sensor_only_recording_defaults():
    package_root = Path(__file__).resolve().parents[1]
    src_root = Path(__file__).resolve().parents[2]
    cameras_launch = (package_root / "launch" / "cameras.launch.py").read_text(encoding="utf-8")
    franka_launch = (package_root / "launch" / "franka.launch.py").read_text(encoding="utf-8")
    sonopet_tcp_launch = (
        src_root / "fr3_sonopet_description" / "launch" / "sonopet_tcp.launch.py"
    ).read_text(encoding="utf-8")
    experiment_launch = (package_root / "launch" / "experiment.launch.py").read_text(
        encoding="utf-8"
    )
    recording_launch = (package_root / "launch" / "recording.launch.py").read_text(encoding="utf-8")
    microphone_launch = (package_root / "launch" / "microphone.launch.py").read_text(
        encoding="utf-8"
    )
    rviz = (package_root / "rviz" / "experiment.rviz").read_text(encoding="utf-8")

    assert "cameras.yaml" in cameras_launch
    assert "camera['namespace']" in cameras_launch or 'camera["namespace"]' in cameras_launch
    assert "camera['camera_name']" in cameras_launch or 'camera["camera_name"]' in cameras_launch
    assert 'DeclareLaunchArgument("pointcloud_enable", default_value="false")' in cameras_launch
    assert '"pointcloud.enable": pointcloud_enable' in cameras_launch
    assert 'LaunchConfiguration("pointcloud_enable")' in cameras_launch
    assert "static_transform_publisher" in cameras_launch
    assert "translation_xyz" in cameras_launch
    assert "quaternion_xyzw" in cameras_launch
    assert '"publish_tf": "false"' in cameras_launch
    assert '"pointcloud.stream_filter"' in cameras_launch
    assert '"enable_color": "true"' in cameras_launch
    assert "raster.yaml" in experiment_launch
    assert "motion.yaml" in experiment_launch
    assert "parameters=[motion_config]" in experiment_launch
    assert '" hand:=false"' in franka_launch
    assert '" ee_id:=none"' in franka_launch
    assert "franka_gripper" not in franka_launch
    assert "load_gripper" not in franka_launch
    assert 'remappings=[("joint_states", "franka/joint_states")]' not in franka_launch
    assert 'moveit_controller_config["controller_names"] = ["fr3_arm_controller"]' in franka_launch
    assert '"joints": FR3_ARM_JOINTS' in franka_launch
    assert '"interfaces": ["position", "velocity", "effort"]' in franka_launch
    assert "joint_state_publisher" not in franka_launch
    assert "/sonopet/preview/joint_states" in franka_launch
    assert "preview_robot_state_publisher" in franka_launch
    assert '"frame_prefix": "preview/"' in franka_launch
    assert '"--frame-id", "base"' in franka_launch
    assert '"--child-frame-id", "preview/base"' in franka_launch
    assert '"frame_prefix": "preview/"' in sonopet_tcp_launch
    pointcloud_arg = 'launch_arguments={"pointcloud_enable": LaunchConfiguration("rviz")}'
    assert pointcloud_arg in experiment_launch
    assert "parameters=[raster_config]" in experiment_launch
    assert "/sonopet/planning_cloud" in rviz
    assert "Transient Local" in rviz
    assert "/sonopet/raster_plan/poses" in rviz
    assert "/sonopet/raster_plan/markers" in rviz
    assert "fr3_sonopet_interfaces/MotionControlPanel" in rviz
    assert "rviz_common/Displays" not in rviz
    assert "rviz_default_plugins/PublishPoint" in rviz
    assert "rviz_default_plugins/RobotModel" in rviz
    assert "/robot_description" in rviz
    assert "/sonopet/robot_description" in rviz
    assert "Preview Franka Robot" in rviz
    assert "Preview Sonopet Tool" in rviz
    assert "Value: /sonopet/preview/joint_states" not in rviz
    assert "Enabled: false" in rviz
    assert "TF Prefix: preview" in rviz
    assert "All Enabled: false" in rviz
    assert "Color Transformer: RGB8" in rviz
    assert "recording_topics.yaml" in recording_launch
    assert "cameras.yaml" in recording_launch
    assert "bringup_config_dir" in recording_launch
    assert "description_config_dir" in recording_launch
    assert "fr3_sonopet_description" in recording_launch
    assert "microphone.yaml" in microphone_launch


def test_motion_panel_plugin_is_exported_from_interfaces_package():
    src_root = Path(__file__).resolve().parents[2]
    interfaces_root = src_root / "fr3_sonopet_interfaces"
    cmake = (interfaces_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package = (interfaces_root / "package.xml").read_text(encoding="utf-8")
    plugin = (interfaces_root / "plugin_description.xml").read_text(encoding="utf-8")
    panel = (interfaces_root / "src" / "motion_control_panel.cpp").read_text(encoding="utf-8")

    assert "pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)" in cmake
    assert "fr3_sonopet_interfaces/MotionControlPanel" in plugin
    assert "<build_depend>rviz_common</build_depend>" in package
    assert "<exec_depend>rclcpp_action</exec_depend>" in package
    assert "setPreviewDisplayMode(true)" in panel
    assert "setPreviewDisplayMode(false)" in panel
    assert "Preview Franka Robot" in panel
    assert "Preview Sonopet Tool" in panel
    assert "Franka Robot" in panel
    assert "Sonopet Tool" in panel
    assert 'name == "TF"' in panel
    assert "Re-sample Raster" in panel
    assert "BuildRasterPlan" in panel
    assert "/sonopet/build_raster_plan" in panel
    assert "goal.update_selected_center = false" in panel

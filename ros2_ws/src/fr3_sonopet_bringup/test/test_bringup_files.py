from pathlib import Path


def test_expected_launch_files_exist():
    package_root = Path(__file__).resolve().parents[1]
    expected = {
        "experiment.launch.py",
        "fake_experiment.launch.py",
        "franka.launch.py",
        "cameras.launch.py",
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
    microphone = (package_root / "config" / "microphone.yaml").read_text(encoding="utf-8")

    assert "pointcloud_enable: false" in cameras
    assert "pointcloud_snapshots:" in recording
    assert "/in_hand_d405/d405_in_hand/color/image_raw" in recording
    assert "/fixed_d405/d405_fixed/color/image_raw" in recording
    assert "/microphone/audio" in recording
    assert "iMM-6C" in microphone
    assert "imm6c" in microphone
    assert "fail_if_preferred_not_found: true" in microphone


def test_launch_files_use_sensor_only_recording_defaults():
    package_root = Path(__file__).resolve().parents[1]
    cameras_launch = (package_root / "launch" / "cameras.launch.py").read_text(encoding="utf-8")
    recording_launch = (package_root / "launch" / "recording.launch.py").read_text(encoding="utf-8")
    microphone_launch = (package_root / "launch" / "microphone.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument("pointcloud_enable", default_value="false")' in cameras_launch
    assert '"pointcloud.enable": LaunchConfiguration("pointcloud_enable")' in cameras_launch
    assert "recording_topics.yaml" in recording_launch
    assert "microphone.yaml" in microphone_launch

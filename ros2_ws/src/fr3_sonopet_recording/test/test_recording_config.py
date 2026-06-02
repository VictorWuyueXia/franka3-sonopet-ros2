from fr3_sonopet_recording.recording_config import (
    artifact_root,
    experiment_run_id,
    load_recording_config,
    local_timestamp_label,
    recording_run_label,
    recording_topics,
    wall_clock_timestamp,
)


def test_load_recording_config_keeps_only_live_recording_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("FR3_SONOPET_REPO", str(tmp_path))
    recording_path = tmp_path / "recording.yaml"
    camera_path = tmp_path / "cameras.yaml"
    recording_path.write_text(
        """
recording:
  audio:
    topic: /microphone/audio
  cameras:
    in_hand:
      rgb_topic: /RealSense_D405/in_hand/color/image_rect_raw
      pointcloud_topic: /RealSense_D405/in_hand/depth/color/points
      parameter_service: /RealSense_D405/in_hand/set_parameters
    fixed:
      rgb_topic: /RealSense_D405/fixed/color/image_rect_raw
      pointcloud_topic: /RealSense_D405/fixed/depth/color/points
      parameter_service: /RealSense_D405/fixed/set_parameters
""",
        encoding="utf-8",
    )
    camera_path.write_text(
        """
realsense:
  fps: 15.0
pointcloud_snapshots:
  timeout_sec: 5.0
""",
        encoding="utf-8",
    )

    config = load_recording_config(recording_path, camera_path)

    assert config.artifact_root == artifact_root()
    assert config.artifact_root == tmp_path / "data_collection" / "experiments"
    assert config.video_fps == 15.0
    assert config.cutting_topic == "/sonopet/cutting"
    assert not hasattr(config.cameras[0], "pointcloud_topic")
    assert not hasattr(config.cameras[0], "parameter_service")
    assert recording_topics(config) == (
        "/microphone/audio",
        "/sonopet/cutting",
        "/RealSense_D405/in_hand/color/image_rect_raw",
        "/RealSense_D405/fixed/color/image_rect_raw",
    )


def test_experiment_run_id_uses_local_wall_clock():
    run_id = experiment_run_id()
    time_part, setting_part = run_id.split("_", 1)

    assert len(time_part) == 15
    assert time_part[8] == "T"
    assert "Z" not in run_id
    assert setting_part == "sample_device_setting"
    date_part, clock_part = time_part.split("T")
    assert len(date_part) == 8
    assert len(clock_part) == 6
    assert date_part.isdigit()
    assert clock_part.isdigit()


def test_recording_run_label_names_cutting_intervals():
    assert recording_run_label(1) == "run_1"
    assert recording_run_label(3) == "run_3"


def test_wall_clock_timestamp_is_epoch_seconds():
    stamp = wall_clock_timestamp()

    assert isinstance(stamp, float)
    assert stamp > 1_700_000_000.0


def test_local_timestamp_label_uses_compact_local_wall_clock():
    from datetime import datetime

    local_tz = datetime.now().astimezone().tzinfo
    epoch = datetime(2026, 5, 27, 15, 51, 18, tzinfo=local_tz).timestamp()
    label = local_timestamp_label(epoch)

    assert label == "202605271551"

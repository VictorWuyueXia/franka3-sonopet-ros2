from fr3_sonopet_recording.recording_config import load_recording_config, recording_topics


def test_load_recording_config_keeps_only_live_recording_keys(tmp_path):
    recording_path = tmp_path / "recording.yaml"
    camera_path = tmp_path / "cameras.yaml"
    recording_path.write_text(
        """
recording:
  artifact_root: artifacts/experiments
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

    assert config.video_fps == 15.0
    assert config.snapshot_timeout_sec == 5.0
    assert recording_topics(config) == (
        "/microphone/audio",
        "/RealSense_D405/in_hand/color/image_rect_raw",
        "/RealSense_D405/fixed/color/image_rect_raw",
    )

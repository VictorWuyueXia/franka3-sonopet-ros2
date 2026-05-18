from fr3_sonopet_recording.recording_config import load_recording_config, recording_topics


def test_load_recording_config_declares_rgb_audio_and_snapshot_policy(tmp_path):
    config_path = tmp_path / "recording.yaml"
    config_path.write_text(
        """
recording:
  artifact_root: artifacts/experiments
  rgb_video:
    codec: MJPG
    fps: 30.0
  audio:
    topic: /microphone/audio
    directory: audio
  pointcloud_snapshots:
    enabled: true
    timeout_sec: 5.0
    output_format: pcd
    start_label: start
    end_label: end
  cameras:
    in_hand:
      directory: camera_in_hand
      rgb_topic: /in_hand/color/image_raw
      pointcloud_topic: /in_hand/depth/color/points
      parameter_service: /in_hand/set_parameters
""",
        encoding="utf-8",
    )

    config = load_recording_config(config_path)

    assert config.video_codec == "MJPG"
    assert config.snapshots.output_format == "pcd"
    assert recording_topics(config) == (
        "/microphone/audio",
        "/in_hand/color/image_raw",
    )

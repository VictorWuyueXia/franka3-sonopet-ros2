from pathlib import Path


def test_recording_node_delegates_pointcloud_capture_to_action_client():
    package_root = Path(__file__).resolve().parents[1]
    node_source = (
        package_root / "src" / "fr3_sonopet_recording" / "recording_node.py"
    ).read_text(encoding="utf-8")

    assert "ActionClient(" in node_source
    assert '"/sonopet/capture_pointcloud"' in node_source
    assert "goal.artifact_root = str(" in node_source
    assert "ActionServer(\n            self,\n            CapturePointCloud" not in node_source
    assert "write_pointcloud_pcd" not in node_source
    assert "TransformListener" not in node_source
    assert "/joint_states" not in node_source
    assert "SetParameters" not in node_source


def test_recording_node_exposes_cutting_workflow_and_artifact_discard():
    package_root = Path(__file__).resolve().parents[1]
    node_source = (
        package_root / "src" / "fr3_sonopet_recording" / "recording_node.py"
    ).read_text(encoding="utf-8")

    assert "/sonopet/set_artifact_saving" in node_source
    assert "/sonopet/cutting" in node_source
    assert "class ActiveRecordingRun" in node_source
    assert "def _prepare_session_folder(self) -> None:" in node_source
    assert "recording_run_label" in node_source
    assert "def _next_existing_name(self, path: Path) -> Path:" in node_source
    assert "wall_clock_timestamp" in node_source
    assert "def _copy_config_artifacts(self) -> None:" in node_source
    assert "def _on_cutting_flag(self, msg: Bool) -> None:" in node_source
    assert "def _start_recording_run(self) -> None:" in node_source
    assert "def _write_rgb_frame(self, camera_key: str, msg: Image) -> None:" in node_source
    assert "def _write_audio_packet(self, msg: AudioChunk) -> None:" in node_source
    assert "def _stop_recording_run(self) -> None:" in node_source
    assert "self._run_lock = Lock()" in node_source
    assert "def _close_run_files(self, active_run: ActiveRecordingRun) -> None:" in node_source
    assert "boundary_error: Exception | None = None" in node_source
    assert "def _discard_experiment_folder(self) -> None:" in node_source
    assert "if self._active_run is None:" in node_source
    assert "shutil.rmtree(self._artifact_path)" in node_source
    assert "rgb_timestamps.csv" not in node_source
    assert "RgbVideoRecorder" not in node_source
    assert "AudioWavRecorder" not in node_source


def test_recording_node_metadata_and_naming_policy_are_encoded():
    package_root = Path(__file__).resolve().parents[1]
    node_source = (
        package_root / "src" / "fr3_sonopet_recording" / "recording_node.py"
    ).read_text(encoding="utf-8")

    assert '"run": active_run.label' in node_source
    assert '"timestamp": scan.timestamp' in node_source
    assert "started_at_local" in node_source
    assert "timestamp_local" in node_source
    assert "camera_in_hand" in node_source
    assert "camera_fixed" in node_source
    assert "rgb.avi" in node_source
    assert "audio.wav" in node_source
    assert "audio_meta.json" in node_source
    assert "manifest.json" in node_source
    assert "started_at" in node_source
    assert "stopped_at" in node_source
    assert "frame_rate" in node_source
    assert "sample_rate_hz" in node_source
    assert "config" in node_source and "bringup" in node_source and "description" in node_source

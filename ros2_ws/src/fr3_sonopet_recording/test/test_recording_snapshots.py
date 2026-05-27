from pathlib import Path

from fr3_sonopet_recording import artifact_writers
from fr3_sonopet_recording.artifact_writers import capture_pointcloud_snapshot


def test_pointcloud_snapshot_enables_receives_writes_and_disables(tmp_path, monkeypatch):
    enabled_states = []

    def _write_cloud(path, cloud):
        assert path == tmp_path / "pointcloud_start.pcd"
        assert cloud == {"timeout": 5.0}
        return 42

    monkeypatch.setattr(artifact_writers, "write_pointcloud_pcd", _write_cloud)

    result = capture_pointcloud_snapshot(
        "start",
        tmp_path / "pointcloud_start.pcd",
        5.0,
        enabled_states.append,
        lambda timeout: {"timeout": timeout},
    )

    assert enabled_states == [True, False]
    assert result.success is True
    assert result.point_count == 42


def test_pointcloud_snapshot_disables_after_failure(tmp_path):
    enabled_states = []

    def _receive_cloud(_timeout):
        raise RuntimeError("no pointcloud")

    result = capture_pointcloud_snapshot(
        "end",
        tmp_path / "pointcloud_end.pcd",
        0.1,
        enabled_states.append,
        _receive_cloud,
    )

    assert enabled_states == [True, False]
    assert result.success is False
    assert "no pointcloud" in result.error


def test_recording_node_exposes_cutting_workflow_and_artifact_discard():
    package_root = Path(__file__).resolve().parents[1]
    node_source = (
        package_root / "src" / "fr3_sonopet_recording" / "recording_node.py"
    ).read_text(encoding="utf-8")

    assert "/sonopet/capture_pointcloud" in node_source
    assert "/sonopet/captured_planning_cloud" in node_source
    assert "/sonopet/set_artifact_saving" in node_source
    assert "/sonopet/cutting" in node_source
    assert "class ActiveRecordingRun" in node_source
    assert "def _prepare_experiment_folder(self) -> None:" in node_source
    assert "def _copy_config_artifacts(self) -> None:" in node_source
    assert "def _on_cutting_flag(self, msg: Bool) -> None:" in node_source
    assert "def _start_recording_run(self) -> None:" in node_source
    assert "def _write_rgb_frame(self, camera_key: str, msg: Image) -> None:" in node_source
    assert "def _write_audio_packet(self, msg: AudioChunk) -> None:" in node_source
    assert "def _stop_recording_run(self) -> None:" in node_source
    assert "def _next_existing_name(self, path: Path) -> Path:" in node_source
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

    assert "%Y-%m-%d:%H-%M-%S" in node_source
    assert "centiseconds // 10" in node_source
    assert "centiseconds % 10" in node_source
    assert 'path.with_name(f"{path.stem}_{index}{path.suffix}")' in node_source
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

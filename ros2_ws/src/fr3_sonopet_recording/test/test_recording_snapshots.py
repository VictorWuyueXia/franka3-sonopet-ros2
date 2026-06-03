from pathlib import Path


def _recording_node_source() -> str:
    package_root = Path(__file__).resolve().parents[1]
    return (
        package_root / "src" / "fr3_sonopet_recording" / "recording_node.py"
    ).read_text(encoding="utf-8")


def test_recording_node_delegates_pointcloud_capture_to_action_client():
    node_source = _recording_node_source()

    assert "ActionClient(" in node_source
    assert '"/realsense/capture_pointcloud"' in node_source
    assert "goal.artifact_root = str(" in node_source
    assert "ActionServer(\n            self,\n            CapturePointCloud" not in node_source
    assert "write_pointcloud_pcd" not in node_source
    assert "TransformListener" not in node_source
    assert "/joint_states" not in node_source
    assert "SetParameters" not in node_source


def test_recording_node_exposes_cutting_workflow_and_artifact_discard():
    node_source = _recording_node_source()

    assert "/set_artifact_saving" in node_source
    assert "/sonopet/cutting" in node_source
    assert 'ARTIFACT_PATH_TOPIC = "/sonopet/artifact_path"' in node_source
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in node_source
    assert "self._artifact_path_pub.publish(String(data=str(self._artifact_path)))" in node_source
    assert "class ActiveRecordingRun" in node_source
    assert "def _prepare_session_folder(self) -> None:" in node_source
    assert "recording_run_label" in node_source
    assert "wall_clock_timestamp" in node_source
    assert "def _copy_config_artifacts(self) -> None:" in node_source
    assert "def _on_cutting_flag(self, msg: Bool) -> None:" in node_source
    assert "self._require_workflow_ready()" in node_source
    assert "SONOPET ERROR: Sonopet is not connected or not ready" in node_source
    assert "VIDEO ERROR: no fresh RGB frames" in node_source
    assert "not starting this cutting run" in node_source
    assert "Startup planning cloud capture failed; recording remains available" in node_source
    assert "RECORDING ERROR: {run_label} was not recorded" in node_source
    assert "pointcloud_stop failed after recording stopped" in node_source
    assert "def _start_recording_run(self) -> None:" in node_source
    assert "def _write_rgb_frame(self, camera_key: str, msg: Image) -> None:" in node_source
    assert "def _write_audio_packet(self, msg: AudioChunk) -> None:" in node_source
    assert "def _stop_recording_run(self) -> None:" in node_source
    assert "self._run_lock = Lock()" in node_source
    assert "self._boundary_lock = Lock()" in node_source
    assert "def _close_run_files(self, active_run: ActiveRecordingRun) -> None:" in node_source
    assert "boundary_error: Exception | None = None" in node_source
    assert "def _discard_experiment_folder(self) -> None:" in node_source
    assert "if self._active_run is None:" in node_source
    assert "shutil.rmtree(self._artifact_path)" in node_source
    assert "rgb_timestamps.csv" not in node_source
    assert "RgbVideoRecorder" not in node_source
    assert "AudioWavRecorder" not in node_source


def test_recording_node_metadata_and_naming_policy_are_encoded():
    node_source = _recording_node_source()

    assert '"run": active_run.label' in node_source
    assert '"timestamp": scan.timestamp' in node_source
    assert "started_at_local" in node_source
    assert "timestamp_local" in node_source
    assert "camera_in_hand" in node_source
    assert "camera_fixed" in node_source
    assert "rgb_{run_index}.avi" in node_source
    assert "audio_{run_index}.wav" in node_source
    assert "audio_meta_{active_run.index}.json" in node_source
    assert "manifest_{active_run.index}.json" in node_source
    assert "started_at" in node_source
    assert "stopped_at" in node_source
    assert "frame_rate" in node_source
    assert "sample_rate_hz" in node_source
    assert "chunks_received" in node_source
    assert "chunks_gap_filled_source" in node_source
    assert "chunks_gap_filled_recorder" in node_source
    assert "samples_written" in node_source
    assert "duration_s" in node_source
    assert "config" in node_source and "bringup" in node_source and "description" in node_source


def test_recording_node_fills_missing_audio_chunks_with_silence():
    node_source = _recording_node_source()

    assert "audio_next_chunk_index" in node_source
    assert "missing_chunks = chunk_index - active_run.audio_next_chunk_index" in node_source
    assert 'array("h", [0]) * len(samples)' in node_source
    assert "Filled {missing_chunks} missing recorder audio chunks with silence" in node_source
    assert "msg.gap_fill" in node_source


def test_recording_node_sequences_pointcloud_and_media_boundaries():
    node_source = _recording_node_source()
    start_source = node_source[
        node_source.index("    def _start_recording_run(self) -> None:")
        : node_source.index("    def _write_rgb_frame")
    ]
    stop_source = node_source[
        node_source.index("    def _stop_recording_run(self) -> None:")
        : node_source.index("    def _close_run_files")
    ]

    assert start_source.index(
        'self._capture_pointcloud(\n                "pointcloud_start"'
    ) < start_source.index("self._active_run = active_run")
    assert start_source.index("self._active_run = active_run") < start_source.index(
        "self._run_count += 1"
    )
    assert "Recording run became active during pointcloud_start" in start_source

    assert stop_source.index("self._active_run = None") < stop_source.index(
        "self._close_run_files(active_run)"
    )
    assert stop_source.index("self._close_run_files(active_run)") < stop_source.index(
        'self._capture_pointcloud(\n                    "pointcloud_stop"'
    )
    assert stop_source.index("except Exception as exc:") < stop_source.index(
        "self._write_run_metadata(active_run, stopped_at)"
    )

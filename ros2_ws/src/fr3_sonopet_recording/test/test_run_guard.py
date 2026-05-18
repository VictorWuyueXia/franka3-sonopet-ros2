import pytest
from fr3_sonopet_recording.run_guard import RecordingRunGuard


def test_recording_guard_rejects_double_start(tmp_path):
    guard = RecordingRunGuard()

    guard.start("run_a", tmp_path / "run_a")

    with pytest.raises(RuntimeError, match="already active"):
        guard.start("run_b", tmp_path / "run_b")


def test_recording_guard_rejects_stop_without_start():
    guard = RecordingRunGuard()

    with pytest.raises(RuntimeError, match="No recording is active"):
        guard.stop("run_a")

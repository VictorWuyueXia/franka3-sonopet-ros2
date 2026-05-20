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

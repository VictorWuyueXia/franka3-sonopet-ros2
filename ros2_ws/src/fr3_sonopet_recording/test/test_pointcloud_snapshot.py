from fr3_sonopet_recording.pointcloud_snapshot import PointCloudSnapshotStateMachine


def test_pointcloud_snapshot_enables_receives_writes_and_disables(tmp_path):
    enabled_states = []

    state_machine = PointCloudSnapshotStateMachine(
        set_enabled=enabled_states.append,
        receive_cloud=lambda timeout: {"timeout": timeout},
        write_cloud=lambda path, cloud: 42,
    )

    result = state_machine.capture("start", tmp_path / "pointcloud_start.pcd", 5.0)

    assert enabled_states == [True, False]
    assert result.success is True
    assert result.point_count == 42
    assert result.path == tmp_path / "pointcloud_start.pcd"


def test_pointcloud_snapshot_disables_after_failure(tmp_path):
    enabled_states = []

    def _receive_cloud(_timeout):
        raise RuntimeError("no pointcloud")

    state_machine = PointCloudSnapshotStateMachine(
        set_enabled=enabled_states.append,
        receive_cloud=_receive_cloud,
        write_cloud=lambda path, cloud: 0,
    )

    result = state_machine.capture("end", tmp_path / "pointcloud_end.pcd", 0.1)

    assert enabled_states == [True, False]
    assert result.success is False
    assert "no pointcloud" in result.error

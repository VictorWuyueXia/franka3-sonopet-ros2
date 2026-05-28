import json
from pathlib import Path

import numpy as np
import pytest
from std_msgs.msg import Header

from fr3_sonopet_pointcloud.cloud_processing import (
    BASE_FRAME,
    camera_dir,
    make_colored_cloud,
    next_existing_name,
    pointcloud_xyz_rgb,
    transform_points,
    trim_sensor_cloud,
    write_colored_pcd,
    write_pointcloud_metadata,
)


def test_trim_policy_uses_distance_and_farthest_fraction():
    xyz = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rgb = np.arange(5, dtype=np.float32)

    trimmed_xyz, trimmed_rgb = trim_sensor_cloud(xyz, rgb, 0.5, 0.25)

    assert np.allclose(np.sort(trimmed_xyz[:, 0]), np.array([0.1, 0.2, 0.3]))
    assert set(trimmed_rgb.tolist()) == {0.0, 1.0, 2.0}


def test_transform_cloud_to_base_frame():
    points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.array([1.0, -1.0, 0.5], dtype=np.float64)

    transformed = transform_points(points, transform)

    assert np.allclose(transformed, points + np.array([1.0, -1.0, 0.5]))


def test_colored_cloud_and_pcd_preserve_rgb(tmp_path):
    points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    rgb = np.array([255.0, 65280.0], dtype=np.float32)

    cloud = make_colored_cloud(points, rgb, Header().stamp)
    xyz, colors = pointcloud_xyz_rgb(cloud)
    pcd_path = tmp_path / "cloud.pcd"
    count = write_colored_pcd(pcd_path, xyz, colors)

    assert cloud.header.frame_id == BASE_FRAME
    assert count == 2
    assert np.allclose(xyz, points)
    assert np.allclose(colors, rgb)
    pcd = pcd_path.read_text(encoding="utf-8")
    assert "FIELDS x y z rgb" in pcd
    assert "0.1 0.2 0.3 255" in pcd


def test_pointcloud_reader_rejects_empty_frame():
    cloud = make_colored_cloud(
        np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        np.array([255.0], dtype=np.float32),
        Header().stamp,
    )
    cloud.header.frame_id = ""

    with pytest.raises(RuntimeError, match="frame_id"):
        pointcloud_xyz_rgb(cloud)


def test_duplicate_pointcloud_names_follow_recording_convention(tmp_path):
    base = camera_dir(tmp_path, "in_hand") / "pointcloud_start.pcd"
    base.parent.mkdir(parents=True)
    base.write_text("", encoding="utf-8")
    base.with_name("pointcloud_start_1.pcd").write_text("", encoding="utf-8")

    candidate = next_existing_name(base)

    assert candidate.name == "pointcloud_start_2.pcd"


def test_pointcloud_metadata_records_config_and_timestamps(tmp_path):
    metadata = tmp_path / "pointcloud_meta.json"
    captures = [
        {
            "camera": "in_hand",
            "label": "pointcloud_start",
            "path": "camera_in_hand/pointcloud_start.pcd",
            "point_count": 2,
            "timestamp": 1_700_000_000.0,
            "timestamp_local": "202605271551",
        }
    ]

    write_pointcloud_metadata(metadata, 0.5, 0.2, captures)
    payload = json.loads(metadata.read_text(encoding="utf-8"))

    assert payload["frame_id"] == BASE_FRAME
    assert payload["trim_distance_m"] == 0.5
    assert payload["trim_farthest_fraction"] == 0.2
    assert payload["captures"] == captures


def test_node_source_rejects_stale_vendor_state_and_owns_capture_action():
    package_root = Path(__file__).resolve().parents[1]
    node = (
        package_root / "src" / "fr3_sonopet_pointcloud" / "pointcloud_node.py"
    ).read_text(encoding="utf-8")
    action = (
        package_root.parent / "fr3_sonopet_interfaces" / "action" / "CapturePointCloud.action"
    ).read_text(encoding="utf-8")

    assert 'CAPTURE_ACTION = "/realsense/capture_pointcloud"' in node
    assert 'VENDOR_STALE_TOPIC = "/sonopet/vendor_joint_state_stale"' in node
    assert "VENDOR_LIVE_MAX_AGE_S = 0.5" in node
    assert 'STALE_MESSAGE = "Cannot capture pointcloud: cannot get joint state"' in node
    assert "goal_handle.abort()" in node
    assert "ActionServer(" in node
    assert "string artifact_root" in action
    assert "float64[] timestamps" in action
    assert "string[] timestamp_locals" in action

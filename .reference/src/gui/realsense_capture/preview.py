from __future__ import annotations

from pathlib import Path

from .constants import OVERLAY_TEXT
from .stream import build_display, log_status, require_cv2, require_open3d


def write_pointcloud(path: Path, pointcloud) -> None:
    o3d = require_open3d()
    o3d.io.write_point_cloud(str(path), pointcloud)


def show_pointcloud(pointcloud, title: str) -> None:
    o3d = require_open3d()
    log_status(title)
    geometries = [pointcloud, _build_coordinate_frame(o3d, pointcloud)]
    o3d.visualization.draw_geometries(geometries)


def _build_coordinate_frame(o3d, pointcloud):
    extent = pointcloud.get_axis_aligned_bounding_box().get_extent()
    frame_size = max(float(max(extent)) * 0.15, 0.05)
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0.0, 0.0, 0.0])


def build_display_panel(camera_name: str, serial_number: str, color_image, depth_image):
    cv2 = require_cv2()
    panel = build_display(color_image, depth_image)
    cv2.putText(
        panel,
        camera_name,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        f"serial={serial_number}",
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        OVERLAY_TEXT,
        (10, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return panel

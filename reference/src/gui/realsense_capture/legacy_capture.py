from __future__ import annotations

import time

from .constants import REPO_ROOT, WINDOW_NAME
from .pointcloud.storage import PointCloudStorage
from .preview import build_display_panel, show_pointcloud, write_pointcloud
from .stream import DEFAULT_WARMUP_FRAMES, RealSenseStream, log_status, require_cv2


def run_legacy_single_capture(
    serial: str,
    width: int,
    height: int,
    fps: int,
    clip_distance_max: float,
    label: str,
) -> None:
    cv2 = require_cv2()
    storage = PointCloudStorage(REPO_ROOT)
    storage.ensure_directories()
    stream = RealSenseStream(
        camera_name=str(label),
        serial_request=str(serial),
        width=int(width),
        height=int(height),
        fps=int(fps),
        clip_distance_max=float(clip_distance_max),
    )
    stream.start()
    stream.warm_up(DEFAULT_WARMUP_FRAMES)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 360)
    try:
        while True:
            bundle = stream.poll_frame()
            if bundle is None:
                continue
            panel = build_display_panel(
                camera_name=str(label),
                serial_number=str(stream.info["serial"]),
                color_image=bundle.color_image,
                depth_image=bundle.depth_image,
            )
            cv2.imshow(WINDOW_NAME, panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log_status("Quitting stream.")
                return
            if key == ord("c"):
                _capture_legacy_pointcloud(storage, stream, label)
    except KeyboardInterrupt:
        log_status("Interrupted by keyboard.")
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def _capture_legacy_pointcloud(storage: PointCloudStorage, stream: RealSenseStream, label: str) -> None:
    pointcloud = stream.capture_pointcloud(None)
    if pointcloud is None or len(pointcloud.points) <= 0:
        log_status("Capture failed because no valid point cloud was returned.")
        return
    raw_path = storage.build_scan_path(str(label), int(time.time()))
    write_pointcloud(raw_path, pointcloud)
    log_status(f"Saved point cloud to {raw_path}")
    show_pointcloud(pointcloud, "Close the point-cloud viewer to resume streaming.")

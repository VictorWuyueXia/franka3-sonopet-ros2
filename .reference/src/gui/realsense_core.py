from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.realsense_capture.stream import (
    DEFAULT_CLIP_DISTANCE_M,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WARMUP_FRAMES,
    DEFAULT_WIDTH,
    FrameBundle,
    RealSenseStream,
    apply_filters,
    build_display,
    build_filter_pipeline,
    log_status,
    make_intrinsic,
    pointcloud_from_frames,
    remove_outliers,
    require_cv2,
    require_open3d,
    require_realsense,
)

__all__ = [
    "DEFAULT_CLIP_DISTANCE_M",
    "DEFAULT_FPS",
    "DEFAULT_HEIGHT",
    "DEFAULT_WARMUP_FRAMES",
    "DEFAULT_WIDTH",
    "FrameBundle",
    "RealSenseStream",
    "apply_filters",
    "build_display",
    "build_filter_pipeline",
    "log_status",
    "make_intrinsic",
    "pointcloud_from_frames",
    "remove_outliers",
    "require_cv2",
    "require_open3d",
    "require_realsense",
]

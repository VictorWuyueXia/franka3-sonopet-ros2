from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from realsense_boot import DEFAULT_D405_PRESET_PATH, format_device_label, start_configured_pipeline

DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_CLIP_DISTANCE_M = 2.0
DEFAULT_WARMUP_FRAMES = 30
DEFAULT_MERGE_DISTANCE_TRUNCATION_M = 0.5


def log_status(message: str) -> None:
    print(f"[RealSense] {message}")


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python is required for streaming and capture.") from exc
    return cv2


def require_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("pyrealsense2 is required for RealSense streaming and capture.") from exc
    return rs


def require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("open3d is required for point cloud capture and visualization.") from exc
    return o3d


def is_device_busy_error(exc: Exception) -> bool:
    message = str(exc)
    return "Device or resource busy" in message or "errno=16" in message


def build_stream_start_error(camera_name: str, exc: Exception) -> RuntimeError:
    if is_device_busy_error(exc):
        return RuntimeError(
            f"{camera_name}: failed to start RealSense stream because the device is busy. "
            "Stop any other RealSense app, ROS camera node, or experiment GUI using this camera, then retry."
        )
    return RuntimeError(f"{camera_name}: failed to start RealSense stream ({exc})")


def build_filter_pipeline():
    """
    Returns a filter pipeline that operates in disparity space.

    This keeps short-range D405 geometry stable near depth edges.
    """

    rs = require_realsense()
    to_disp = rs.disparity_transform(True)
    from_disp = rs.disparity_transform(False)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 8)
    spatial.set_option(rs.option.holes_fill, 0)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    hole_fill = rs.hole_filling_filter()
    hole_fill.set_option(rs.option.holes_fill, 1)
    return [to_disp, spatial, temporal, from_disp, hole_fill]


def apply_filters(depth_frame, filters):
    frame = depth_frame
    for filter_stage in filters:
        frame = filter_stage.process(frame)
    return frame


def make_intrinsic(intrinsics):
    o3d = require_open3d()
    return o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width,
        intrinsics.height,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.ppx,
        intrinsics.ppy,
    )


def remove_outliers(pcd, nb_neighbors: int, std_ratio: float, radius: float, min_points: int):
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    pcd, _ = pcd.remove_radius_outlier(
        nb_points=min_points,
        radius=radius,
    )
    return pcd


def truncate_depth_image(depth_np: np.ndarray, max_distance_m: float) -> tuple[np.ndarray, int, int]:
    truncated = np.array(depth_np, copy=True)
    valid_before = int(np.count_nonzero(truncated > 0.0))
    truncated[truncated > float(max_distance_m)] = 0.0
    valid_after = int(np.count_nonzero(truncated > 0.0))
    return truncated, valid_before, valid_after


def pointcloud_from_frames(
    depth_frame,
    color_frame,
    intrinsics,
    depth_scale: float,
    clip_distance_max: float,
    merge_distance_truncation_m: float | None = None,
    camera_name: str | None = None,
):
    """Build an Open3D point cloud from aligned RealSense frames."""

    cv2 = require_cv2()
    o3d = require_open3d()
    if not depth_frame or not color_frame:
        return None

    depth_np = np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(depth_scale)
    if merge_distance_truncation_m is not None:
        depth_np, valid_before, valid_after = truncate_depth_image(depth_np, float(merge_distance_truncation_m))
        if valid_after != valid_before:
            label = str(camera_name or "pointcloud")
            log_status(
                f"{label}: depth truncation @ {float(merge_distance_truncation_m):.3f} m: "
                f"{valid_before:,} -> {valid_after:,} valid depth pixels "
                f"({valid_before - valid_after:,} removed)"
            )
    color_np = np.asanyarray(color_frame.get_data())
    color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_np),
        o3d.geometry.Image(depth_np),
        depth_scale=1.0,
        depth_trunc=float(clip_distance_max),
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, make_intrinsic(intrinsics))
    if len(pcd.points) <= 0:
        return pcd

    before = len(pcd.points)
    pcd = remove_outliers(
        pcd=pcd,
        nb_neighbors=30,
        std_ratio=1.5,
        radius=0.02,
        min_points=15,
    )
    log_status(
        f"Outlier removal: {before:,} -> {len(pcd.points):,} points "
        f"({before - len(pcd.points):,} removed)"
    )
    return pcd


def build_display(color_image, depth_image):
    cv2 = require_cv2()
    depth_float = depth_image.astype(np.float32)
    valid = depth_float > 0
    if valid.any():
        d_min = float(depth_float[valid].min())
        d_max = float(depth_float[valid].max())
        depth_norm = np.zeros_like(depth_float)
        depth_norm[valid] = (depth_float[valid] - d_min) / (d_max - d_min + 1e-6) * 255.0
    else:
        depth_norm = np.zeros_like(depth_float)

    depth_colormap = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
    return np.hstack((color_image, depth_colormap))


@dataclass
class FrameBundle:
    color_frame: object
    filtered_depth_frame: object
    color_image: np.ndarray
    depth_image: np.ndarray
    capture_timestamp_ms: float


class RealSenseStream:
    """Single-camera stream wrapper reused by the dual-camera entrypoint."""

    def __init__(
        self,
        camera_name: str,
        serial_request: str,
        width: int,
        height: int,
        fps: int,
        clip_distance_max: float,
    ) -> None:
        self.camera_name = str(camera_name)
        self.serial_request = str(serial_request)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.clip_distance_max = float(clip_distance_max)
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.intrinsics = None
        self.info = None
        self.preset_path = None
        self.filters = build_filter_pipeline()
        self.last_bundle: FrameBundle | None = None

    def start(self) -> None:
        try:
            (
                self.pipeline,
                self.align,
                self.depth_scale,
                self.intrinsics,
                self.info,
                self.preset_path,
            ) = start_configured_pipeline(
                self.width,
                self.height,
                self.fps,
                self.serial_request,
                DEFAULT_D405_PRESET_PATH,
            )
        except Exception as exc:
            raise build_stream_start_error(self.camera_name, exc) from exc
        log_status(f"{self.camera_name}: using device {format_device_label(self.info)}")
        log_status(f"{self.camera_name}: loaded preset {self.preset_path}")

    def warm_up(self, frame_count: int) -> None:
        if self.pipeline is None:
            raise RuntimeError(f"{self.camera_name}: pipeline is not started")
        log_status(f"{self.camera_name}: warming up for {int(frame_count)} frames")
        for _ in range(int(frame_count)):
            self.pipeline.wait_for_frames()

    def poll_frame(self) -> FrameBundle | None:
        if self.pipeline is None or self.align is None:
            raise RuntimeError(f"{self.camera_name}: pipeline is not started")
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None
        filtered_depth = apply_filters(depth_frame, self.filters)
        bundle = FrameBundle(
            color_frame=color_frame,
            filtered_depth_frame=filtered_depth,
            color_image=np.asanyarray(color_frame.get_data()),
            depth_image=np.asanyarray(filtered_depth.get_data()),
            capture_timestamp_ms=float(depth_frame.get_timestamp()),
        )
        self.last_bundle = bundle
        return bundle

    def capture_pointcloud(self, frame_bundle: FrameBundle | None, merge_distance_truncation_m: float | None = None):
        bundle = frame_bundle if frame_bundle is not None else self.last_bundle
        if bundle is None:
            return None
        if self.intrinsics is None or self.depth_scale is None:
            raise RuntimeError(f"{self.camera_name}: pipeline intrinsics are unavailable")
        return pointcloud_from_frames(
            depth_frame=bundle.filtered_depth_frame,
            color_frame=bundle.color_frame,
            intrinsics=self.intrinsics,
            depth_scale=float(self.depth_scale),
            clip_distance_max=float(self.clip_distance_max),
            merge_distance_truncation_m=merge_distance_truncation_m,
            camera_name=self.camera_name,
        )

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.intrinsics = None
        self.info = None
        self.preset_path = None
        self.last_bundle = None

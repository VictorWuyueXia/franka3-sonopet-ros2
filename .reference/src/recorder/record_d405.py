#!/usr/bin/env python3
"""
Intel RealSense D405 recorder.

- Captures a base-frame pointcloud scan at the start.
- Streams color to rgb.avi with per-frame machine timestamps in a CSV.
- Optional --depth-video: aligned depth stream to depth.mp4 (8-bit scaled mm).
- Captures a base-frame pointcloud scan at the end.

Stops on SIGINT/SIGTERM. Designed to be launched by launcher.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.realsense_capture.constants import BASE_FRAME_NAME
from gui.realsense_capture.pointcloud.alignment import resolve_base_alignment
from gui.realsense_capture.pointcloud.transforms import transform_pointcloud
from gui.realsense_capture.preview import write_pointcloud
from gui.realsense_capture.robot_pose import RobotTcpPoseResolver
from gui.realsense_capture.roles import build_camera_role
from gui.realsense_capture.stream import (
    DEFAULT_CLIP_DISTANCE_M,
    DEFAULT_HEIGHT,
    DEFAULT_MERGE_DISTANCE_TRUNCATION_M,
    DEFAULT_WARMUP_FRAMES,
    DEFAULT_WIDTH,
    RealSenseStream,
)


WIDTH = DEFAULT_WIDTH
HEIGHT = DEFAULT_HEIGHT

# Linear map for depth preview video (mm); outside range clips to 0–255.
DEPTH_VIDEO_MM_LO = 200.0
DEPTH_VIDEO_MM_HI = 6000.0


def log_d405(message: str) -> None:
    print(f"[d405] {message}", flush=True)


def depth_u16_to_bgr_u8(depth: np.ndarray) -> np.ndarray:
    """uint16 depth (mm, 0 = invalid) -> BGR uint8 for VideoWriter."""

    d = depth.astype(np.float32)
    valid = d > 0
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        scaled = np.clip(
            (d[valid] - DEPTH_VIDEO_MM_LO) / (DEPTH_VIDEO_MM_HI - DEPTH_VIDEO_MM_LO),
            0.0,
            1.0,
        )
        gray[valid] = (scaled * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def build_camera_trim_m(args) -> tuple[float, float, float]:
    return (
        float(args.camera_in_hand_parent_x_trim_mm) / 1000.0,
        float(args.camera_in_hand_parent_y_trim_mm) / 1000.0,
        float(args.camera_in_hand_parent_z_trim_mm) / 1000.0,
    )


def build_scan_metadata(
    tag: str,
    serial: str,
    camera_role: str,
    pointcloud_path: Path,
    raw_point_count: int,
    base_point_count: int,
    base_alignment,
) -> dict:
    ts = time.time()
    robot_capture = base_alignment.robot_capture
    return {
        "tag": str(tag),
        "serial": str(serial),
        "camera_role": str(camera_role),
        "timestamp_unix": float(ts),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
        "pointcloud_path": str(pointcloud_path.name),
        "pointcloud_frame": BASE_FRAME_NAME,
        "raw_point_count": int(raw_point_count),
        "base_point_count": int(base_point_count),
        "merge_distance_truncation_m": float(DEFAULT_MERGE_DISTANCE_TRUNCATION_M),
        "base_transform": base_alignment.base_transform.to_dict(),
        "robot_capture": None
        if robot_capture is None
        else {
            "robot_ip": robot_capture.robot_ip,
            "q_current": robot_capture.q_current,
            "base_to_parent_frame": robot_capture.base_to_tcp.to_dict(),
        },
    }


def save_pointcloud_scan(
    stream: RealSenseStream,
    frame_bundle,
    role_config,
    robot_pose_resolver: RobotTcpPoseResolver,
    out_dir: Path,
    tag: str,
    camera_trim_m: tuple[float, float, float],
    camera_yaw_trim_deg: float,
) -> None:
    """Save one aligned pointcloud scan in the robot base frame."""

    log_d405(f"capturing {tag} pointcloud scan for role={role_config.camera_name}")
    raw_cloud = stream.capture_pointcloud(
        frame_bundle,
        merge_distance_truncation_m=float(DEFAULT_MERGE_DISTANCE_TRUNCATION_M),
    )
    if raw_cloud is None or len(raw_cloud.points) <= 0:
        raise RuntimeError(f"{tag} pointcloud scan is empty")
    base_alignment = resolve_base_alignment(
        role_config=role_config,
        robot_pose_resolver=robot_pose_resolver,
        robot_capture_by_frame={},
        camera_in_hand_parent_trim_m=camera_trim_m,
        camera_in_hand_parent_yaw_trim_deg=float(camera_yaw_trim_deg),
    )
    if base_alignment.base_transform is None:
        raise RuntimeError(base_alignment.skip_reason or f"Missing transform into {BASE_FRAME_NAME}")
    base_cloud = transform_pointcloud(raw_cloud, base_alignment.base_transform)
    pointcloud_path = out_dir / f"pointcloud_{tag}.pcd"
    write_pointcloud(pointcloud_path, base_cloud)
    metadata = build_scan_metadata(
        tag,
        str(role_config.serial),
        str(role_config.camera_name),
        pointcloud_path,
        len(raw_cloud.points),
        len(base_cloud.points),
        base_alignment,
    )
    (out_dir / f"pointcloud_{tag}_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    log_d405(f"captured {tag} pointcloud -> {pointcloud_path}")


def open_video_writer(path: Path, fps: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, int(fps), (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {path}")
    return writer


def open_timestamp_csv(path: Path, header: list[str]):
    csv_file = open(path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    return csv_file, csv_writer


class D405Recorder:
    """Record RGB video and base-frame pointcloud scans from one D405."""

    def __init__(self, args) -> None:
        self.args = args
        self.fps = max(1, int(args.fps))
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.role_config = build_camera_role(str(args.camera_role), str(args.serial))
        self.robot_pose_resolver = RobotTcpPoseResolver(args.robot_ip)
        self.stream = RealSenseStream(
            camera_name=str(args.camera_role),
            serial_request=str(args.serial),
            width=WIDTH,
            height=HEIGHT,
            fps=self.fps,
            clip_distance_max=float(DEFAULT_CLIP_DISTANCE_M),
        )
        self.rgb_path = self.out_dir / "rgb.avi"
        self.depth_path = self.out_dir / "depth.mp4"
        self.writer = None
        self.ts_csv = None
        self.csv_w = None
        self.depth_writer = None
        self.depth_ts_csv = None
        self.depth_csv_w = None
        self.stop = {"flag": False}
        self.last_bundle = None
        self.frame_idx = 0
        self.scan_started = False

    def run(self) -> int:
        self._install_signal_handlers()
        try:
            self._start_stream_and_scan()
            self._open_recording_outputs()
            self._record_loop()
        finally:
            self._close_recording_outputs()
            try:
                self._save_end_scan()
            finally:
                self._stop_streams()
            self._log_summary()
        return 0

    def _install_signal_handlers(self) -> None:
        def _handle(signum, _frame):
            log_d405(f"received signal {signum}, stopping")
            self.stop["flag"] = True

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

    def _start_stream_and_scan(self) -> None:
        self.stream.start()
        serial = str(self.stream.info["serial"])
        log_d405(
            f"started, serial={serial}, role={self.args.camera_role}, "
            f"fps={self.fps}, depth_video={self.args.depth_video}"
        )
        self.stream.warm_up(DEFAULT_WARMUP_FRAMES)
        start_bundle = self.stream.poll_frame()
        self._save_scan("start", start_bundle)
        self.scan_started = True

    def _save_scan(self, tag: str, frame_bundle) -> None:
        save_pointcloud_scan(
            self.stream,
            frame_bundle,
            self.role_config,
            self.robot_pose_resolver,
            self.out_dir,
            tag,
            build_camera_trim_m(self.args),
            float(self.args.camera_in_hand_parent_yaw_trim_deg),
        )

    def _open_recording_outputs(self) -> None:
        self.writer = open_video_writer(self.rgb_path, self.fps)
        log_d405(f"writing rgb -> {self.rgb_path}")
        self.ts_csv, self.csv_w = open_timestamp_csv(
            self.out_dir / "rgb_timestamps.csv",
            ["frame_index", "video_file", "unix_time", "rs_frame_timestamp_ms"],
        )
        if not self.args.depth_video:
            return
        self.depth_writer = open_video_writer(self.depth_path, self.fps)
        log_d405(f"writing depth -> {self.depth_path}")
        self.depth_ts_csv, self.depth_csv_w = open_timestamp_csv(
            self.out_dir / "depth_timestamps.csv",
            ["frame_index", "video_file", "unix_time", "rs_frame_timestamp_ms"],
        )

    def _record_loop(self) -> None:
        while not self.stop["flag"]:
            bundle = self.stream.poll_frame()
            if bundle is None:
                continue
            self._write_frame(bundle)
            self.frame_idx += 1
            self.last_bundle = bundle

    def _write_frame(self, bundle) -> None:
        self.writer.write(bundle.color_image)
        self.csv_w.writerow(
            [
                self.frame_idx,
                self.rgb_path.name,
                f"{time.time():.6f}",
                f"{bundle.capture_timestamp_ms:.3f}",
            ]
        )
        if not self.args.depth_video:
            return
        self.depth_writer.write(depth_u16_to_bgr_u8(bundle.depth_image))
        self.depth_csv_w.writerow(
            [
                self.frame_idx,
                self.depth_path.name,
                f"{time.time():.6f}",
                f"{bundle.capture_timestamp_ms:.3f}",
            ]
        )

    def _close_recording_outputs(self) -> None:
        if self.writer is not None:
            self.writer.release()
        if self.ts_csv is not None:
            self.ts_csv.close()
        if self.depth_writer is not None:
            self.depth_writer.release()
        if self.depth_ts_csv is not None:
            self.depth_ts_csv.close()

    def _save_end_scan(self) -> None:
        if not self.scan_started:
            return
        final_bundle = self.last_bundle if self.last_bundle is not None else self.stream.poll_frame()
        self._save_scan("end", final_bundle)

    def _stop_streams(self) -> None:
        self.stream.stop()
        self.robot_pose_resolver.close()

    def _log_summary(self) -> None:
        msg = f"wrote {self.frame_idx} rgb frames -> {self.rgb_path.name}"
        if self.args.depth_video:
            msg += f", depth -> {self.depth_path.name}"
        log_d405(msg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--serial", default="", help="Optional D405 serial number")
    parser.add_argument("--camera-role", required=True, help="Configured role: camera-fixed or camera-in-hand")
    parser.add_argument("--robot-ip", default="172.16.0.2", help="Robot IP used for camera-in-hand base alignment")
    parser.add_argument("--fps", type=int, default=15, help="Color/depth frame rate (default 10)")
    parser.add_argument(
        "--depth-video",
        action="store_true",
        help="Record aligned depth as depth.mp4 alongside rgb.mp4",
    )
    parser.add_argument("--camera-in-hand-parent-x-trim-mm", type=float, default=0.0)
    parser.add_argument("--camera-in-hand-parent-y-trim-mm", type=float, default=0.0)
    parser.add_argument("--camera-in-hand-parent-z-trim-mm", type=float, default=5.0)
    parser.add_argument("--camera-in-hand-parent-yaw-trim-deg", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    return D405Recorder(parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())

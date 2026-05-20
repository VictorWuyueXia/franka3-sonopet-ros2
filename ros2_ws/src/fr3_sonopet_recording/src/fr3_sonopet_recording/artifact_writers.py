from __future__ import annotations

import csv
import json
import wave
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sensor_msgs_py import point_cloud2


@dataclass(frozen=True)
class SnapshotResult:
    label: str
    path: Path
    success: bool
    point_count: int = 0
    error: str = ""


class RgbVideoRecorder:
    """Write a camera RGB stream to an AVI file and a timestamp sidecar."""

    def __init__(self, video_path: Path, timestamp_path: Path, fps: float) -> None:
        self.video_path = video_path
        self.timestamp_path = timestamp_path
        self.fps = float(fps)
        self.frame_count = 0
        self._writer: cv2.VideoWriter | None = None
        self._csv_file = timestamp_path.open("w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["frame_index", "stamp_sec", "stamp_nanosec"])

    def write_frame(self, frame_bgr: np.ndarray, stamp_sec: int, stamp_nanosec: int) -> None:
        if self._writer is None:
            height, width = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open RGB video writer: {self.video_path}")

        self._writer.write(frame_bgr)
        self._csv.writerow([self.frame_count, int(stamp_sec), int(stamp_nanosec)])
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._csv_file.close()


class AudioWavRecorder:
    """Write AudioChunk samples to a PCM WAV file."""

    def __init__(self, wav_path: Path) -> None:
        self.wav_path = wav_path
        self.chunk_count = 0
        self.sample_count = 0
        self.sample_rate_hz: int | None = None
        self.channels: int | None = None
        self.encoding = "S16_LE"
        self._wav: wave.Wave_write | None = None

    def write_samples(
        self,
        samples: list[int],
        sample_rate_hz: int,
        channels: int,
        encoding: str,
    ) -> None:
        if encoding != "S16_LE":
            raise ValueError(f"Unsupported audio encoding: {encoding}")
        if len(samples) % channels != 0:
            raise ValueError("Audio sample count must be divisible by channel count")

        if self._wav is None:
            self.sample_rate_hz = int(sample_rate_hz)
            self.channels = int(channels)
            self.encoding = encoding
            self._wav = wave.open(str(self.wav_path), "wb")
            self._wav.setnchannels(self.channels)
            self._wav.setsampwidth(2)
            self._wav.setframerate(self.sample_rate_hz)
        elif self.sample_rate_hz != int(sample_rate_hz) or self.channels != int(channels):
            raise ValueError("Audio format changed during recording")

        pcm = array("h", (int(value) for value in samples))
        self._wav.writeframes(pcm.tobytes())
        self.chunk_count += 1
        self.sample_count += len(samples)

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None

    def metadata(self) -> dict[str, Any]:
        duration = 0.0
        if self.sample_rate_hz and self.channels:
            duration = self.sample_count / float(self.sample_rate_hz * self.channels)
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "encoding": self.encoding,
            "chunks": self.chunk_count,
            "samples": self.sample_count,
            "duration_s": duration,
        }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_pointcloud_pcd(path: Path, cloud_msg) -> int:
    """Write XYZ fields from a PointCloud2 message to an ASCII PCD artifact."""

    raw_points = point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    points: list[tuple[float, float, float]] = []
    for point in raw_points:
        if hasattr(point, "dtype") and point.dtype.names:
            xyz = (float(point["x"]), float(point["y"]), float(point["z"]))
        else:
            xyz = (float(point[0]), float(point[1]), float(point[2]))
        points.append(xyz)

    with path.open("w", encoding="utf-8") as stream:
        stream.write("# .PCD v0.7 - Point Cloud Data file format\n")
        stream.write("VERSION 0.7\n")
        stream.write("FIELDS x y z\n")
        stream.write("SIZE 4 4 4\n")
        stream.write("TYPE F F F\n")
        stream.write("COUNT 1 1 1\n")
        stream.write(f"WIDTH {len(points)}\n")
        stream.write("HEIGHT 1\n")
        stream.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        stream.write(f"POINTS {len(points)}\n")
        stream.write("DATA ascii\n")
        for x, y, z in points:
            stream.write(f"{x:.9g} {y:.9g} {z:.9g}\n")
    return len(points)


def capture_pointcloud_snapshot(
    label: str,
    path: Path,
    timeout_sec: float,
    set_enabled: Callable[[bool], None],
    receive_cloud: Callable[[float], Any],
) -> SnapshotResult:
    """Enable the RealSense pointcloud stream only around one PCD capture."""

    result: SnapshotResult | None = None
    try:
        set_enabled(True)
        point_count = write_pointcloud_pcd(path, receive_cloud(timeout_sec))
        result = SnapshotResult(label=label, path=path, success=True, point_count=point_count)
    except Exception as exc:
        result = SnapshotResult(label=label, path=path, success=False, error=str(exc))
    finally:
        try:
            set_enabled(False)
        except Exception as exc:
            if result is None or result.success:
                result = SnapshotResult(label=label, path=path, success=False, error=str(exc))
            else:
                result = SnapshotResult(
                    label=label,
                    path=path,
                    success=False,
                    error=f"{result.error}; disable failed: {exc}",
                )
    return result

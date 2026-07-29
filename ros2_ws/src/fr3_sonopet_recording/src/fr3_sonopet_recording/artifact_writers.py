from __future__ import annotations

import csv
import json
import wave
from array import array
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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

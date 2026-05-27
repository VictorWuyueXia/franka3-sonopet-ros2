from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraRecordingSpec:
    key: str
    rgb_topic: str
    pointcloud_topic: str
    parameter_service: str


@dataclass(frozen=True)
class RecordingConfig:
    artifact_root: Path
    audio_topic: str
    cutting_topic: str
    video_fps: float
    snapshot_timeout_sec: float
    cameras: tuple[CameraRecordingSpec, ...]


def _require_absolute_topic(topic: str) -> str:
    if not topic.startswith("/"):
        raise ValueError(f"Recording topic must be absolute: {topic}")
    return topic


def _camera_from_payload(key: str, payload: dict[str, Any]) -> CameraRecordingSpec:
    return CameraRecordingSpec(
        key=key,
        rgb_topic=_require_absolute_topic(str(payload["rgb_topic"])),
        pointcloud_topic=_require_absolute_topic(str(payload["pointcloud_topic"])),
        parameter_service=_require_absolute_topic(str(payload["parameter_service"])),
    )


def load_recording_config(recording_path: str | Path, camera_path: str | Path) -> RecordingConfig:
    """Load only the recording values that remain intentionally configurable."""

    recording_payload = yaml.safe_load(Path(recording_path).read_text(encoding="utf-8"))
    camera_payload = yaml.safe_load(Path(camera_path).read_text(encoding="utf-8"))
    recording = recording_payload["recording"]
    audio = recording["audio"]
    cameras = recording["cameras"]

    return RecordingConfig(
        artifact_root=Path(str(recording["artifact_root"])),
        audio_topic=_require_absolute_topic(str(audio["topic"])),
        cutting_topic="/sonopet/cutting",
        video_fps=float(camera_payload["realsense"]["fps"]),
        snapshot_timeout_sec=float(camera_payload["pointcloud_snapshots"]["timeout_sec"]),
        cameras=(
            _camera_from_payload("in_hand", cameras["in_hand"]),
            _camera_from_payload("fixed", cameras["fixed"]),
        ),
    )


def recording_topics(config: RecordingConfig) -> tuple[str, ...]:
    """Return continuously consumed topics for validation and launch-time diagnostics."""

    return tuple([config.audio_topic, config.cutting_topic, *(camera.rgb_topic for camera in config.cameras)])

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraRecordingSpec:
    key: str
    directory: str
    rgb_topic: str
    pointcloud_topic: str
    parameter_service: str


@dataclass(frozen=True)
class SnapshotConfig:
    enabled: bool
    timeout_sec: float
    output_format: str
    start_label: str
    end_label: str
    require_success: bool


@dataclass(frozen=True)
class RecordingConfig:
    artifact_root: Path
    audio_topic: str
    audio_directory: str
    video_codec: str
    video_fps: float
    cameras: tuple[CameraRecordingSpec, ...]
    snapshots: SnapshotConfig


def _require_absolute_topic(topic: str) -> str:
    if not topic.startswith("/"):
        raise ValueError(f"Recording topic must be absolute: {topic}")
    return topic


def _camera_from_payload(key: str, payload: dict[str, Any]) -> CameraRecordingSpec:
    return CameraRecordingSpec(
        key=key,
        directory=str(payload["directory"]),
        rgb_topic=_require_absolute_topic(str(payload["rgb_topic"])),
        pointcloud_topic=_require_absolute_topic(str(payload["pointcloud_topic"])),
        parameter_service=_require_absolute_topic(str(payload["parameter_service"])),
    )


def load_recording_config(path: str | Path) -> RecordingConfig:
    """Load recording artifact policy from the bringup YAML configuration."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    recording = payload["recording"]
    video = recording["rgb_video"]
    audio = recording["audio"]
    snapshots = recording["pointcloud_snapshots"]
    cameras = recording["cameras"]

    return RecordingConfig(
        artifact_root=Path(str(recording["artifact_root"])),
        audio_topic=_require_absolute_topic(str(audio["topic"])),
        audio_directory=str(audio["directory"]),
        video_codec=str(video["codec"]),
        video_fps=float(video["fps"]),
        cameras=(
            _camera_from_payload("in_hand", cameras["in_hand"]),
            _camera_from_payload("fixed", cameras["fixed"]),
        ),
        snapshots=SnapshotConfig(
            enabled=bool(snapshots["enabled"]),
            timeout_sec=float(snapshots["timeout_sec"]),
            output_format=str(snapshots["output_format"]),
            start_label=str(snapshots["start_label"]),
            end_label=str(snapshots["end_label"]),
            require_success=bool(snapshots["require_success"]),
        ),
    )


def recording_topics(config: RecordingConfig) -> tuple[str, ...]:
    """Return continuously consumed topics for validation and launch-time diagnostics."""

    return tuple([config.audio_topic, *(camera.rgb_topic for camera in config.cameras)])

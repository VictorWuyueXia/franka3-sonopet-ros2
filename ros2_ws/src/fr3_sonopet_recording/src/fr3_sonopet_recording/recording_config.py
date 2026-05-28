from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def wall_clock_timestamp() -> float:
    """Return local wall-clock epoch seconds aligned with Sonopet JSONL timestamps."""
    return time.time()


def local_timestamp_label(epoch_seconds: float) -> str:
    """Return a compact local wall-clock label, for example 202605271551."""
    return datetime.fromtimestamp(epoch_seconds).astimezone().strftime("%Y%m%d%H%M")


def artifact_root() -> Path:
    """Return the frozen experiment artifact directory under the ROS workspace."""
    return Path(os.environ["FR3_SONOPET_REPO"]) / "ros2_ws" / "artifacts" / "experiments"


def experiment_run_id() -> str:
    """Return a local wall-clock session id aligned with operator-visible time."""
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")


def recording_run_label(run_index: int) -> str:
    """Return the manifest run tag for one cutting interval within a session."""
    if run_index < 1:
        raise ValueError(f"Recording run index must be >= 1, got {run_index}")
    return f"run_{run_index}"


@dataclass(frozen=True)
class CameraRecordingSpec:
    key: str
    rgb_topic: str


@dataclass(frozen=True)
class RecordingConfig:
    artifact_root: Path
    audio_topic: str
    cutting_topic: str
    video_fps: float
    cameras: tuple[CameraRecordingSpec, ...]


def _require_absolute_topic(topic: str) -> str:
    if not topic.startswith("/"):
        raise ValueError(f"Recording topic must be absolute: {topic}")
    return topic


def _camera_from_payload(key: str, payload: dict[str, Any]) -> CameraRecordingSpec:
    return CameraRecordingSpec(
        key=key,
        rgb_topic=_require_absolute_topic(str(payload["rgb_topic"])),
    )


def load_recording_config(recording_path: str | Path, camera_path: str | Path) -> RecordingConfig:
    """Load only the recording values that remain intentionally configurable."""

    recording_payload = yaml.safe_load(Path(recording_path).read_text(encoding="utf-8"))
    camera_payload = yaml.safe_load(Path(camera_path).read_text(encoding="utf-8"))
    recording = recording_payload["recording"]
    audio = recording["audio"]
    cameras = recording["cameras"]

    return RecordingConfig(
        artifact_root=artifact_root(),
        audio_topic=_require_absolute_topic(str(audio["topic"])),
        cutting_topic="/sonopet/cutting",
        video_fps=float(camera_payload["realsense"]["fps"]),
        cameras=(
            _camera_from_payload("in_hand", cameras["in_hand"]),
            _camera_from_payload("fixed", cameras["fixed"]),
        ),
    )


def recording_topics(config: RecordingConfig) -> tuple[str, ...]:
    """Return continuously consumed topics for validation and launch-time diagnostics."""

    return tuple(
        [
            config.audio_topic,
            config.cutting_topic,
            *(camera.rgb_topic for camera in config.cameras),
        ]
    )

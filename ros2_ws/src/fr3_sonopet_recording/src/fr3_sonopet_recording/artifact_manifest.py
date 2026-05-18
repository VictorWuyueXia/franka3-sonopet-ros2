from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RecordingManifest:
    run_id: str
    artifact_path: str
    video_format: str = "avi"
    audio_format: str = "wav"
    pointcloud_format: str = "pcd"


def manifest_dict(manifest: RecordingManifest) -> dict[str, str]:
    return asdict(manifest)

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RecordingManifest:
    run_id: str
    artifact_path: str
    storage_id: str = "mcap"


def manifest_dict(manifest: RecordingManifest) -> dict[str, str]:
    return asdict(manifest)


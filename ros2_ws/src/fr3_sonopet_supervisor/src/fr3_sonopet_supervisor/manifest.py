from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    phase: str
    artifact_root: str


def manifest_dict(manifest: RunManifest) -> dict[str, str]:
    return asdict(manifest)


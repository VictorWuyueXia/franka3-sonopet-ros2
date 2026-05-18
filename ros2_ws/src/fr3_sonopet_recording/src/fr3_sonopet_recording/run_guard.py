from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActiveRun:
    run_id: str
    artifact_path: Path


class RecordingRunGuard:
    """Small state boundary preventing ambiguous recording start and stop requests."""

    def __init__(self) -> None:
        self.active_run: ActiveRun | None = None

    def start(self, run_id: str, artifact_path: Path) -> ActiveRun:
        if self.active_run is not None:
            raise RuntimeError(f"Recording is already active: {self.active_run.run_id}")
        if not run_id.strip():
            raise ValueError("run_id cannot be empty")
        self.active_run = ActiveRun(run_id=run_id, artifact_path=artifact_path)
        return self.active_run

    def stop(self, run_id: str) -> ActiveRun:
        if self.active_run is None:
            raise RuntimeError("No recording is active")
        if run_id.strip() and run_id != self.active_run.run_id:
            raise RuntimeError(
                f"Cannot stop run_id={run_id}; active run is {self.active_run.run_id}"
            )
        stopped = self.active_run
        self.active_run = None
        return stopped

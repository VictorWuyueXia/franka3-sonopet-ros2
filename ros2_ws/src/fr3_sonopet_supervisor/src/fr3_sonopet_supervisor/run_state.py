from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunState:
    run_id: str
    phase: str = "idle"
    ready: bool = False
    blocking_reason: str = ""


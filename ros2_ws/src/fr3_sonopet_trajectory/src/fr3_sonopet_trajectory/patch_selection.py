from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchSelection:
    frame_id: str
    center_xyz: tuple[float, float, float]
    source: str = "rviz_point"


def validate_selection(selection: PatchSelection) -> None:
    if not selection.frame_id:
        raise ValueError("Patch selection must include a frame_id")
    if len(selection.center_xyz) != 3:
        raise ValueError("Patch selection center must contain exactly three values")


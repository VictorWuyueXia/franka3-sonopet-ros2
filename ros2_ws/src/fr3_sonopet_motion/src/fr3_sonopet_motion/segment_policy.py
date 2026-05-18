from __future__ import annotations

DEFAULT_SEGMENTS = (
    "current_to_idle",
    "idle_to_parking",
    "parking_to_first",
    "raster",
    "retract",
    "return_to_start",
)


def validate_segment_order(segments: tuple[str, ...] = DEFAULT_SEGMENTS) -> None:
    if "raster" not in segments:
        raise ValueError("Motion segment policy must include raster")
    if len(set(segments)) != len(segments):
        raise ValueError("Motion segment names must be unique")


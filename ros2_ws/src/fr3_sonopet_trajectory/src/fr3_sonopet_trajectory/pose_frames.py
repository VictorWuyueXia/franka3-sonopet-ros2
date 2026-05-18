from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FramePolicy:
    base_frame: str = "fr3_link0"
    tool_frame: str = "sonopet_tcp"
    planning_cloud_frame: str = "d405_in_hand_color_optical_frame"


def require_base_frame(frame_id: str, policy: FramePolicy) -> None:
    if frame_id != policy.base_frame:
        raise ValueError(f"Expected frame {policy.base_frame}, got {frame_id}")


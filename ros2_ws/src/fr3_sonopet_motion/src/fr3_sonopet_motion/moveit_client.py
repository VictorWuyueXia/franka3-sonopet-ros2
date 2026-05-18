from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoveItPolicy:
    planning_group: str = "fr3_arm"
    base_frame: str = "fr3_link0"
    tool_frame: str = "sonopet_tcp"


def describe_policy(policy: MoveItPolicy) -> str:
    return f"group={policy.planning_group}, base={policy.base_frame}, tool={policy.tool_frame}"


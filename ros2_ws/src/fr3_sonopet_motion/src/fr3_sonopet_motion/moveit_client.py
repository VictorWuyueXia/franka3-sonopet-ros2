from __future__ import annotations

from dataclasses import dataclass


# Immutable configuration describing core MoveIt planning settings
@dataclass(frozen=True)
class MoveItPolicy:
    planning_group: str = "fr3_arm"   # Name of the robot arm/group to use for planning
    base_frame: str = "fr3_link0"     # World/base coordinate frame for reference
    tool_frame: str = "sonopet_tcp"   # Tool/end-effector reference frame

# Returns human-readable summary of planning policy details
def describe_policy(policy: MoveItPolicy) -> str:
    return f"group={policy.planning_group}, base={policy.base_frame}, tool={policy.tool_frame}"

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StitchingPolicy:
    enabled: bool = False
    fixed_cloud_moves: bool = True
    coarse_method: str = "point_to_plane_icp"
    refine_method: str = "colored_icp"


def describe_policy(policy: StitchingPolicy) -> str:
    moving = "fixed_cloud" if policy.fixed_cloud_moves else "in_hand_cloud"
    return f"{moving}:{policy.coarse_method}->{policy.refine_method}"


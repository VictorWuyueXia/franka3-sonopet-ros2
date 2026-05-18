"""CPU partition helpers for runtime thread placement."""

from __future__ import annotations

import os
from typing import Set, Tuple


def get_cpu_partition() -> Tuple[Set[int], Set[int]]:
    """Return (rt_cpu_set, ik_cpu_set) from current allowed CPU set."""

    allowed = sorted(int(cpu_id) for cpu_id in os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("No allowed CPU cores from sched_getaffinity(0)")

    # High-core policy (>=16 allowed cores):
    # - dedicate first 4 cores to realtime communication threads
    # - dedicate most remaining cores to IK/PyBullet work
    # - leave the last 2 cores unpinned so OS/desktop/helper threads can breathe
    if len(allowed) >= 16:
        rt_cpus = set(allowed[:4])
        if len(allowed) > 6:
            ik_pool = allowed[4:-2]
        else:
            ik_pool = allowed[4:]
        ik_cpus = set(ik_pool)
        if not ik_cpus:
            ik_cpus = set(allowed[4:])
        if not ik_cpus:
            ik_cpus = set(allowed)
        return rt_cpus, ik_cpus

    if len(allowed) < 3:
        all_cpus = set(allowed)
        return all_cpus, all_cpus

    rt_cpus = set(allowed[:2])
    ik_cpus = set(allowed[2:])
    return rt_cpus, ik_cpus


def format_cpu_set(cpu_ids: Set[int]) -> str:
    """Format CPU id set as sorted comma-separated string."""

    return ",".join(str(v) for v in sorted(cpu_ids))


"""Compatibility wrappers for affinity helpers.

Core implementations now live in datacollection_franky.util.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from .util.cpu_partition import format_cpu_set, get_cpu_partition
from .util.thread_affinity import (
    cap_all_rt_thread_priorities,
    cap_rt_thread_priorities as _cap_rt_thread_priorities,
    list_process_thread_ids,
    pin_all_rt_threads_to_cpus,
    pin_current_thread_to_cpus,
    pin_rt_threads_to_cpus as _pin_rt_threads_to_cpus,
)


def pin_rt_threads_to_cpus(cpu_ids: Set[int], thread_ids: Optional[Iterable[int]] = None) -> int:
    """Pin RT threads in current process to CPU ids."""

    if thread_ids is None:
        return pin_all_rt_threads_to_cpus(cpu_ids)
    return _pin_rt_threads_to_cpus(cpu_ids, thread_ids)


def cap_rt_thread_priorities(priority_cap: int, thread_ids: Optional[Iterable[int]] = None) -> List[Dict[str, int]]:
    """Lower RT thread priorities to at most priority_cap."""

    if thread_ids is None:
        return cap_all_rt_thread_priorities(priority_cap)
    return _cap_rt_thread_priorities(priority_cap, thread_ids)


def collect_thread_runtime_snapshot() -> Dict[str, object]:
    raise RuntimeError(
        "collect_thread_runtime_snapshot is retired. "
        "Use concise runtime logging and util.thread_affinity helpers instead."
    )


def diff_thread_runtime_snapshots(
    baseline_snapshot: Dict[str, object],
    current_snapshot: Dict[str, object],
    rt_cpu_set: Optional[Set[int]] = None,
) -> Dict[str, object]:
    raise RuntimeError(
        "diff_thread_runtime_snapshots is retired. "
        "Use direct pin-and-cap policy application instead."
    )


def select_new_rt_thread_ids(
    baseline_snapshot: Dict[str, object],
    current_snapshot: Dict[str, object],
) -> List[int]:
    raise RuntimeError(
        "select_new_rt_thread_ids is retired. "
        "Use direct RT-thread scanning in util.thread_affinity."
    )

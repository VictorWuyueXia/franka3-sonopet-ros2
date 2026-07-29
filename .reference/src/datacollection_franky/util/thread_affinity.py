"""Thread-level affinity and RT-priority helpers."""

from __future__ import annotations

import os
import threading
from typing import Dict, Iterable, List, Set


def pin_current_thread_to_cpus(cpu_ids: Set[int]) -> int:
    """Pin current Python thread to provided CPU ids."""

    if not cpu_ids:
        raise ValueError("cpu_ids must be non-empty")
    native_tid = int(threading.get_native_id())
    os.sched_setaffinity(native_tid, cpu_ids)
    return native_tid


def list_process_thread_ids() -> List[int]:
    """List all OS thread ids for current process."""

    tids: List[int] = []
    for entry in os.listdir("/proc/self/task"):
        if entry.isdigit():
            tids.append(int(entry))
    tids.sort()
    return tids


def _iter_existing_thread_ids(thread_ids: Iterable[int]) -> List[int]:
    existing = set(list_process_thread_ids())
    filtered: List[int] = []
    for tid in thread_ids:
        int_tid = int(tid)
        if int_tid in existing:
            filtered.append(int_tid)
    return sorted(set(filtered))


def _iter_all_thread_ids() -> List[int]:
    return list_process_thread_ids()


def _is_rt_policy(policy_id: int) -> bool:
    return int(policy_id) in (int(os.SCHED_FIFO), int(os.SCHED_RR))


def _iter_target_rt_threads(thread_ids: Iterable[int]) -> List[int]:
    targets = _iter_existing_thread_ids(thread_ids)
    out: List[int] = []
    for tid in targets:
        try:
            policy = int(os.sched_getscheduler(tid))
        except ProcessLookupError:
            continue
        if _is_rt_policy(policy):
            out.append(int(tid))
    return out


def _iter_all_rt_threads() -> List[int]:
    out: List[int] = []
    for tid in _iter_all_thread_ids():
        try:
            policy = int(os.sched_getscheduler(tid))
        except ProcessLookupError:
            continue
        if _is_rt_policy(policy):
            out.append(int(tid))
    return out


def pin_rt_threads_to_cpus(cpu_ids: Set[int], thread_ids: Iterable[int]) -> int:
    """Pin specified RT threads in current process to CPU ids."""

    if not cpu_ids:
        raise ValueError("cpu_ids must be non-empty")
    pinned = 0
    for tid in _iter_target_rt_threads(thread_ids):
        try:
            os.sched_setaffinity(tid, cpu_ids)
        except ProcessLookupError:
            continue
        pinned += 1
    return pinned


def pin_all_rt_threads_to_cpus(cpu_ids: Set[int]) -> int:
    """Pin all current RT threads in process to CPU ids."""

    if not cpu_ids:
        raise ValueError("cpu_ids must be non-empty")
    pinned = 0
    for tid in _iter_all_rt_threads():
        try:
            os.sched_setaffinity(tid, cpu_ids)
        except ProcessLookupError:
            continue
        pinned += 1
    return pinned


def cap_rt_thread_priorities(priority_cap: int, thread_ids: Iterable[int]) -> List[Dict[str, int]]:
    """Lower specified RT thread priorities to at most priority_cap."""

    if int(priority_cap) <= 0:
        raise ValueError("priority_cap must be > 0")
    updates: List[Dict[str, int]] = []
    for tid in _iter_target_rt_threads(thread_ids):
        try:
            policy = int(os.sched_getscheduler(tid))
            before_priority = int(getattr(os.sched_getparam(tid), "sched_priority", 0))
        except ProcessLookupError:
            continue
        max_priority = int(os.sched_get_priority_max(policy))
        min_priority = int(os.sched_get_priority_min(policy))
        target_priority = max(min(int(priority_cap), max_priority), min_priority)
        if before_priority <= target_priority:
            continue
        try:
            os.sched_setparam(tid, os.sched_param(target_priority))
            after_priority = int(getattr(os.sched_getparam(tid), "sched_priority", 0))
        except ProcessLookupError:
            continue
        updates.append(
            {
                "tid": int(tid),
                "policy_id": int(policy),
                "before_priority": int(before_priority),
                "after_priority": int(after_priority),
            }
        )
    return updates


def cap_all_rt_thread_priorities(priority_cap: int) -> List[Dict[str, int]]:
    """Lower all RT thread priorities in current process."""

    return cap_rt_thread_priorities(int(priority_cap), _iter_all_rt_threads())


def apply_rt_thread_policy(rt_cpu_set: Set[int], priority_cap: int) -> Dict[str, int]:
    """Apply RT thread affinity and fixed priority cap policy."""

    pinned = pin_all_rt_threads_to_cpus(rt_cpu_set)
    updates = cap_all_rt_thread_priorities(int(priority_cap))
    return {
        "pinned_rt_threads": int(pinned),
        "priority_updates": int(len(updates)),
    }


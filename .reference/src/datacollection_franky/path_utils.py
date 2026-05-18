"""Compatibility wrappers for legacy path helper imports."""

from __future__ import annotations

from typing import Optional

from .sim_paths import find_simulation_dir as _find_simulation_dir
from .urdf_tcp import find_fr3_urdf as _find_fr3_urdf


def find_simulation_dir(start_dir: str, max_up: int = 6) -> Optional[str]:
    """Compatibility wrapper for simulation directory discovery."""

    return _find_simulation_dir(start_dir, int(max_up))


def find_fr3_urdf(sim_dir: str) -> str:
    """Compatibility wrapper for configured FR3 URDF discovery."""

    return _find_fr3_urdf(sim_dir)

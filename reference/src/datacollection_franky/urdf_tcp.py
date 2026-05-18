"""URDF helpers for locating the FR3 model."""

from __future__ import annotations

import os


def find_fr3_urdf(sim_dir: str) -> str:
    """Return the nominal FR3 URDF path."""

    cand = os.path.join(sim_dir, "fr3.urdf")
    if os.path.isfile(cand):
        return cand
    for fn in os.listdir(sim_dir):
        if fn.lower().endswith(".urdf") and "fr3" in fn.lower():
            return os.path.join(sim_dir, fn)
    raise FileNotFoundError(f"Could not find FR3 URDF in {sim_dir}")


def write_configured_fr3_urdf(source_urdf_path: str) -> str:
    """Backward-compatible passthrough for callers expecting this helper."""
    return str(source_urdf_path)


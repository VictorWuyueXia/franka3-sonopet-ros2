"""Simulation path discovery helpers."""

from __future__ import annotations

import os
from typing import Optional


def find_simulation_dir(start_dir: str, max_up: int) -> Optional[str]:
    """Walk upwards and find a directory containing simulation/simWithPyBullet.py."""

    cur = os.path.abspath(start_dir)
    for _ in range(int(max_up)):
        cand = os.path.join(cur, "simulation")
        if os.path.isfile(os.path.join(cand, "simWithPyBullet.py")):
            return cand
        if os.path.isfile(os.path.join(cur, "simWithPyBullet.py")) and os.path.isfile(os.path.join(cur, "fr3.urdf")):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return None


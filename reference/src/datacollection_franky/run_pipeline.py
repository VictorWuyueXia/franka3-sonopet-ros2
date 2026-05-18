"""run_pipeline.py

Runnable entry point.

This file is inside the package directory. When executed as a script, Python does not
automatically treat the current directory as an importable package name.

So we explicitly add the parent folder to sys.path, then import the package.

Use:
  python datacollection_franky/run_pipeline.py --pcd-path <...> [--robot-ip <...>]
"""

from __future__ import annotations

import os
import sys

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_pkg_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from datacollection_franky.pipeline import main


if __name__ == "__main__":
    main()

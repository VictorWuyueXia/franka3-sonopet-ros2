from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_leaf_module():
    """Load the TF probe leaf module without importing package-level side effects."""

    module_path = Path(__file__).resolve().parent / "realsense_capture" / "ros_tf_probe.py"
    spec = importlib.util.spec_from_file_location("standalone_ros_tf_probe", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for TF probe: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_leaf_module()
main = _MODULE.main
parse_args = _MODULE.parse_args


if __name__ == "__main__":
    main()

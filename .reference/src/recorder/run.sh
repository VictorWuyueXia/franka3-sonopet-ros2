#!/usr/bin/env bash
# One-click runner. Uses the active shell environment's python3.
#
# Expected usage:
#   conda activate sonopet
#   ./run.sh
#
# System packages required (Ubuntu 24.04):
#   sudo apt install -y python3-pip alsa-utils \
#                       librealsense2-utils librealsense2-dev
# (RealSense udev rules: see https://github.com/IntelRealSense/librealsense)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[run.sh] python3 not found in PATH." >&2
    exit 1
fi

if [ "${CONDA_DEFAULT_ENV:-}" != "sonopet" ]; then
    echo "[run.sh] expected conda env 'sonopet' but found '${CONDA_DEFAULT_ENV:-<none>}'." >&2
    echo "[run.sh] run 'conda activate sonopet' first." >&2
    exit 1
fi

exec python3 launcher.py "$@"

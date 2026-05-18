#!/usr/bin/env bash

# Source this file in a Ubuntu 24.04 terminal before building or running the workspace.
set -euo pipefail

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy was not found at /opt/ros/jazzy/setup.bash."
  echo "Install ROS 2 Jazzy on Ubuntu 24.04, then source this file again."
  return 1 2>/dev/null || exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/jazzy/setup.bash

if [[ -f "${PROJECT_ROOT}/ros2_ws/install/setup.bash" ]]; then
  source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
fi

export FR3_SONOPET_REPO="${PROJECT_ROOT}"
echo "ROS 2 Jazzy environment loaded for ${PROJECT_ROOT}."

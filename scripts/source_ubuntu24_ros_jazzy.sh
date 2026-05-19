#!/usr/bin/env bash

# Source this file in a Ubuntu 24.04 terminal before building or running the workspace.
# Do not enable errexit in the caller's shell: colcon may exit 1 when optional vendor
# packages fail even though fr3_sonopet_* built successfully.
_sourced=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  _sourced=1
  _fr3_sonopet_saved_shell_opts="$(set +o)"
else
  set -euo pipefail
fi

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy was not found at /opt/ros/jazzy/setup.bash."
  echo "Install ROS 2 Jazzy on Ubuntu 24.04, then source this file again."
  return 1 2>/dev/null || exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set +u
source /opt/ros/jazzy/setup.bash
set -u

# Prefer the Ubuntu-managed Franka CMake package when an older manual libfranka
# install exists under /usr/local. The Franka ROS 2 Jazzy stack requires the
# package-managed API surface shipped for this ROS distribution.
case ":${CMAKE_IGNORE_PREFIX_PATH:-}:" in
  *:/usr/local:*) ;;
  *) export CMAKE_IGNORE_PREFIX_PATH="${CMAKE_IGNORE_PREFIX_PATH:+${CMAKE_IGNORE_PREFIX_PATH}:}/usr/local" ;;
esac

if [[ -d /usr/local/include/franka && -d /usr/include/franka ]]; then
  FR3_SONOPET_LOCAL_INCLUDE="${PROJECT_ROOT}/.local_include"
  mkdir -p "${FR3_SONOPET_LOCAL_INCLUDE}"
  ln -sfn /usr/include/franka "${FR3_SONOPET_LOCAL_INCLUDE}/franka"
  if [[ -d /usr/include/research_interface ]]; then
    ln -sfn /usr/include/research_interface "${FR3_SONOPET_LOCAL_INCLUDE}/research_interface"
  fi

  # GCC searches /usr/local/include before /usr/include by default. The local
  # shim ensures Franka ROS 2 compiles against the package-managed Jazzy headers.
  case " ${CXXFLAGS:-} " in
    *" -I${FR3_SONOPET_LOCAL_INCLUDE} "*) ;;
    *) export CXXFLAGS="-I${FR3_SONOPET_LOCAL_INCLUDE}${CXXFLAGS:+ ${CXXFLAGS}}" ;;
  esac
  case " ${CFLAGS:-} " in
    *" -I${FR3_SONOPET_LOCAL_INCLUDE} "*) ;;
    *) export CFLAGS="-I${FR3_SONOPET_LOCAL_INCLUDE}${CFLAGS:+ ${CFLAGS}}" ;;
  esac
fi

if [[ -f "${PROJECT_ROOT}/ros2_ws/install/setup.bash" ]]; then
  set +u
  source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
  set -u
fi

export FR3_SONOPET_REPO="${PROJECT_ROOT}"
echo "ROS 2 Jazzy environment loaded for ${PROJECT_ROOT}."

if [[ "${_sourced}" -eq 1 ]]; then
  eval "${_fr3_sonopet_saved_shell_opts}"
  unset _fr3_sonopet_saved_shell_opts _sourced
fi

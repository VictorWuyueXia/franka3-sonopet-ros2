#!/usr/bin/env bash

# Prepare an Ubuntu 24.04 host for this ROS 2 Jazzy workspace.
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This bootstrap script targets Ubuntu 24.04 systems with apt."
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  python3-colcon-common-extensions \
  python3-pip \
  python3-pytest \
  python3-rosdep \
  python3-vcstool \
  python3-yaml

# rosdep is system-wide; ignore the already-initialized case for repeatability.
sudo rosdep init 2>/dev/null || true
rosdep update

echo "Ubuntu ROS tooling is ready. Install ROS 2 Jazzy before building this workspace."

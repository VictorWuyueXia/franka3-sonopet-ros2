from __future__ import annotations

import os
from pathlib import Path

os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

CAMERA_FIXED_NAME = "camera-fixed"
CAMERA_IN_HAND_NAME = "camera-in-hand"
BASE_FRAME_NAME = "fr3_link0"
TCP_FRAME_NAME = "fr3_hand_tcp"
WINDOW_NAME = "RealSense - Dual Capture"
OVERLAY_TEXT = "Press 'c' capture  |  'q' quit"
ROS_SETUP_BASH = "/opt/ros/jazzy/setup.bash"
SYSTEM_PYTHON = "/usr/bin/python3"
HOME_DIR = Path.home()
REPO_ROOT = Path(__file__).resolve().parents[2]
CAMERA_ROLE_CONFIG_PATH = REPO_ROOT / "config" / "realsense_camera_roles.json"
ROS_TF_PROBE_PATH = REPO_ROOT / "gui" / "ros_tf_probe.py"
PATH_SANITIZE_COMMAND = (
    'export PATH=$(echo "$PATH" | tr \':\' \'\\n\' | '
    'grep -v miniconda | grep -v anaconda | tr \'\\n\' \':\' | sed \'s/:$//\')'
)

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realsense_boot import (
    DEFAULT_D405_PRESET_PATH,
    build_device_records,
    format_device_label,
    resolve_device,
)


def list_devices():
    device_records = build_device_records()
    if not device_records:
        raise RuntimeError("No RealSense device detected.")
    for index, record in enumerate(device_records):
        print(f"[{index}] {format_device_label(record['info'])}")


def build_launch_command(serial_number, preset_path):
    ros_serial_number = f"_{serial_number}"
    return [
        "ros2",
        "launch",
        "realsense2_camera",
        "rs_launch.py",
        f"serial_no:={ros_serial_number}",
        f"json_file_path:={preset_path}",
        "enable_color:=true",
        "enable_depth:=true",
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Boot the D405 ROS driver for hand-eye calibration."
    )
    parser.add_argument(
        "--serial",
        type=str,
        help="RealSense full serial number or last 4 digits.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print connected RealSense devices and exit.",
    )
    parser.add_argument(
        "launch_args",
        nargs=argparse.REMAINDER,
        help="Additional ros2 launch args appended after the preset and serial settings.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        list_devices()
        return

    _device, device_info = resolve_device(args.serial)
    preset_path = DEFAULT_D405_PRESET_PATH.resolve()
    launch_command = build_launch_command(device_info["serial"], preset_path)
    if args.launch_args:
        launch_command.extend(args.launch_args)

    print(f"Using device: {format_device_label(device_info)}")
    print(f"Loading preset on boot: {preset_path}")
    print("Launching hand-eye camera driver...")
    print(" ".join(launch_command))
    subprocess.run(launch_command, check=True)


if __name__ == "__main__":
    main()

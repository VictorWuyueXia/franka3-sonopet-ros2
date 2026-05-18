#!/usr/bin/env python3
"""
One-click launcher: creates ./experiment_<YYYYMMDD>_<HHMMSS>/ and starts a
recorder process for every detected Intel RealSense D405 and every detected
USB ALSA capture device.

Press Ctrl+C once to stop everything cleanly.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CAMERA_ROLE_CONFIG_PATH = REPO_ROOT / "config" / "realsense_camera_roles.json"


def discover_realsense_serials() -> list[tuple[str, str]]:
    """Return [(serial, product_name)] for every connected RealSense device."""
    try:
        import pyrealsense2 as rs  # imported lazily so --help works without it
    except ImportError:
        print("[launcher] pyrealsense2 not installed; cannot enumerate cameras.",
              file=sys.stderr)
        return []
    out = []
    try:
        for d in rs.context().query_devices():
            sn = d.get_info(rs.camera_info.serial_number)
            name = d.get_info(rs.camera_info.name)
            out.append((sn, name))
    except Exception as e:
        print(f"[launcher] RealSense enumeration failed: {e}", file=sys.stderr)
    return out


def discover_capture_devices() -> list[tuple[str, str]]:
    """Return [(plughw_id, description)] for every ALSA capture device."""
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        print("[launcher] arecord not found (sudo apt install alsa-utils).", file=sys.stderr)
        return []
    if out.returncode != 0:
        return []
    pat = re.compile(r"card\s+(\d+):.*?device\s+(\d+):", re.IGNORECASE)
    devices = []
    for line in (out.stdout or "").splitlines():
        m = pat.search(line)
        if m:
            card, dev = int(m.group(1)), int(m.group(2))
            devices.append((f"plughw:{card},{dev}", line.strip()))
    return devices


def parse_csv_list(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def sanitize_label(text: str) -> str:
    return str(text).replace(":", "_").replace(",", "_").replace(" ", "_")


def load_camera_role_by_serial(config_path: Path) -> dict[str, str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    role_by_serial = {}
    for role_name, role_payload in payload.items():
        serial = str(role_payload.get("serial") or "").strip()
        if serial:
            role_by_serial[serial] = str(role_name)
    return role_by_serial


def build_camera_targets(
    detected_cameras: list[tuple[str, str]],
    requested_serials: list[str],
    role_by_serial: dict[str, str],
) -> list[tuple[str, str, str]]:
    camera_by_serial = {str(serial): str(name) for serial, name in detected_cameras}
    serials = requested_serials if requested_serials else list(camera_by_serial.keys())
    targets = []
    for serial in serials:
        role_name = role_by_serial.get(str(serial))
        product_name = camera_by_serial.get(str(serial), str(serial))
        if role_name is None:
            print(f"[launcher] skipping unmapped RealSense serial {serial} ({product_name})", flush=True)
            continue
        print(f"[launcher] camera: {serial} ({product_name}) -> role={role_name}", flush=True)
        targets.append((str(serial), str(role_name), str(product_name)))
    return targets


def build_mic_targets(
    detected_mics: list[tuple[str, str]],
    requested_devices: list[str],
) -> list[tuple[str, str, str]]:
    desc_by_device = {str(device): str(description) for device, description in detected_mics}
    devices = requested_devices if requested_devices else list(desc_by_device.keys())
    targets = []
    for device in devices:
        device_name = desc_by_device.get(str(device), str(device))
        label = sanitize_label(str(device))
        print(f"[launcher] mic: {device} - {device_name}", flush=True)
        targets.append((str(device), label, device_name))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(HERE),
                        help="Parent directory for experiment_* (default: recorder folder).")
    parser.add_argument("--camera-serials", default="",
                        help="Comma-separated D405 serials. Empty = use all detected.")
    parser.add_argument("--mic-devices", default="",
                        help="Comma-separated ALSA devices (e.g. plughw:2,0). "
                             "Empty = use all detected.")
    parser.add_argument("--robot-ip", default="172.16.0.2",
                        help="Robot IP used for camera-in-hand pointcloud base alignment.")
    parser.add_argument("--fps", type=int, default=15, help="D405 frame rate (default 10).")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-mic", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.base_dir).resolve() / f"experiment_{stamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[launcher] experiment dir: {exp_dir}", flush=True)

    # ---- enumerate cameras ----
    camera_targets: list[tuple[str, str, str]] = []  # (serial, role, product_name)
    if not args.no_camera:
        role_by_serial = load_camera_role_by_serial(CAMERA_ROLE_CONFIG_PATH)
        cams = discover_realsense_serials()
        if args.camera_serials:
            camera_targets = build_camera_targets(cams, parse_csv_list(args.camera_serials), role_by_serial)
        else:
            if not cams:
                print("[launcher] no RealSense cameras detected.", flush=True)
            camera_targets = build_camera_targets(cams, [], role_by_serial)

    # ---- enumerate mics ----
    mic_targets: list[tuple[str, str, str]] = []  # (alsa_device, label, device_name)
    if not args.no_mic:
        mics = discover_capture_devices()
        if args.mic_devices:
            mic_targets = build_mic_targets(mics, parse_csv_list(args.mic_devices))
        else:
            if not mics:
                print("[launcher] no ALSA capture devices detected.", flush=True)
            mic_targets = build_mic_targets(mics, [])

    procs: list[tuple[str, subprocess.Popen]] = []
    py = sys.executable

    for serial, role_name, _product_name in camera_targets:
        cam_dir = exp_dir / f"camera_{sanitize_label(role_name)}"
        cam_dir.mkdir(exist_ok=True)
        cmd = [py, str(HERE / "record_d405.py"),
               "--out-dir", str(cam_dir),
               "--serial", serial,
               "--camera-role", role_name,
               "--robot-ip", str(args.robot_ip),
               "--fps", str(args.fps)]
        procs.append((f"d405:{role_name}", subprocess.Popen(cmd)))

    for device, label, device_name in mic_targets:
        mic_dir = exp_dir / f"audio_{label}"
        mic_dir.mkdir(exist_ok=True)
        cmd = [py, str(HERE / "record_mic.py"),
               "--out-dir", str(mic_dir),
               "--device", device,
               "--device-name", device_name]
        procs.append((f"mic:{label}", subprocess.Popen(cmd)))

    if not procs:
        print("[launcher] nothing to run.", file=sys.stderr)
        return 1

    (exp_dir / "launcher_meta.txt").write_text(
        f"start_unix={time.time():.6f}\nstart_iso={datetime.now().isoformat()}\n"
        f"fps={args.fps}\n"
        f"robot_ip={args.robot_ip}\n"
        f"cameras=" + ",".join(f"{role}:{sn}" for sn, role, _ in camera_targets) + "\n"
        f"mics=" + ",".join(f"{d}:{name}" for d, _, name in mic_targets) + "\n"
        f"pids=" + ",".join(f"{n}:{p.pid}" for n, p in procs) + "\n"
    )

    stopping = {"flag": False}

    def _shutdown(signum, _frame):
        if stopping["flag"]:
            return
        stopping["flag"] = True
        print(f"\n[launcher] signal {signum} -> stopping children", flush=True)
        for name, p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGINT)
                except Exception as e:
                    print(f"[launcher] could not signal {name}: {e}", file=sys.stderr)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            alive = [(n, p) for n, p in procs if p.poll() is None]
            if not alive:
                break
            if not stopping["flag"] and len(alive) < len(procs):
                for n, p in procs:
                    if p.poll() is not None:
                        print(f"[launcher] {n} exited with code {p.returncode}", flush=True)
                _shutdown(signal.SIGINT, None)
            time.sleep(0.2)
    finally:
        deadline = time.time() + 10.0
        for name, p in procs:
            remaining = max(0.0, deadline - time.time())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"[launcher] killing {name} (pid {p.pid})", file=sys.stderr)
                p.kill()

    with (exp_dir / "launcher_meta.txt").open("a") as f:
        f.write(f"end_unix={time.time():.6f}\nend_iso={datetime.now().isoformat()}\n")
    print(f"[launcher] done. data in {exp_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

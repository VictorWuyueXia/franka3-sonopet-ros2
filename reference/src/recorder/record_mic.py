#!/usr/bin/env python3
"""
USB microphone recorder using ALSA `arecord`.

Writes:
  - audio.wav        (S16_LE PCM, 48 kHz mono by default)
  - audio_meta.txt   (start/end machine timestamps + arecord command)

Auto-detects the first plausible USB capture device. Stops on SIGINT/SIGTERM.
"""

from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_DEVICE_KEYWORDS = [
    "ugreen", "realtek", "cm720", "usb audio",
    "imm6c", "imm-6c", "dayton", "cmm30315",
]


def list_capture_devices_cli() -> int:
    """CLI helper: print one 'plughw:C,D\tdescription' per line, for the launcher."""
    for hw, desc in list_capture_devices():
        print(f"{hw}\t{desc}")
    return 0


def list_capture_devices() -> list[tuple[str, str]]:
    """Return [(plughw_id, description)] from `arecord -l`."""

    out = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5)
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


def auto_select_device(keywords: list[str]) -> tuple[str, str]:
    devices = list_capture_devices()
    if not devices:
        print("[mic] no capture device found; falling back to 'default'", flush=True)
        return "default", "default"
    if len(devices) == 1:
        print(f"[mic] only device: {devices[0][0]} - {devices[0][1]}", flush=True)
        return devices[0][0], devices[0][1]
    kws = [k.lower() for k in keywords]
    scored = [(hw, desc, sum(1 for k in kws if k in desc.lower())) for hw, desc in devices]
    scored.sort(key=lambda x: -x[2])
    print(f"[mic] selected {scored[0][0]} - {scored[0][1]}", flush=True)
    return scored[0][0], scored[0][1]


def lookup_device_name(device: str) -> str:
    for hw, desc in list_capture_devices():
        if str(hw) == str(device):
            return str(desc)
    return str(device)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--list-devices":
        return list_capture_devices_cli()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto", help="ALSA device or 'auto'")
    parser.add_argument("--device-name", default="", help="Human-readable microphone name")
    parser.add_argument("--rate", type=int, default=48000)
    parser.add_argument("--channels", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "audio.wav"
    meta_path = out_dir / "audio_meta.txt"

    device = args.device
    if device.strip().lower() == "auto":
        device, device_name = auto_select_device(DEFAULT_DEVICE_KEYWORDS)
    else:
        device_name = str(args.device_name).strip() or lookup_device_name(device)

    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",
        "-r", str(args.rate),
        "-c", str(args.channels),
        "-t", "wav",
        str(wav_path),
    ]
    print(f"[mic] running: {' '.join(cmd)}", flush=True)

    start_ts = time.time()
    try:
        proc = subprocess.Popen(cmd)
    except FileNotFoundError:
        print("[mic] ERROR: arecord not found. Install with: sudo apt install alsa-utils",
              file=sys.stderr)
        return 1

    stop = {"flag": False}

    def _handle(signum, _frame):
        print(f"[mic] received signal {signum}, stopping arecord", flush=True)
        stop["flag"] = True
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    ret = proc.wait()
    end_ts = time.time()

    meta_path.write_text(
        f"command={' '.join(cmd)}\n"
        f"device={device}\n"
        f"device_name={device_name}\n"
        f"sample_rate={args.rate}\n"
        f"channels={args.channels}\n"
        f"format=S16_LE\n"
        f"start_unix={start_ts:.6f}\n"
        f"end_unix={end_ts:.6f}\n"
        f"duration_s={end_ts - start_ts:.3f}\n"
        f"arecord_exit_code={ret}\n"
    )
    print(f"[mic] wrote {wav_path} ({end_ts - start_ts:.2f}s)", flush=True)
    return 0 if ret in (0, -2, 130) else ret  # SIGINT exits are fine


if __name__ == "__main__":
    sys.exit(main())

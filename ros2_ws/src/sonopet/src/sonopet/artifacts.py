from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

SAMPLE_RATE_HZ = 200
FOOTPEDAL_PORT = "/dev/ttyACM0"
FOOTPEDAL_BAUD = 115200
SONOPET_SETTINGS = {
    "suction_power": "not_exposed_by_sonopet_live_data",
    "irrigation": "not_exposed_by_sonopet_live_data",
}


def wall_clock_timestamp() -> float:
    """Return local epoch seconds aligned with the recorder artifact metadata."""
    return time.time()


def local_timestamp_label(epoch_seconds: float) -> str:
    """Return compact local wall-clock time for operator-visible metadata."""
    return datetime.fromtimestamp(epoch_seconds).astimezone().strftime("%Y%m%d%H%M")


def sonopet_case_path(directory: Path, epoch_seconds: float) -> Path:
    """Return the legacy Sonopet case filename used by the DAQ JSON stream."""
    timestamp = datetime.fromtimestamp(epoch_seconds).astimezone()
    nanosecond_suffix = time.time_ns() % 10_000_000_000
    filename = f"{timestamp:%Y_%m_%d_T_%H_%M_%S}_{nanosecond_suffix:010d}.json"
    return next_existing_name(directory / f"SonopetCase_{filename}")


def next_existing_name(path: Path) -> Path:
    """Return the first path following the project duplicate-suffix convention."""
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def relative_artifact_path(root: Path, path: Path) -> str:
    """Return the artifact path relative to the shared experiment directory."""
    return str(path.relative_to(root))


def write_sonopet_metadata(root: Path, intervals: list[dict]) -> Path:
    """Write the Sonopet interval metadata beside the JSONL DAQ artifacts."""
    metadata_path = root / "sonopet" / "sonopet_meta.json"
    payload = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "footpedal": {
            "port": FOOTPEDAL_PORT,
            "baud": FOOTPEDAL_BAUD,
        },
        "settings": SONOPET_SETTINGS,
        "intervals": intervals,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path

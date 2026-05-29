import json
from pathlib import Path

from sonopet.artifacts import (
    FOOTPEDAL_BAUD,
    FOOTPEDAL_PORT,
    SAMPLE_RATE_HZ,
    SONOPET_SETTINGS,
    next_existing_name,
    write_sonopet_metadata,
)


def test_duplicate_sonopet_names_follow_recording_convention(tmp_path):
    base = tmp_path / "sonopet" / "sonopet.jsonl"
    base.parent.mkdir()
    base.write_text("", encoding="utf-8")
    base.with_name("sonopet_1.jsonl").write_text("", encoding="utf-8")

    candidate = next_existing_name(base)

    assert candidate.name == "sonopet_2.jsonl"


def test_sonopet_metadata_keeps_interval_timestamps_and_settings(tmp_path):
    intervals = [
        {
            "path": "sonopet/sonopet.jsonl",
            "started_at": 1_700_000_000.0,
            "started_at_local": "202605271551",
            "stopped_at": 1_700_000_005.0,
            "stopped_at_local": "202605271552",
            "sample_count": 1000,
        }
    ]

    metadata_path = write_sonopet_metadata(tmp_path, intervals)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert payload["sample_rate_hz"] == SAMPLE_RATE_HZ
    assert payload["footpedal"] == {"port": FOOTPEDAL_PORT, "baud": FOOTPEDAL_BAUD}
    assert payload["settings"] == SONOPET_SETTINGS
    assert payload["intervals"] == intervals


def test_sonopet_node_source_owns_cutting_artifact_and_footpedal_paths():
    package_root = Path(__file__).resolve().parents[1]
    node = (package_root / "src" / "sonopet" / "sonopet_node.py").read_text(encoding="utf-8")
    setup = (package_root / "setup.py").read_text(encoding="utf-8")

    assert 'CUTTING_TOPIC = "/sonopet/cutting"' in node
    assert 'ARTIFACT_PATH_TOPIC = "/sonopet/artifact_path"' in node
    assert "create_subscription(\n            String" in node
    assert "create_subscription(\n            Bool" in node
    assert "serial.Serial(FOOTPEDAL_PORT, FOOTPEDAL_BAUD" in node
    assert 'self._footpedal.write(b"1")' in node
    assert 'self._footpedal.write(b"0")' in node
    assert 'self._socket.sendall(b"grab")' in node
    assert 'self._socket.sendall(b"stop")' in node
    assert '"bin/sonopet_live_data"' in setup
    assert (package_root / "bin" / "sonopet_live_data").exists()

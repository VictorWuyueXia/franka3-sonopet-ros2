from pathlib import Path

import pytest
from fr3_sonopet_microphone.audio_devices import select_input_device


def _microphone_source() -> str:
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1]
    return (
        package_root / "src" / "fr3_sonopet_microphone" / "microphone_node.py"
    ).read_text(encoding="utf-8")


def test_selects_imm6c_preferred_device_by_case_insensitive_substring():
    devices = [
        {"name": "Laptop Mic", "max_input_channels": 1, "default_samplerate": 44100.0},
        {"name": "USB iMM-6C Calibrated Microphone", "max_input_channels": 1},
    ]

    selected = select_input_device(devices, preferred_names=["imm6c", "iMM-6C"])

    assert selected.index == 1
    assert selected.name == "USB iMM-6C Calibrated Microphone"


def test_selects_first_matching_preferred_name_order():
    devices = [
        {"name": "iMM-6C backup", "max_input_channels": 1},
        {"name": "Primary imm6c interface", "max_input_channels": 2},
    ]

    selected = select_input_device(devices, preferred_names=["primary IMM6C", "iMM-6C"])

    assert selected.index == 1
    assert selected.channels == 2


def test_no_preferred_microphone_match_fails_loudly():
    devices = [{"name": "Laptop Mic", "max_input_channels": 1}]

    with pytest.raises(RuntimeError, match="No configured microphone matched"):
        select_input_device(devices, preferred_names=["iMM-6C", "imm6c"])


def test_microphone_chunks_carry_sequence_and_gap_fill_status():
    source = _microphone_source()
    interfaces_root = Path(__file__).resolve().parents[2] / "fr3_sonopet_interfaces"
    audio_msg = (interfaces_root / "msg" / "AudioChunk.msg").read_text(encoding="utf-8")

    assert "uint64 chunk_index" in audio_msg
    assert "bool gap_fill" in audio_msg
    assert "chunk_index: int" in source
    assert "gap_fill: bool" in source
    assert "self._next_chunk_index += 1" in source
    assert "msg.chunk_index = chunk.index" in source
    assert "msg.gap_fill = chunk.gap_fill" in source
    assert "Filled {dropped} dropped microphone chunks with silence" in source

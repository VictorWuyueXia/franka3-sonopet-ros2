import pytest
from fr3_sonopet_microphone.audio_devices import select_input_device


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

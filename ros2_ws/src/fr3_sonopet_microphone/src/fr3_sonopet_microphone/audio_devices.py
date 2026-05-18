from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AudioInputDevice:
    """Resolved input device identity used by the microphone node."""

    index: int | None
    name: str
    channels: int
    default_sample_rate_hz: int


def _device_name(device: dict[str, Any]) -> str:
    return str(device.get("name", "")).strip()


def _input_channels(device: dict[str, Any]) -> int:
    return int(device.get("max_input_channels", 0) or 0)


def _default_sample_rate_hz(device: dict[str, Any]) -> int:
    return int(round(float(device.get("default_samplerate", 0.0) or 0.0)))


def list_input_devices(devices: list[dict[str, Any]]) -> list[AudioInputDevice]:
    """Normalize sounddevice query output into input-capable device records."""

    inputs: list[AudioInputDevice] = []
    for index, device in enumerate(devices):
        channels = _input_channels(device)
        if channels <= 0:
            continue
        inputs.append(
            AudioInputDevice(
                index=index,
                name=_device_name(device),
                channels=channels,
                default_sample_rate_hz=_default_sample_rate_hz(device),
            )
        )
    return inputs


def select_input_device(
    devices: list[dict[str, Any]],
    preferred_names: list[str],
    explicit_device: str = "",
    fail_if_preferred_not_found: bool = True,
) -> AudioInputDevice:
    """Resolve the microphone device from explicit config or preferred name substrings."""

    input_devices = list_input_devices(devices)
    if not input_devices:
        raise RuntimeError("No audio input devices are available")

    explicit = explicit_device.strip()
    if explicit:
        if explicit.isdigit():
            explicit_index = int(explicit)
            for device in input_devices:
                if device.index == explicit_index:
                    return device
        explicit_lower = explicit.lower()
        for device in input_devices:
            if explicit_lower in device.name.lower():
                return device
        raise RuntimeError(f"Configured microphone device was not found: {explicit}")

    normalized_preferences = [name.strip().lower() for name in preferred_names if name.strip()]
    for preferred_name in normalized_preferences:
        for device in input_devices:
            if preferred_name in device.name.lower():
                return device

    if fail_if_preferred_not_found:
        available = ", ".join(device.name for device in input_devices)
        preferred = ", ".join(preferred_names)
        raise RuntimeError(
            f"No configured microphone matched [{preferred}]. Available input devices: {available}"
        )

    return input_devices[0]

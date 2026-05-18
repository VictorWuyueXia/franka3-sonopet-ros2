from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioFormat:
    sample_rate_hz: int = 48_000
    channels: int = 1
    encoding: str = "S16_LE"


def describe_audio_format(audio_format: AudioFormat) -> str:
    # Compact format text keeps runtime logs useful during launch smoke tests.
    return f"{audio_format.sample_rate_hz}Hz/{audio_format.channels}ch/{audio_format.encoding}"

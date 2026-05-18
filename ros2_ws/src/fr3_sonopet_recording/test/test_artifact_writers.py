import csv
import wave

import numpy as np
from fr3_sonopet_recording.artifact_writers import AudioWavRecorder, RgbVideoRecorder


def test_rgb_video_recorder_writes_video_and_timestamp_rows(tmp_path):
    recorder = RgbVideoRecorder(
        tmp_path / "rgb.avi",
        tmp_path / "rgb_timestamps.csv",
        fps=30.0,
        codec="MJPG",
    )

    recorder.write_frame(np.zeros((8, 8, 3), dtype=np.uint8), 12, 345)
    recorder.close()

    assert (tmp_path / "rgb.avi").exists()
    with (tmp_path / "rgb_timestamps.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows == [
        ["frame_index", "stamp_sec", "stamp_nanosec"],
        ["0", "12", "345"],
    ]


def test_audio_wav_recorder_writes_pcm_and_metadata(tmp_path):
    recorder = AudioWavRecorder(tmp_path / "audio.wav")

    recorder.write_samples([0, 100, -100, 0], sample_rate_hz=48_000, channels=1, encoding="S16_LE")
    recorder.close()

    with wave.open(str(tmp_path / "audio.wav"), "rb") as stream:
        assert stream.getframerate() == 48_000
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getnframes() == 4
    assert recorder.metadata()["samples"] == 4
    assert recorder.metadata()["chunks"] == 1

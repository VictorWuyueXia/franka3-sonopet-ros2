#!/usr/bin/env python3
"""Stream from USB/default mic and show real-time waveform + spectrogram."""

import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal as scipy_signal
from scipy.io import wavfile
from datetime import datetime

# Save recordings next to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Config
SAMPLE_RATE = 44100
CHUNK = 1024
BUFFER_SEC = 2.0
N_BUFFER = int(BUFFER_SEC * SAMPLE_RATE)
DEVICE = 10 # None = default input; or set index from list (e.g. 2 for USB mic)


def list_input_devices():
    """Print all input devices; default is marked."""
    default = sd.default.device[0]
    print("Input devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            mark = " (DEFAULT)" if i == default else ""
            print(f"  [{i}] {dev['name']} (ch: {dev['max_input_channels']}, sr: {dev['default_samplerate']}){mark}")

# Ring buffer for audio
buffer = np.zeros(N_BUFFER, dtype=np.float32)
buf_idx = [0]  # list so closure can mutate
recorded_chunks = []  # accumulated for saving to file


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    recorded_chunks.append(indata.copy())
    n = min(frames, N_BUFFER)
    start = buf_idx[0]
    end = (start + n) % N_BUFFER
    if end > start:
        buffer[start:end] = indata[:n, 0]
    else:
        buffer[start:] = indata[: N_BUFFER - start, 0]
        buffer[:end] = indata[N_BUFFER - start : n, 0]
    buf_idx[0] = end


def update_plot(frame):
    ax1.clear()
    ax2.clear()

    # Waveform (last BUFFER_SEC of audio)
    t = np.arange(N_BUFFER) / SAMPLE_RATE
    ax1.plot(t, buffer, "b-", linewidth=0.5)
    ax1.set_ylim(-1, 1)
    ax1.set_xlim(0, BUFFER_SEC)
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (s)")
    ax1.set_title("Waveform")

    # Spectrogram of buffer using FFT
    n_fft = 512
    hop = 256
    f, t_spec, Sxx = scipy_signal.spectrogram(
        buffer, SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft - hop
    )
    # Log scale for visibility
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    ax2.pcolormesh(
        t_spec, f, Sxx_db, shading="auto", cmap="viridis", vmin=-80, vmax=0
    )
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Spectrogram")
    ax2.set_ylim(0, 8000)

    return []


def main():
    global ax1, ax2
    recorded_chunks.clear()
    list_input_devices()
    dev_idx = DEVICE if DEVICE is not None else sd.default.device[0]
    dev_name = sd.query_devices(dev_idx)["name"]
    print(f"\nUsing device [{dev_idx}]: {dev_name}\n")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    plt.tight_layout()

    stream = sd.InputStream(
        device=dev_idx,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        blocksize=CHUNK,
        callback=audio_callback,
    )
    stream.start()

    try:
        ani = animation.FuncAnimation(
            fig, update_plot, interval=50, blit=False, cache_frame_data=False
        )
        plt.show()
    finally:
        stream.stop()
        stream.close()
        if recorded_chunks:
            recorded = np.concatenate(recorded_chunks, axis=0)[:, 0]
            wav_name = datetime.now().strftime("recording_%Y%m%d_%H%M%S.wav")
            wav_path = os.path.join(SCRIPT_DIR, wav_name)
            wavfile.write(wav_path, SAMPLE_RATE, (recorded * 32767).astype(np.int16))
            print(f"Saved: {wav_path}")

            # FFT of full recording
            n = len(recorded)
            if n > 0:
                fft_vals = np.fft.rfft(recorded)
                fft_mag = np.abs(fft_vals) / n
                freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
                fig_fft, ax_fft = plt.subplots(figsize=(8, 4))
                ax_fft.plot(freqs, 20 * np.log10(fft_mag + 1e-12), "b-", linewidth=0.5)
                ax_fft.set_xlim(0, min(22050, SAMPLE_RATE / 2))
                ax_fft.set_xlabel("Frequency (Hz)")
                ax_fft.set_ylabel("Magnitude (dB)")
                ax_fft.set_title("FFT of recorded audio")
                ax_fft.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Plot Sonopet runtime telemetry exported by sonopet_rise_parser_json.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV = Path(
    "/media/btllab/B2EEF271EEF22CEB/Ubuntu/franka3-sonopet-ros2/data_collection/experiments/20260601T155242_chicken_1_90_50_15_3/sonopet/combined_runtime_data.csv"
)
Y_LIMITS = {"Frequency_Hz": (25350, 25450), "FreqTrackingQuality": (-0.01, 0.01)}


def time_axis_seconds(frame: pd.DataFrame) -> pd.Series:
    """Return experiment-relative seconds from the best available timestamp column."""
    source = "timestamp" if "timestamp" in frame.columns else "time_s"
    time_s = pd.to_numeric(frame[source], errors="coerce")
    return time_s - time_s.dropna().iloc[0]


def plot_group(frame: pd.DataFrame, time_s: pd.Series, columns: list[str], title: str, path: Path) -> None:
    """Save one multi-channel telemetry figure."""
    available = [column for column in columns if column in frame.columns]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 2.2 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for axis, column in zip(axes, available, strict=True):
        values = pd.to_numeric(frame[column], errors="coerce")
        axis.plot(time_s, values, linewidth=0.8)
        if column in Y_LIMITS:
            axis.set_ylim(*Y_LIMITS[column])
        axis.set_ylabel(column)
        axis.grid(True, alpha=0.3)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Time from first sample (s)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_overlay(frame: pd.DataFrame, time_s: pd.Series, columns: list[str], title: str, path: Path) -> None:
    """Save a normalized overlay for quick channel-to-channel timing inspection."""
    fig, axis = plt.subplots(figsize=(12, 5))
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        spread = values.max() - values.min()
        if spread == 0 or pd.isna(spread):
            continue
        axis.plot(time_s, (values - values.min()) / spread, linewidth=0.8, label=column)

    axis.set_title(title)
    axis.set_xlabel("Time from first sample (s)")
    axis.set_ylabel("Normalized value")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    time_s = time_axis_seconds(frame)
    out_dir = args.out_dir or args.csv.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_group(
        frame,
        time_s,
        ["Canister_Pressure_psi", "HP_Pressure_psi", "FreqTrackingQuality", "Frequency_Hz"],
        "Sonopet Pressure And Frequency Tracking",
        out_dir / "sonopet_pressure_frequency_tracking.png",
    )
    plot_group(
        frame,
        time_s,
        ["FootPedalPos_pct", "Power_W", "Frequency_Hz", "FreqTrackingQuality"],
        "Sonopet Drive Telemetry",
        out_dir / "sonopet_drive_telemetry.png",
    )
    plot_group(
        frame,
        time_s,
        ["HP_Pressure_psi", "Canister_Pressure_psi", "Irrigation_Actual_ml_per_min"],
        "Sonopet Pressure And Irrigation",
        out_dir / "sonopet_pressure_irrigation.png",
    )
    plot_group(
        frame,
        time_s,
        ["V_Handpiece_mag_V", "I_Handpiece_mag_A", "Z_Handpiece_mag_Ohm", "PowerSupplyVoltage"],
        "Sonopet Electrical Telemetry",
        out_dir / "sonopet_electrical_telemetry.png",
    )
    plot_overlay(
        frame,
        time_s,
        [
            "FootPedalPos_pct",
            "Power_W",
            "Frequency_Hz",
            "HP_Pressure_psi",
            "Irrigation_Actual_ml_per_min",
        ],
        "Normalized Sonopet Telemetry Overlay",
        out_dir / "sonopet_normalized_overlay.png",
    )

    print(f"Saved Sonopet runtime plots to {out_dir}")


if __name__ == "__main__":
    main()

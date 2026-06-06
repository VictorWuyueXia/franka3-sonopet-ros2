#!/usr/bin/env python3
"""Summarize successful experiment info.txt files and plot mass-balance trends."""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path(
    "/media/btllab/B2EEF271EEF22CEB/Ubuntu/franka3-sonopet-ros2/data_collection"
)
FAILED_NAME_MARKERS = ("_failed", "miss_sonopet", "miss_pointcloud")


def is_success_experiment(path: Path) -> bool:
    """Trial success is encoded by absence of known failure markers in the folder name."""
    name = path.name.lower()
    return not any(marker in name for marker in FAILED_NAME_MARKERS)


def numeric_prefix(value: str) -> float | None:
    """Extract the leading measurement value while preserving units in separate columns."""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


def normalize_key(raw_key: str) -> str:
    """Normalize handwritten key variants such as 'temp 3' into stable table columns."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", raw_key.strip().lower()).strip("_")


def parse_info_file(info_path: Path) -> dict:
    """Convert one info.txt file into numeric measurement columns plus free-text notes."""
    row: dict[str, object] = {
        "experiment": info_path.parent.name,
        "info_path": str(info_path),
    }
    notes: list[str] = []
    for raw_line in info_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            notes.append(line)
            continue
        key, value = line.split(":", 1)
        key = normalize_key(key)
        value = value.strip()
        row[key] = value
        number = numeric_prefix(value)
        if number is not None:
            row[f"{key}_value"] = number
    row["notes"] = " | ".join(notes)
    return row


def load_info_table(data_root: Path) -> list[dict]:
    """Load only successful experiment info files from the data-collection tree."""
    info_paths = sorted((data_root / "experiments").glob("*/info.txt"))
    rows = [parse_info_file(path) for path in info_paths if is_success_experiment(path.parent)]
    if not rows:
        raise RuntimeError(f"No successful info.txt files found under {data_root / 'experiments'}")
    rows = sorted(rows, key=lambda row: str(row["experiment"]))
    for row in rows:
        row["sample_mass_change_g"] = (
            row["sample_weight_after_cut_value"] - row["sample_weight_before_cut_value"]
        )
        row["sample_mass_removed_g"] = -row["sample_mass_change_g"]
        row["canister_mass_change_g"] = (
            row["canister_weight_after_cut_value"] - row["canister_weight_before_cut_value"]
        )
        row["luken_trap_mass_change_g"] = (
            row["luken_trap_weight_after_cut_value"]
            - row["luken_trap_weight_before_cut_value"]
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Write all discovered info.txt fields without imposing a fixed schema."""
    fieldnames = sorted({key for row in rows for key in row})
    leading = ["experiment", "info_path", "velocity", "sample_mass_change_g", "notes"]
    ordered = [key for key in leading if key in fieldnames]
    ordered.extend(key for key in fieldnames if key not in ordered)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(rows: list[dict], path: Path) -> None:
    """Write the compact human-readable table used for quick experiment review."""
    columns = [
        "experiment",
        "velocity",
        "sample_weight_before_cut",
        "sample_weight_after_cut",
        "sample_mass_change_g",
        "canister_volume",
        "notes",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(columns) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            values = [markdown_value(row, col) for col in columns]
            fh.write("| " + " | ".join(values) + " |\n")


def markdown_value(row: dict, column: str) -> str:
    """Format derived numeric columns for direct reading in the markdown summary."""
    value = row.get(column, "")
    if isinstance(value, float):
        return f"{value:+.2f}" if column.endswith("_change_g") else f"{value:.2f}"
    return str(value)


def row_values(rows: list[dict], key: str) -> list:
    """Return one plotted column in experiment order."""
    return [row[key] for row in rows]


def experiment_labels(rows: list[dict]) -> list[str]:
    """Use folder names as stable x-axis labels."""
    return [str(row["experiment"]) for row in rows]


def save_weight_before_after(rows: list[dict], fig_dir: Path) -> None:
    """Plot paired sample weights so water uptake appears as an upward post-cut segment."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = experiment_labels(rows)
    for idx, row in enumerate(rows):
        color = "#2e7d32" if row["sample_mass_change_g"] > 0 else "#1f4e79"
        ax.plot(
            [idx - 0.18, idx + 0.18],
            [row["sample_weight_before_cut_value"], row["sample_weight_after_cut_value"]],
            marker="o",
            color=color,
            linewidth=2,
        )
    ax.set_xticks(list(range(len(rows))))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Sample weight (g)")
    ax.set_title("Sample Weight Before and After Cutting")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "sample_weight_before_after.png")
    plt.close(fig)


def save_sample_mass_change(rows: list[dict], fig_dir: Path) -> None:
    """Plot signed post-cut mass change; positive values indicate net water uptake."""
    labels = experiment_labels(rows)
    changes = row_values(rows, "sample_mass_change_g")
    colors = ["#2e7d32" if value > 0 else "#1f4e79" for value in changes]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, changes, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(list(range(len(rows))))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("After - before sample weight (g)")
    ax.set_title("Signed Sample Mass Change")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "sample_mass_change.png")
    plt.close(fig)


def save_collected_mass_summary(rows: list[dict], fig_dir: Path) -> None:
    """Compare sample mass change with material and fluid collected downstream."""
    labels = experiment_labels(rows)
    series = [
        ("sample removed", row_values(rows, "sample_mass_removed_g"), "#1f4e79"),
        ("canister gained", row_values(rows, "canister_mass_change_g"), "#d96c2c"),
        ("luken trap gained", row_values(rows, "luken_trap_mass_change_g"), "#2e7d32"),
        ("tissue from luken", row_values(rows, "tissue_take_out_from_luken_value"), "#7a3b9e"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    x = list(range(len(rows)))
    for offset, (name, values, color) in zip(offsets, series, strict=True):
        ax.bar([idx + offset for idx in x], values, width=width, label=name, color=color)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mass change (g)")
    ax.set_title("Mass Balance Signals From info.txt")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "mass_balance_summary.png")
    plt.close(fig)


def save_velocity_trends(rows: list[dict], fig_dir: Path) -> None:
    """Plot velocity against sample mass response to expose setting-level trends."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    velocities = row_values(rows, "velocity_value")
    changes = row_values(rows, "sample_mass_change_g")
    ax.scatter(velocities, changes, s=70, color="#1f4e79")
    for row in rows:
        label = str(row["experiment"]).split("_", 1)[1]
        ax.annotate(
            label,
            (row["velocity_value"], row["sample_mass_change_g"]),
            fontsize=7,
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Raster velocity (mm/s)")
    ax.set_ylabel("After - before sample weight (g)")
    ax.set_title("Sample Mass Change vs Raster Velocity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "velocity_vs_sample_mass_change.png")
    plt.close(fig)


def save_temperature_summary(rows: list[dict], fig_dir: Path) -> None:
    """Plot temperature fields when handwritten info.txt includes them."""
    temp_cols = [
        col
        for col in ["temp_1_value", "temp_2_value", "temp_3_value"]
        if any(col in row for row in rows)
    ]
    temp_rows = [row for row in rows if any(col in row for col in temp_cols)]
    if not temp_cols or not temp_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.22
    x = list(range(len(temp_rows)))
    colors = ["#1f4e79", "#d96c2c", "#2e7d32"]
    offsets = [width * (idx - (len(temp_cols) - 1) / 2.0) for idx in range(len(temp_cols))]
    for offset, col, color in zip(offsets, temp_cols, colors, strict=False):
        values = [row.get(col, float("nan")) for row in temp_rows]
        ax.bar(
            [idx + offset for idx in x],
            values,
            width=width,
            label=col.replace("_value", ""),
            color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_labels(temp_rows), rotation=35, ha="right")
    ax.set_ylabel("Temperature")
    ax.set_title("Recorded Temperatures")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "temperature_summary.png")
    plt.close(fig)


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    """Write the parsed table, a compact summary, and all figures."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "info_txt_summary.csv")
    write_markdown_summary(rows, out_dir / "info_txt_summary.md")
    save_weight_before_after(rows, fig_dir)
    save_sample_mass_change(rows, fig_dir)
    save_collected_mass_summary(rows, fig_dir)
    save_velocity_trends(rows, fig_dir)
    save_temperature_summary(rows, fig_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = load_info_table(args.data_root)
    out_dir = args.out if args.out is not None else args.data_root / "info_txt_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(rows, out_dir)
    print(f"Loaded {len(rows)} successful experiments.")
    print(f"Wrote summary and figures to {out_dir}")


if __name__ == "__main__":
    main()

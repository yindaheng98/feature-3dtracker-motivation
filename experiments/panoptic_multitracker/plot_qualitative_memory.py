#!/usr/bin/env python3
"""Plot sampled PyTorch CUDA allocator usage for qualitative tracker runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = ("opend4rt", "spatracker", "mvtap", "lapa")
LABELS = {
    "opend4rt": "Open-d4rt",
    "spatracker": "SpaTrackerV2",
    "mvtap": "MV-TAP",
    "lapa": "LAPA",
}
PLOT_EVENTS = {
    "opend4rt": {"before_tracking": "tracking start", "after_tracking": "tracking end"},
    "spatracker": {
        "before_front": "Front start",
        "after_front": "Front end",
        "before_tracker": "Tracker start",
        "after_tracker": "Tracker end",
    },
    "mvtap": {"before_tracking": "tracking start", "after_tracking": "tracking end"},
    "lapa": {
        "cotracker_on_device": "CoTracker start",
        "cotracker_view_2_done": "CoTracker end",
        "dino_on_device": "DINO start",
        "dino_view_2_done": "DINO end",
        "before_lapa": "LAPA start",
        "after_lapa": "LAPA end",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dirs", type=Path, nargs="+")
    return parser.parse_args()


def read_trace(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty memory trace: {path}")
    numeric = (
        "elapsed_s",
        "allocated_mib",
        "reserved_mib",
        "peak_allocated_mib",
        "peak_reserved_mib",
    )
    return [
        {**row, **{key: float(row[key]) for key in numeric}}
        for row in rows
    ]


def plot_scene(scene_dir: Path) -> None:
    manifest = json.loads((scene_dir / "manifest.json").read_text())
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    summaries = {}
    for model, axis in zip(MODELS, axes.flat):
        rows = read_trace(scene_dir / "memory" / f"{model}.csv")
        elapsed = np.asarray([row["elapsed_s"] for row in rows], np.float64)
        allocated = np.asarray([row["allocated_mib"] for row in rows], np.float64)
        reserved = np.asarray([row["reserved_mib"] for row in rows], np.float64)
        peak_allocated = max(float(row["peak_allocated_mib"]) for row in rows)
        peak_reserved = max(float(row["peak_reserved_mib"]) for row in rows)
        events = [row for row in rows if row["event"]]
        completed = bool(events) and events[-1]["event"] == "runner_end"
        if not completed:
            raise RuntimeError(f"Refusing incomplete memory trace for {model}")

        axis.plot(elapsed, allocated / 1024, label="allocated (active tensors)", linewidth=1.8)
        axis.plot(elapsed, reserved / 1024, label="reserved (allocator capacity)", linewidth=1.5)
        plotted_events = [row for row in events if row["event"] in PLOT_EVENTS[model]]
        for row in plotted_events:
            event_time = float(row["elapsed_s"])
            axis.axvline(event_time, color="black", alpha=0.16, linewidth=0.7)
            axis.text(
                event_time,
                0.98,
                PLOT_EVENTS[model][str(row["event"])],
                rotation=90,
                transform=axis.get_xaxis_transform(),
                va="top",
                ha="right",
                fontsize=6,
                alpha=0.65,
            )
        axis.set_title(
            f"{LABELS[model]} — peak allocated {peak_allocated / 1024:.2f} GiB, "
            f"reserved {peak_reserved / 1024:.2f} GiB"
        )
        axis.set_xlabel("wall-clock seconds")
        axis.set_ylabel("PyTorch CUDA memory (GiB)")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize=8)
        summaries[model] = {
            "completed": completed,
            "samples": len(rows),
            "duration_s": float(elapsed[-1]),
            "sample_interval_median_ms": (
                float(np.median(np.diff(elapsed)) * 1000) if len(elapsed) > 1 else None
            ),
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
            "events": [
                {"elapsed_s": float(row["elapsed_s"]), "name": str(row["event"])}
                for row in events
            ],
        }

    scene = str(manifest["scene"])
    figure.suptitle(
        f"{scene}: sampled PyTorch CUDA allocator usage — "
        f"{manifest['frames']} frames, {manifest['points']} queries\n"
        "The x-axis is process wall time, not video-frame progress; model setup is included."
    )
    figure.savefig(scene_dir / "cuda_memory_over_time.png", dpi=180)
    plt.close(figure)
    (scene_dir / "memory_summary.json").write_text(
        json.dumps(
            {
                "scene": scene,
                "frames": manifest["frames"],
                "points": manifest["points"],
                "measurement": (
                    "PyTorch allocator sampled in a background thread; allocated is active "
                    "tensor memory, reserved is allocator-managed capacity, and neither is "
                    "whole-process nvidia-smi memory"
                ),
                "models": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Plotted memory trace for {scene} to {scene_dir}")


def main() -> None:
    args = parse_args()
    for scene_dir in args.scene_dirs:
        plot_scene(scene_dir.resolve())


if __name__ == "__main__":
    main()

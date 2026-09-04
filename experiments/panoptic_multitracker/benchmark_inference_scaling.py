#!/usr/bin/env python3
"""Run and plot point-count/frame-count inference scaling benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
EXPERIMENT = ROOT / "experiments/panoptic_multitracker"
MODEL_LABELS = {
    "opend4rt": "Open-d4rt",
    "spatracker": "SpaTrackerV2",
    "mvtap": "MV-TAP",
    "lapa": "LAPA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_LABELS), default=list(MODEL_LABELS)
    )
    parser.add_argument("--point-counts", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--frame-counts", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--fixed-frames", type=int, default=16)
    parser.add_argument("--fixed-points", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/panoptic_multitracker/inference_scaling",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def one_path(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {pattern}, found {len(matches)}")
    return matches[0]


def model_command(
    model: str,
    frames: int,
    points: int,
    output: Path,
    warmup: int,
    repeats: int,
) -> list[str]:
    common_timing = [
        "--timing-warmup",
        str(warmup),
        "--timing-repeats",
        str(repeats),
    ]
    source = ROOT / "data/panoptic_tracking/tapvid3d_minival/tap3d_juggle/juggle_7.npz"
    if model == "opend4rt":
        config = one_path(
            "checkpoints/huggingface/hub/models--Lijiaxin0111--OpenD4RT/"
            "snapshots/*/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
        )
        checkpoint = config.with_name("opend4rt.ckpt")
        return [
            str(PYTHON),
            str(EXPERIMENT / "run_d4rt.py"),
            str(source),
            "--model-config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--device",
            "cuda",
            "--frames",
            str(frames),
            "--max-points",
            str(points),
            "--query-chunk-size",
            "32",
            *common_timing,
        ]
    if model == "spatracker":
        return [
            str(PYTHON),
            str(EXPERIMENT / "run_spatracker.py"),
            str(source),
            "--output",
            str(output),
            "--device",
            "cuda:0",
            "--frames",
            str(frames),
            "--max-points",
            str(points),
            "--support-points",
            "64",
            *common_timing,
        ]
    if model == "mvtap":
        return [
            str(PYTHON),
            str(EXPERIMENT / "run_mvtap.py"),
            "--mc-dir",
            str(ROOT / "output/panoptic_multitracker/lapa/mc"),
            "--npz-root",
            str(ROOT / "data/panoptic_tracking/tapvid3d_minival"),
            "--checkpoint",
            str(ROOT / "checkpoints/MVTAP.ckpt"),
            "--output",
            str(output),
            "--scene",
            "juggle",
            "--cameras",
            "7",
            "8",
            "9",
            "--frames",
            str(frames),
            "--max-points",
            str(points),
            "--device",
            "cuda:0",
            *common_timing,
        ]
    if model == "lapa":
        checkpoint = one_path(
            "checkpoints/huggingface/hub/models--bishoygaloaa--LAPA-Joint/"
            "snapshots/*/lapa.pt"
        )
        return [
            str(PYTHON),
            str(EXPERIMENT / "run_lapa.py"),
            "--checkpoint",
            str(checkpoint),
            "--mc-dir",
            str(ROOT / "output/panoptic_multitracker/lapa/mc"),
            "--feature-dir",
            str(ROOT / "output/panoptic_multitracker/lapa/features_eval"),
            "--data-root",
            str(EXPERIMENT / "lapa_juggle7"),
            "--output",
            str(output),
            "--device",
            "cuda:0",
            "--frames",
            str(frames),
            "--max-points",
            str(points),
            *common_timing,
        ]
    raise ValueError(model)


def run_signature(command: list[str], physical_gpu: str) -> dict:
    code_paths = [Path(command[1]), EXPERIMENT / "inference_timing.py"]
    code_sha256 = {}
    for path in code_paths:
        code_sha256[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = {}
    for index, token in enumerate(command[2:], start=2):
        if command[index - 1] == "--output":
            continue
        path = Path(token)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file() and path not in code_paths:
            stat = path.stat()
            inputs[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        elif path.is_dir():
            stat = path.stat()
            inputs[str(path)] = {"directory_mtime_ns": stat.st_mtime_ns}
    return {
        "command": command,
        "physical_gpu": physical_gpu,
        "code_sha256": code_sha256,
        "inputs": inputs,
    }


def valid_result(
    path: Path,
    warmup: int,
    repeats: int,
    frames: int,
    points: int,
    signature: dict,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
        timing = payload["timing"]
        if "front" in timing:
            timing = timing["front"]
        saved_signature = json.loads(path.with_suffix(".run.json").read_text())
        return (
            timing["warmup"] == warmup
            and timing["repeats"] == repeats
            and payload["frames"] == frames
            and payload["points"] == points
            and saved_signature == signature
        )
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return False


def run_configuration(
    model: str,
    frames: int,
    points: int,
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = model_command(model, frames, points, output, args.warmup, args.repeats)
    signature = run_signature(command, args.physical_gpu)
    if args.resume and valid_result(
        output, args.warmup, args.repeats, frames, points, signature
    ):
        print(f"reuse {model}: frames={frames} points={points}", flush=True)
        return command
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".log")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.physical_gpu,
            "HF_HOME": str(ROOT / "checkpoints/huggingface"),
            "TORCH_HOME": str(ROOT / "checkpoints/torch"),
            "HF_HUB_OFFLINE": "1",
            "MPLCONFIGDIR": str(args.output_dir / "matplotlib"),
        }
    )
    print(f"run   {model}: frames={frames} points={points}", flush=True)
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(
            f"{model} failed for frames={frames}, points={points} "
            f"with exit {completed.returncode}:\n{tail}"
        )
    payload = json.loads(output.read_text())
    if payload.get("frames") != frames or payload.get("points") != points:
        raise RuntimeError(
            f"{model} emitted frames={payload.get('frames')} points={payload.get('points')} "
            f"for requested frames={frames} points={points}"
        )
    output.with_suffix(".run.json").write_text(json.dumps(signature, indent=2) + "\n")
    return command


def timing_values(payload: dict) -> dict[str, object]:
    timing = payload["timing"]
    if "front" in timing:
        front = timing["front"]
        tracker = timing["tracker"]
        samples = {
            "front": front["samples_seconds"],
            "tracker": tracker["samples_seconds"],
        }
        median = front["median_seconds"] + tracker["median_seconds"]
        p25 = front["p25_seconds"] + tracker["p25_seconds"]
        p75 = front["p75_seconds"] + tracker["p75_seconds"]
        peak = max(
            front["peak_allocated_mib"],
            tracker["peak_allocated_mib"],
        )
        incremental = max(
            front["incremental_peak_mib"],
            tracker["incremental_peak_mib"],
        )
        method = "sum_of_separately_measured_stage_summaries"
    else:
        samples = np.asarray(timing["samples_seconds"], dtype=np.float64)
        median = float(np.median(samples))
        p25 = float(np.percentile(samples, 25))
        p75 = float(np.percentile(samples, 75))
        samples = samples.tolist()
        peak = timing["peak_allocated_mib"]
        incremental = timing["incremental_peak_mib"]
        method = "synchronized_forward_samples"
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "p25_seconds": p25,
        "p75_seconds": p75,
        "peak_allocated_mib": peak,
        "incremental_peak_mib": incremental,
        "timing_method": method,
    }


def plot_sweep(records: list[dict], sweep: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, model in zip(axes.flat, MODEL_LABELS):
        rows = sorted(
            (row for row in records if row["sweep"] == sweep and row["model"] == model),
            key=lambda row: row["value"],
        )
        if not rows:
            axis.set_visible(False)
            continue
        x = np.asarray([row["value"] for row in rows])
        y = np.asarray([row["median_seconds"] for row in rows])
        lower = y - np.asarray([row["p25_seconds"] for row in rows])
        upper = np.asarray([row["p75_seconds"] for row in rows]) - y
        axis.errorbar(x, y, yerr=np.stack([lower, upper]), marker="o", capsize=4)
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel("Tracked points" if sweep == "points" else "Input frames")
        axis.set_ylabel("Inference time (s)")
        axis.grid(True, alpha=0.3)
        for x_value, y_value in zip(x, y):
            axis.annotate(f"{y_value:.2f}s", (x_value, y_value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    figure.suptitle(
        "Inference latency vs tracked points (16 frames)"
        if sweep == "points"
        else "Inference latency vs input frames (32 points)"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = args.output_dir / "raw"
    commands = []
    result_paths: dict[tuple[str, int, int], Path] = {}
    for model in args.models:
        for points in args.point_counts:
            output = raw_root / model / f"frames_{args.fixed_frames}_points_{points}.json"
            commands.append(
                run_configuration(model, args.fixed_frames, points, output, args)
            )
            result_paths[(model, args.fixed_frames, points)] = output
        for frames in args.frame_counts:
            key = (model, frames, args.fixed_points)
            if key in result_paths:
                continue
            output = raw_root / model / f"frames_{frames}_points_{args.fixed_points}.json"
            commands.append(run_configuration(model, frames, args.fixed_points, output, args))
            result_paths[key] = output

    records = []
    for model in args.models:
        for sweep, values in (("points", args.point_counts), ("frames", args.frame_counts)):
            for value in values:
                frames = args.fixed_frames if sweep == "points" else value
                points = value if sweep == "points" else args.fixed_points
                path = result_paths[(model, frames, points)]
                payload = json.loads(path.read_text())
                if payload.get("frames") != frames or payload.get("points") != points:
                    raise RuntimeError(
                        f"Shape mismatch in {path}: expected ({frames}, {points}), "
                        f"observed ({payload.get('frames')}, {payload.get('points')})"
                    )
                row = {
                    "sweep": sweep,
                    "value": value,
                    "model": model,
                    "frames": frames,
                    "points": points,
                    **timing_values(payload),
                    "source_json": str(path.relative_to(ROOT)),
                }
                records.append(row)

    for model in args.models:
        for sweep in ("points", "frames"):
            rows = sorted(
                (r for r in records if r["model"] == model and r["sweep"] == sweep),
                key=lambda r: r["value"],
            )
            baseline = rows[0]["median_seconds"]
            for row in rows:
                row["relative_to_smallest"] = row["median_seconds"] / baseline

    (args.output_dir / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    fields = [key for key in records[0] if key != "samples_seconds"]
    with (args.output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    (args.output_dir / "commands.json").write_text(
        json.dumps([shlex.join(command) for command in commands], indent=2) + "\n"
    )
    plot_sweep(records, "points", args.output_dir / "inference_time_vs_points.png")
    plot_sweep(records, "frames", args.output_dir / "inference_time_vs_frames.png")
    print(f"Wrote {len(records)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()

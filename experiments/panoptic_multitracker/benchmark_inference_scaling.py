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
    parser.add_argument(
        "--sweeps", nargs="+", choices=("points", "frames"), default=["points", "frames"]
    )
    parser.add_argument(
        "--point-counts",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64, 96, 128, 160, 192, 221],
    )
    parser.add_argument(
        "--frame-counts",
        nargs="+",
        type=int,
        default=[8, 16, 32, 48, 64, 80, 96, 112, 128, 150],
    )
    parser.add_argument("--fixed-frames", type=int, default=16)
    parser.add_argument("--fixed-points", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--physical-gpu", default="1")
    parser.add_argument(
        "--lapa-feature-dir",
        type=Path,
        default=ROOT / "output/panoptic_multitracker/lapa/features_eval_221",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/panoptic_multitracker/inference_scaling",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--record-oom",
        action="store_true",
        help="Record the first CUDA OOM per model and sweep, then skip larger values",
    )
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
    lapa_feature_dir: Path,
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
            str(lapa_feature_dir.resolve()),
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


def valid_oom(path: Path, frames: int, points: int, signature: dict) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
        return (
            payload["status"] == "oom"
            and payload["frames"] == frames
            and payload["points"] == points
            and payload["signature"] == signature
        )
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return False


def is_cuda_oom(log_text: str) -> bool:
    lowered = log_text.lower()
    return "cuda out of memory" in lowered or "torch.outofmemoryerror" in lowered


def run_configuration(
    model: str,
    frames: int,
    points: int,
    output: Path,
    args: argparse.Namespace,
) -> dict:
    command = model_command(
        model,
        frames,
        points,
        output,
        args.warmup,
        args.repeats,
        args.lapa_feature_dir,
    )
    signature = run_signature(command, args.physical_gpu)
    if args.resume and valid_result(
        output, args.warmup, args.repeats, frames, points, signature
    ):
        print(f"reuse {model}: frames={frames} points={points}", flush=True)
        return {
            "status": "ok",
            "model": model,
            "frames": frames,
            "points": points,
            "command": command,
            "path": output,
        }
    oom_path = output.with_suffix(".oom.json")
    if args.record_oom and args.resume and valid_oom(oom_path, frames, points, signature):
        print(f"reuse OOM {model}: frames={frames} points={points}", flush=True)
        return {
            "status": "oom",
            "model": model,
            "frames": frames,
            "points": points,
            "command": command,
            "path": oom_path,
        }
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
        log_text = log_path.read_text(errors="replace")
        tail = "\n".join(log_text.splitlines()[-30:])
        if args.record_oom and is_cuda_oom(log_text):
            payload = {
                "status": "oom",
                "model": model,
                "frames": frames,
                "points": points,
                "returncode": completed.returncode,
                "log": str(log_path.relative_to(ROOT)),
                "signature": signature,
            }
            oom_path.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"OOM   {model}: frames={frames} points={points}", flush=True)
            return {**payload, "command": command, "path": oom_path}
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
    return {
        "status": "ok",
        "model": model,
        "frames": frames,
        "points": points,
        "command": command,
        "path": output,
    }


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


def add_oom_region(
    axis, failures: list[dict], model: str, sweep: str, successful_y: np.ndarray
) -> bool:
    actual = sorted(
        (
            row
            for row in failures
            if row["model"] == model and row["sweep"] == sweep and row["status"] == "oom"
        ),
        key=lambda row: row["value"],
    )
    if not actual:
        return False
    first = actual[0]
    related = [
        row["value"]
        for row in failures
        if row["model"] == model and row["sweep"] == sweep
    ]
    right = max(related)
    marker_y = max(float(np.max(successful_y)) * 1.08, 0.05)
    axis.axvspan(first["value"], right, color="red", alpha=0.08, label="not run after OOM")
    axis.scatter(
        [first["value"]], [marker_y], color="red", marker="X", s=80, zorder=5,
        label="observed CUDA OOM",
    )
    axis.annotate(
        "OOM\n(measurement unavailable)",
        (first["value"], marker_y),
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        va="top",
        color="darkred",
        fontsize=8,
    )
    bottom, top = axis.get_ylim()
    axis.set_ylim(bottom, max(top, marker_y * 1.12))
    return True


def plot_sweep(
    records: list[dict], sweep: str, output: Path, failures: list[dict]
) -> None:
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
        show_legend = False
        exponent = None
        if len(x) >= 3:
            exponent, _ = np.polyfit(np.log(x), np.log(y), 1)
            quadratic = y[-1] * np.square(x / x[-1])
            reference_label = "N² reference" if sweep == "points" else "T² reference"
            axis.plot(x, quadratic, color="gray", linestyle=":", label=reference_label)
            show_legend = True
        title = MODEL_LABELS[model]
        if exponent is not None:
            title += f" — empirical p={exponent:.2f}"
        axis.set_title(title)
        axis.set_xlabel("Tracked points" if sweep == "points" else "Input frames")
        axis.set_ylabel("Inference time (s)")
        axis.grid(True, alpha=0.3)
        for x_value, y_value in zip(x, y):
            axis.annotate(f"{y_value:.2f}s", (x_value, y_value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        show_legend |= add_oom_region(axis, failures, model, sweep, y)
        if show_legend:
            axis.legend(fontsize=7, loc="lower right")
    controls = sorted(
        {
            row["frames"] if sweep == "points" else row["points"]
            for row in records
            if row["sweep"] == sweep
        }
    )
    control = controls[0] if len(controls) == 1 else "mixed"
    figure.suptitle(
        f"Inference latency vs tracked points ({control} frames)"
        if sweep == "points"
        else f"Inference latency vs input frames ({control} points)"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_memory_sweep(
    records: list[dict], sweep: str, output: Path, failures: list[dict]
) -> None:
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
        peak = np.asarray([row["peak_allocated_mib"] for row in rows]) / 1024
        incremental = np.asarray([row["incremental_peak_mib"] for row in rows]) / 1024
        axis.plot(x, peak, marker="o", label="absolute peak allocated")
        axis.plot(
            x,
            incremental,
            marker="s",
            linestyle="--",
            label="incremental forward peak",
        )
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel("Tracked points" if sweep == "points" else "Input frames")
        axis.set_ylabel("Peak PyTorch allocated memory (GiB)")
        axis.set_ylim(bottom=0)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        for x_value, y_value in zip(x, peak):
            axis.annotate(
                f"{y_value:.2f}",
                (x_value, y_value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_ylim(0, max(float(np.max(peak)) * 1.12, 0.05))
        add_oom_region(axis, failures, model, sweep, peak)
        axis.legend(fontsize=7, loc="best")
    controls = sorted(
        {
            row["frames"] if sweep == "points" else row["points"]
            for row in records
            if row["sweep"] == sweep
        }
    )
    control = controls[0] if len(controls) == 1 else "mixed"
    figure.suptitle(
        f"Peak allocated CUDA memory vs tracked points ({control} frames)"
        if sweep == "points"
        else f"Peak allocated CUDA memory vs input frames ({control} points)"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def fit_scaling(records: list[dict], failures: list[dict], sweep: str) -> dict:
    variable = "points" if sweep == "points" else "frames"
    fits = {}
    for model in MODEL_LABELS:
        rows = sorted(
            (row for row in records if row["sweep"] == sweep and row["model"] == model),
            key=lambda row: row["value"],
        )
        if len(rows) < 3:
            continue
        x = np.asarray([row["value"] for row in rows], np.float64)
        y = np.asarray([row["median_seconds"] for row in rows], np.float64)
        exponent, log_scale = np.polyfit(np.log(x), np.log(y), 1)
        oom_values = sorted(
            row["value"]
            for row in failures
            if row["model"] == model
            and row["sweep"] == sweep
            and row["status"] == "oom"
        )
        fits[model] = {
            "model": model,
            "sweep": sweep,
            "fit": f"log(time_seconds) = exponent * log({variable}) + log_scale",
            "exponent": float(exponent),
            "log_scale": float(log_scale),
            f"successful_{variable}_counts": x.astype(int).tolist(),
            f"first_cuda_oom_{variable}": oom_values[0] if oom_values else None,
            "interpretation": "empirical fit over successful measurements; not a complexity proof",
        }
    return fits


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = args.output_dir / "raw"
    commands = []
    outcomes: dict[tuple[str, int, int], dict] = {}
    failures = []
    for model in args.models:
        first_point_oom = None
        point_counts = sorted(set(args.point_counts)) if "points" in args.sweeps else []
        for points in point_counts:
            if first_point_oom is not None:
                failures.append(
                    {
                        "sweep": "points",
                        "value": points,
                        "model": model,
                        "frames": args.fixed_frames,
                        "points": points,
                        "status": "not_run_after_oom",
                        "first_oom_points": first_point_oom,
                    }
                )
                continue
            output = raw_root / model / f"frames_{args.fixed_frames}_points_{points}.json"
            outcome = run_configuration(
                model, args.fixed_frames, points, output, args
            )
            commands.append(outcome["command"])
            outcomes[(model, args.fixed_frames, points)] = outcome
            if outcome["status"] == "oom":
                first_point_oom = points
                failures.append(
                    {
                        "sweep": "points",
                        "value": points,
                        "model": model,
                        "frames": args.fixed_frames,
                        "points": points,
                        "status": "oom",
                        "source_json": str(Path(outcome["path"]).relative_to(ROOT)),
                    }
                )

        first_oom = None
        frame_counts = sorted(set(args.frame_counts)) if "frames" in args.sweeps else []
        for frames in frame_counts:
            key = (model, frames, args.fixed_points)
            if first_oom is not None:
                failures.append(
                    {
                        "sweep": "frames",
                        "value": frames,
                        "model": model,
                        "frames": frames,
                        "points": args.fixed_points,
                        "status": "not_run_after_oom",
                        "first_oom_frames": first_oom,
                    }
                )
                continue
            if key in outcomes:
                outcome = outcomes[key]
            else:
                output = raw_root / model / f"frames_{frames}_points_{args.fixed_points}.json"
                outcome = run_configuration(model, frames, args.fixed_points, output, args)
                commands.append(outcome["command"])
                outcomes[key] = outcome
            if outcome["status"] == "oom":
                first_oom = frames
                failures.append(
                    {
                        "sweep": "frames",
                        "value": frames,
                        "model": model,
                        "frames": frames,
                        "points": args.fixed_points,
                        "status": "oom",
                        "source_json": str(Path(outcome["path"]).relative_to(ROOT)),
                    }
                )

    records = []
    for model in args.models:
        requested_sweeps = (
            ("points", args.point_counts),
            ("frames", args.frame_counts),
        )
        for sweep, values in requested_sweeps:
            if sweep not in args.sweeps:
                continue
            for value in values:
                frames = args.fixed_frames if sweep == "points" else value
                points = value if sweep == "points" else args.fixed_points
                outcome = outcomes.get((model, frames, points))
                if outcome is None or outcome["status"] != "ok":
                    continue
                path = Path(outcome["path"])
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
            if not rows:
                continue
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
    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    fits = fit_scaling(records, failures, "frames")
    (args.output_dir / "frame_scaling_fits.json").write_text(
        json.dumps(fits, indent=2) + "\n"
    )
    point_fits = fit_scaling(records, failures, "points")
    (args.output_dir / "point_scaling_fits.json").write_text(
        json.dumps(point_fits, indent=2) + "\n"
    )
    if "points" in args.sweeps:
        plot_sweep(
            records, "points", args.output_dir / "inference_time_vs_points.png", failures
        )
        plot_memory_sweep(
            records, "points", args.output_dir / "peak_memory_vs_points.png", failures
        )
    if "frames" in args.sweeps:
        plot_sweep(
            records, "frames", args.output_dir / "inference_time_vs_frames.png", failures
        )
        plot_memory_sweep(
            records, "frames", args.output_dir / "peak_memory_vs_frames.png", failures
        )
    print(
        f"Wrote {len(records)} successful rows and {len(failures)} failure/skip rows "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()

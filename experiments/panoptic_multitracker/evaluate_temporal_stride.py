#!/usr/bin/env python3
"""Apply one 3D metric protocol and plot the temporal-stride pilot."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPA_ROOT = ROOT / "SpaTrackerV2"
sys.path.insert(0, str(SPA_ROOT))

from models.SpaTrackV2.evaluation.core.tapvid3d_metrics import (  # noqa: E402
    compute_tapvid3d_metrics,
)


MODELS = ("opend4rt", "spatracker", "mvtap", "lapa")


def world_to_camera(points: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    return points @ w2c[:3, :3].T + w2c[:3, 3]


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as pack:
        if "pred_world" in pack.files:
            w2c = np.asarray(pack["w2c_ref"], dtype=np.float64)
            pred = world_to_camera(np.asarray(pack["pred_world"], dtype=np.float64), w2c)
            gt = world_to_camera(np.asarray(pack["gt_world"], dtype=np.float64), w2c)
        else:
            pred = np.asarray(pack["pred_xyz"], dtype=np.float64)
            gt = np.asarray(pack["gt_xyz"], dtype=np.float64)
        pred_visible = np.asarray(pack["pred_visible"], dtype=bool)
        gt_visible = np.asarray(pack["gt_visible"], dtype=bool)
        if gt_visible.ndim == 3:
            gt_visible = gt_visible[0]
        if pred_visible.ndim == 3:
            pred_visible = np.isfinite(pred).all(-1)
        if "intrinsics" in pack.files:
            intrinsics = np.asarray(pack["intrinsics"], dtype=np.float64)
        else:
            source = (
                path.parents[2]
                / "source"
                / "tapvid3d_minival"
                / "tap3d_juggle"
                / "juggle_7.npz"
            )
            with np.load(source) as source_pack:
                intrinsics = np.asarray(source_pack["fx_fy_cx_cy"], dtype=np.float64)
    return pred, pred_visible, gt, gt_visible, intrinsics


def score(path: Path) -> dict[str, float]:
    pred, pred_visible, gt, gt_visible, intrinsics = load_prediction(path)
    finite = np.isfinite(pred).all(-1)
    query_valid = finite[0] & gt_visible[0]
    if not np.any(query_valid):
        raise RuntimeError(f"no finite query-frame predictions in {path}")
    scale = float(
        np.median(np.linalg.norm(gt[0, query_valid], axis=-1))
        / max(np.median(np.linalg.norm(pred[0, query_valid], axis=-1)), 1e-12)
    )
    pred = pred * scale
    pred_visible &= finite
    pred_safe = pred.copy()
    pred_safe[~finite] = 1e9

    # Query frame is deliberately excluded from the common comparison.
    m = compute_tapvid3d_metrics(
        gt_occluded=np.transpose(~gt_visible[1:], (1, 0)),
        gt_tracks=np.transpose(gt[1:], (1, 0, 2)),
        pred_occluded=np.transpose(~pred_visible[1:], (1, 0)),
        pred_tracks=np.transpose(pred_safe[1:], (1, 0, 2)),
        intrinsics_params=intrinsics,
        scaling="none",
        order="n t",
    )
    eval_gt = gt_visible[1:]
    eval_finite = finite[1:]
    covered = eval_gt & eval_finite & pred_visible[1:]
    errors = np.linalg.norm(pred[1:] - gt[1:], axis=-1)
    return {
        "aj3d": float(np.asarray(m["average_jaccard"]).mean()),
        "apd3d": float(np.asarray(m["average_pts_within_thresh"]).mean()),
        "oa": float(np.asarray(m["occlusion_accuracy"]).mean()),
        "epe_m": float(errors[eval_gt & eval_finite].mean()),
        "coverage": float(covered.sum() / max(eval_gt.sum(), 1)),
        "query_scale": scale,
        "frames": int(gt.shape[0]),
        "points": int(gt.shape[1]),
    }


def plot_curves(rows: list[dict], output: Path) -> None:
    labels = {
        "aj3d": "3D average Jaccard",
        "apd3d": "3D APD",
        "oa": "Occlusion accuracy",
        "epe_m": "EPE (m, lower is better)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for metric, ax in zip(labels, axes.flat):
        for model in MODELS:
            selected = sorted((row for row in rows if row["model"] == model), key=lambda x: x["stride"])
            ax.plot([row["stride"] for row in selected], [row[metric] for row in selected], marker="o", label=model)
        ax.set_xlabel("Source-frame stride (30 FPS)")
        ax.set_ylabel(labels[metric])
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("PStudio juggle_7: fixed 16 input frames, query frame excluded")
    fig.savefig(output, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for model in MODELS:
        selected = sorted((row for row in rows if row["model"] == model), key=lambda x: x["stride"])
        baseline = selected[0]["aj3d"]
        retention = [row["aj3d"] / max(baseline, 1e-12) for row in selected]
        ax.plot([row["stride"] for row in selected], retention, marker="o", label=model)
    ax.set_xlabel("Source-frame stride (30 FPS)")
    ax.set_ylabel("3D AJ retention vs stride 1")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output.with_name("aj3d_retention_vs_stride.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--strides", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8])
    args = parser.parse_args()
    rows = []
    for stride in args.strides:
        for model in MODELS:
            path = args.root / f"stride_{stride}" / "results" / model / "prediction.npz"
            row = {"stride": stride, "model": model, **score(path)}
            rows.append(row)

    summary = {"protocol": "fixed_16_frames_query_excluded_query_frame_scale", "rows": rows}
    (args.root / "common_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.root / "common_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_curves(rows, args.root / "accuracy_vs_stride.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

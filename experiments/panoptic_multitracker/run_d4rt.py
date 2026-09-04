#!/usr/bin/env python3
"""Run Open-d4rt on one compact PStudio sequence and retain raw predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
D4RT_ROOT = ROOT / "Open-d4rt"
sys.path.insert(0, str(D4RT_ROOT))

from eval_track3d_in_worldtrack import (  # noqa: E402
    _metrics_for_sequence,
    load_worldtrack_sequence,
)
from infer_track_3d import (  # noqa: E402
    _infer_tracks,
    _resize_video,
    _resolve_device,
    _unwrap_state_dict,
)
from src.core import load_checkpoint, load_yaml_config, seed_everything  # noqa: E402
from src.model import build_model  # noqa: E402
from inference_timing import measure_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--timing-warmup", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=1)
    args = parser.parse_args()

    cfg = load_yaml_config(args.model_config)
    seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)
    model = build_model(cfg["model"]).eval()
    state = _unwrap_state_dict(load_checkpoint(args.checkpoint, map_location="cpu"))
    loaded = model.load_state_dict(state, strict=False)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={len(loaded.missing_keys)} "
            f"unexpected={len(loaded.unexpected_keys)}"
        )
    device = _resolve_device(args.device)
    model.to(device).eval()

    sample = load_worldtrack_sequence(
        args.sequence, num_frames=args.frames if args.frames is not None else 1_000_000
    )
    video = sample["video_rgb"]
    height, width = video.shape[1:3]
    model_h, model_w = [int(v) for v in cfg.get_path("model.input.image_size", [256, 256])]
    visible0 = np.asarray(sample["visibility"][0], dtype=bool)
    query_uv = np.asarray(sample["tracks_uv"][0, visible0], dtype=np.float32)
    depth0 = np.asarray(sample["tracks_xyz_cam"][0, visible0, 2])
    finite = np.isfinite(query_uv).all(-1) & np.isfinite(depth0) & (depth0 > 0)
    query_uv = query_uv[finite]
    gt_xyz = np.asarray(sample["tracks_xyz_world"][:, visible0], dtype=np.float32)[:, finite]
    gt_visible = np.asarray(sample["visibility"][:, visible0], dtype=bool)[:, finite]
    if args.max_points is not None:
        query_uv = query_uv[: args.max_points]
        gt_xyz = gt_xyz[:, : args.max_points]
        gt_visible = gt_visible[:, : args.max_points]
    query_uv[:, 0] /= max(width - 1, 1)
    query_uv[:, 1] /= max(height - 1, 1)
    query_uv = np.clip(query_uv, 0.0, 1.0)

    resized_video = _resize_video(video, image_hw=(model_h, model_w))
    pred, timing = measure_inference(
        lambda: _infer_tracks(
            model=model,
            video_model_rgb=resized_video,
            native_aspect_ratio=width / max(height, 1),
            query_uv_norm=query_uv,
            query_chunk_size=args.query_chunk_size,
        ),
        device,
        args.timing_warmup,
        args.timing_repeats,
    )
    pred_xyz = np.asarray(pred["tracks_xyz_ref0"], dtype=np.float32).transpose(1, 0, 2)
    pred_visible = np.asarray(pred["tracks_visibility"], dtype=bool).T
    metrics = _metrics_for_sequence(gt_xyz, pred_xyz, compute_dyn=True)
    metrics.update(
        {
            "frames": int(video.shape[0]),
            "points": int(gt_xyz.shape[1]),
            "checkpoint_missing": 0,
            "checkpoint_unexpected": 0,
            "timing": timing,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        pred_xyz=pred_xyz,
        pred_visible=pred_visible,
        gt_xyz=gt_xyz,
        gt_visible=gt_visible,
        intrinsics=sample["intrinsics"],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

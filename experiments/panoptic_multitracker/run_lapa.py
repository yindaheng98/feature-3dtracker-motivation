#!/usr/bin/env python3
"""Run LAPA on one prepared sparse-frame sample and retain raw predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LAPA_ROOT = ROOT / "Look-Around-and-Pay-Attention-LAPA-"
sys.path.insert(0, str(LAPA_ROOT))

from lapa.data.mc_dataset import TAPVid3DMCEvalDataset  # noqa: E402
from lapa.eval.protocol import score_tracks  # noqa: E402
from lapa.models.lapa import LAPA  # noqa: E402
from inference_timing import measure_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mc-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-points", type=int, default=64)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--timing-warmup", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=1)
    args = parser.parse_args()

    cache_paths = sorted(args.feature_dir.glob("*/ref*_cam*.h5"))
    if len(cache_paths) != 3:
        raise RuntimeError(f"expected 3 eval caches, found {len(cache_paths)}")
    cache_checks = []
    for path in cache_paths:
        with h5py.File(path) as f:
            observed = np.asarray(f["tracks_2d"], dtype=np.float32)
            gt = np.asarray(f["tracks_2d_gt"], dtype=np.float32)
            cache_checks.append(
                {
                    "path": str(path),
                    "use_cotracker": int(f.attrs["use_cotracker"]),
                    "non_gt_fraction": float(np.any(np.abs(observed - gt) > 1e-4, axis=-1).mean()),
                }
            )
    if any(item["use_cotracker"] != 1 or item["non_gt_fraction"] <= 0.5 for item in cache_checks):
        raise RuntimeError("LAPA cache failed the real-CoTracker gate")

    device = torch.device(args.device)
    model = LAPA(volume_size=16).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    loaded = model.load_state_dict(checkpoint["model"], strict=False)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={len(loaded.missing_keys)} "
            f"unexpected={len(loaded.unexpected_keys)}"
        )
    model.eval()
    dataset = TAPVid3DMCEvalDataset(
        mc_dir=str(args.mc_dir),
        feature_dir=str(args.feature_dir),
        eval_feature_dir=str(args.feature_dir),
        canonical_feature_dir=str(args.feature_dir),
        data_root=str(args.data_root),
        num_views=3,
        max_points=args.max_points,
        use_gt_tracks=False,
    )
    batch = dataset[0]
    frames = min(args.frames or batch["gt_world"].shape[0], batch["gt_world"].shape[0])
    points = batch["queries0"].shape[0]
    queries0 = batch["queries0"][:points].to(device)
    view_k_device = [value.to(device) for value in batch["view_K"]]
    view_w2c_device = [value.to(device) for value in batch["view_w2c_norm"]]
    view_pts_device = [
        [value[:points].to(device) for value in view[:frames]]
        for view in batch["view_points_2d_native"]
    ]
    view_feats_device = [
        [value[:points].to(device) for value in view[:frames]]
        for view in batch["view_features_native"]
    ]
    view_valid_device = [
        value[:frames, :points].to(device) for value in batch["visible_per_view"]
    ]

    def run_model():
        with torch.no_grad():
            return model(
                view_pts_device,
                view_feats_device,
                view_k_device,
                view_w2c_device,
                queries0,
                tuple(batch["image_size"]),
                view_valid=view_valid_device,
            )

    output, timing = measure_inference(
        run_model, device, args.timing_warmup, args.timing_repeats
    )
    pred_norm = output["points_3d"].float().cpu().numpy()
    pred = {
        "pred_world": pred_norm * batch["aabb_half"].numpy() + batch["aabb_center"].numpy(),
        "pred_visible": output["vis_logits"].float().cpu().numpy() > 0,
        "gt_world": batch["gt_world"][:frames, :points].numpy(),
        "gt_visible": batch["visible"][:frames, :points].numpy(),
    }
    k = batch["view_K"][0].numpy()
    w2c = batch["view_w2c_world"][0].numpy()
    intrinsics = np.asarray([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    metrics = score_tracks(
        pred_world=pred["pred_world"],
        gt_world=pred["gt_world"],
        pred_visible=pred["pred_visible"],
        gt_visible=pred["gt_visible"],
        w2c_ref=w2c,
        intrinsics=intrinsics,
        image_size=tuple(batch["image_size"]),
    )
    metrics.update(
        {
            "frames": int(pred["gt_world"].shape[0]),
            "points": int(pred["gt_world"].shape[1]),
            "checkpoint_missing": 0,
            "checkpoint_unexpected": 0,
            "cache_checks": cache_checks,
            "timing": timing,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        pred_world=pred["pred_world"],
        pred_visible=pred["pred_visible"],
        gt_world=pred["gt_world"],
        gt_visible=pred["gt_visible"],
        w2c_ref=w2c,
        intrinsics=intrinsics,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

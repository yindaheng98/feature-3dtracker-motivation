#!/usr/bin/env python3
"""Run SpaTrackerV2 on one official TAPVid-3D PStudio sequence."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SPA_ROOT = ROOT / "SpaTrackerV2"
sys.path.insert(0, str(SPA_ROOT))

from models.SpaTrackV2.evaluation.core.tapvid3d_metrics import (  # noqa: E402
    compute_tapvid3d_metrics,
)
from models.SpaTrackV2.models.predictor import Predictor  # noqa: E402
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import (  # noqa: E402
    VGGT4Track,
)
from models.SpaTrackV2.models.vggt4track.utils.load_fn import (  # noqa: E402
    preprocess_image,
)
from inference_timing import measure_inference  # noqa: E402


def decode_video(payload: np.ndarray, frames: int) -> torch.Tensor:
    images = [
        np.asarray(Image.open(io.BytesIO(bytes(item))).convert("RGB"), dtype=np.uint8).copy()
        for item in payload[:frames]
    ]
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()


def transform_queries(uv: np.ndarray, height: int, width: int) -> np.ndarray:
    new_width = 518
    new_height = round(height * (new_width / width) / 14) * 14
    crop_y = max((new_height - 518) // 2, 0)
    out = uv.copy().astype(np.float32)
    out[:, 0] *= new_width / width
    out[:, 1] = out[:, 1] * new_height / height - crop_y
    return out


def scalar_metrics(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {key: float(np.asarray(value).mean()) for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--max-points", type=int, default=32)
    parser.add_argument("--support-points", type=int, default=64)
    parser.add_argument("--timing-warmup", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=1)
    args = parser.parse_args()

    with np.load(args.sequence, allow_pickle=True) as pack:
        frames = min(args.frames, len(pack["images_jpeg_bytes"]))
        video = decode_video(pack["images_jpeg_bytes"], frames)
        gt_xyz = np.asarray(pack["tracks_XYZ"][:frames], dtype=np.float32)
        gt_visible = np.asarray(pack["visibility"][:frames], dtype=bool)
        intrinsics = np.asarray(pack["fx_fy_cx_cy"], dtype=np.float32)

    height, width = video.shape[-2:]
    fx, fy, cx, cy = intrinsics
    xyz0 = gt_xyz[0]
    uv0 = np.stack(
        [fx * xyz0[:, 0] / xyz0[:, 2] + cx, fy * xyz0[:, 1] / xyz0[:, 2] + cy],
        axis=-1,
    )
    valid = gt_visible[0] & np.isfinite(uv0).all(-1) & np.isfinite(xyz0).all(-1) & (xyz0[:, 2] > 0)
    selected = np.flatnonzero(valid)[: args.max_points]
    if not len(selected):
        raise RuntimeError("No valid frame-0 query points")
    gt_xyz = gt_xyz[:, selected]
    gt_visible = gt_visible[:, selected]
    query_uv = transform_queries(uv0[selected], height, width)
    queries = np.concatenate([np.zeros((len(selected), 1), np.float32), query_uv], axis=-1)

    device = torch.device(args.device)
    front_video = preprocess_image(video)
    front = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().to(device)
    front_input = front_video[None].to(device) / 255.0

    def run_front():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return front(front_input)

    front_out, front_timing = measure_inference(
        run_front, device, args.timing_warmup, args.timing_repeats
    )
    depth = front_out["points_map"][..., 2].squeeze(0).float().cpu().numpy()
    predicted_intrinsics = front_out["intrs"].squeeze(0).float().cpu().numpy()
    del front, front_out
    torch.cuda.empty_cache()

    tracker = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    tracker.spatrack.track_num = args.support_points
    tracker.eval()
    tracker.to(device)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
    def run_tracker():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return tracker.forward(
                front_video,
                depth=depth,
                intrs=predicted_intrinsics,
                extrs=extrinsics,
                queries=queries,
                full_point=True,
                iters_track=4,
                query_no_BA=True,
                fixed_cam=True,
                stage=1,
                support_frame=frames - 1,
            )

    result, tracker_timing = measure_inference(
        run_tracker, device, args.timing_warmup, args.timing_repeats
    )

    pred_xyz = result[4][..., :3].float().cpu().numpy()
    pred_visible = (result[6][..., 0].float().cpu().numpy() > 0.5)
    metrics = scalar_metrics(
        compute_tapvid3d_metrics(
            gt_occluded=~gt_visible,
            gt_tracks=gt_xyz,
            pred_occluded=~pred_visible,
            pred_tracks=pred_xyz,
            intrinsics_params=intrinsics,
            scaling="median",
            order="t n",
        )
    )
    joint = gt_visible & pred_visible & np.isfinite(pred_xyz).all(-1)
    scale = float(np.median(np.linalg.norm(gt_xyz[joint], axis=-1)) / np.median(np.linalg.norm(pred_xyz[joint], axis=-1)))
    metrics["epe_median_scaled_m"] = float(np.linalg.norm(pred_xyz[joint] * scale - gt_xyz[joint], axis=-1).mean())
    metrics.update(
        {
            "frames": frames,
            "points": len(selected),
            "scale": scale,
            "timing": {
                "median_seconds": front_timing["median_seconds"]
                + tracker_timing["median_seconds"],
                "front": front_timing,
                "tracker": tracker_timing,
            },
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        pred_xyz=pred_xyz,
        pred_visible=pred_visible,
        gt_xyz=gt_xyz,
        gt_visible=gt_visible,
        selected_indices=selected,
    )
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

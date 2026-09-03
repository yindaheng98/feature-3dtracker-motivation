#!/usr/bin/env python3
"""Run MV-TAP on calibrated views derived from one PStudio reference clip."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MVTAP_ROOT = ROOT / "MV-TAP"
sys.path.insert(0, str(MVTAP_ROOT))

import model_utils  # noqa: E402
from models.mvtap import MVTAP  # noqa: E402


def decode_video(npz_path: Path, frames: int) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as pack:
        return np.stack(
            [
                np.asarray(Image.open(io.BytesIO(bytes(item))).convert("RGB"), dtype=np.uint8).copy()
                for item in pack["images_jpeg_bytes"][:frames]
            ]
        )


def project(world: np.ndarray, k: np.ndarray, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = world @ w2c[:3, :3].T + w2c[:3, 3]
    depth = camera[..., 2]
    uvw = camera @ k.T
    uv = uvw[..., :2] / np.maximum(uvw[..., 2:], 1e-8)
    return uv.astype(np.float32), depth


def triangulate(uv: np.ndarray, projection: np.ndarray, valid: np.ndarray) -> np.ndarray:
    views, frames, points, _ = uv.shape
    result = np.full((frames, points, 3), np.nan, np.float32)
    for t in range(frames):
        for n in range(points):
            chosen = np.flatnonzero(valid[:, t, n])
            if len(chosen) < 2:
                continue
            rows = []
            for v in chosen:
                p = projection[v]
                x, y = uv[v, t, n]
                rows.extend((x * p[2] - p[0], y * p[2] - p[1]))
            _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
            homogeneous = vh[-1]
            if abs(homogeneous[3]) > 1e-8:
                result[t, n] = homogeneous[:3] / homogeneous[3]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc-dir", type=Path, required=True)
    parser.add_argument("--npz-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="juggle")
    parser.add_argument("--cameras", type=int, nargs="+", default=[7, 8, 9])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--max-points", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    meta = json.loads((args.mc_dir / f"{args.scene}_mc.json").read_text())
    lookup = {int(camera["cam_id"]): camera for camera in meta["cameras"]}
    cameras = [lookup[cam_id] for cam_id in args.cameras]
    ref_id = args.cameras[0]
    with np.load(args.mc_dir / args.scene / "tracks_world" / f"cam_{ref_id}.npz") as tracks:
        world = np.asarray(tracks["tracks_world"][: args.frames], dtype=np.float32)
        source_visible = np.asarray(tracks["visibility"][: args.frames], dtype=bool)
    frame_count = world.shape[0]

    videos = []
    trajectories = []
    visible_per_view = []
    intrinsics = []
    extrinsics = []
    for camera in cameras:
        cam_id = int(camera["cam_id"])
        npz_path = args.npz_root / f"tap3d_{args.scene}" / f"{args.scene}_{cam_id}.npz"
        frames = decode_video(npz_path, frame_count)
        k = np.asarray(camera["K"], dtype=np.float32)
        w2c = np.asarray(camera["w2c"], dtype=np.float32)
        uv, depth = project(world, k, w2c)
        height, width = frames.shape[1:3]
        visible = (
            source_visible
            & (depth > 1e-4)
            & (uv[..., 0] >= 0)
            & (uv[..., 0] < width)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] < height)
        )
        videos.append(frames)
        trajectories.append(uv)
        visible_per_view.append(visible)
        intrinsics.append(k)
        extrinsics.append(w2c)

    video_np = np.stack(videos)
    trajectory_np = np.stack(trajectories)
    visible_np = np.stack(visible_per_view)
    selected = np.flatnonzero(visible_np[:, 0].all(axis=0))[: args.max_points]
    if not len(selected):
        raise RuntimeError("No points visible in every view at frame 0")
    trajectory_np = trajectory_np[:, :, selected]
    visible_np = visible_np[:, :, selected]
    world = world[:, selected]

    video = torch.from_numpy(video_np).permute(0, 1, 4, 2, 3).float()
    views, times, channels, height, width = video.shape
    video = F.interpolate(
        video.reshape(views * times, channels, height, width),
        size=(384, 512),
        mode="bilinear",
        align_corners=False,
    ).reshape(views, times, channels, 384, 512)
    trajectory_np[..., 0] *= 512 / width
    trajectory_np[..., 1] *= 384 / height
    k_np = np.stack(intrinsics)
    k_np[:, 0] *= 512 / width
    k_np[:, 1] *= 384 / height
    w2c_np = np.stack(extrinsics)
    queries_np = np.concatenate(
        [np.zeros((*trajectory_np.shape[:1], len(selected), 1), np.float32), trajectory_np[:, 0]],
        axis=-1,
    )

    device = torch.device(args.device)
    model = MVTAP(
        window_len=16,
        stride=4,
        corr_radius=3,
        corr_levels=4,
        num_virtual_tracks=64,
        hidden_dim=256,
        latent_dim=128,
        model_resolution_H=384,
        model_resolution_W=512,
        use_checkpoint=False,
        view_att=True,
        use_cam_embed=True,
        bilinear_mode="border",
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {key.removeprefix("model."): value for key, value in payload["state_dict"].items()}
    loaded = model.load_state_dict(state, strict=False)
    print(f"checkpoint missing={len(loaded.missing_keys)} unexpected={len(loaded.unexpected_keys)}")
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("MV-TAP checkpoint did not load exactly")
    model.eval().to(device)

    batch = {
        "video": video[None].to(device),
        "trajectory": torch.from_numpy(trajectory_np)[None].to(device),
        "visibility": torch.from_numpy(visible_np)[None].to(device),
        "query_points": torch.from_numpy(queries_np)[None].to(device),
        "intrinsic": torch.from_numpy(np.repeat(k_np[:, None], frame_count, axis=1))[None].to(device),
        "extrinsic": torch.from_numpy(np.repeat(w2c_np[:, None], frame_count, axis=1))[None].to(device),
    }
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            batch["video"],
            batch["query_points"].clone(),
            batch["intrinsic"],
            batch["extrinsic"],
            iters=4,
            is_train=False,
        )
    native = {
        key: float(np.asarray(value).mean())
        for key, value in model_utils.eval_batch(batch, output).items()
    }
    pred_uv = output[0][0].float().cpu().numpy()
    pred_valid = (output[1][0] * output[2][0]).float().cpu().numpy() > 0.6
    projection = np.asarray([k_np[v] @ w2c_np[v, :3] for v in range(views)])
    pred_world = triangulate(pred_uv, projection, pred_valid)
    valid_3d = source_visible[:frame_count, selected] & np.isfinite(pred_world).all(-1)
    error = np.linalg.norm(pred_world[valid_3d] - world[valid_3d], axis=-1)
    thresholds = (0.01, 0.04, 0.16, 0.64, 2.56)
    native["triangulated_mpjpe_m"] = float(error.mean()) if len(error) else float("nan")
    native["triangulated_apd3d"] = float(np.mean([np.mean(error < threshold) for threshold in thresholds])) if len(error) else float("nan")
    native.update({"frames": frame_count, "points": len(selected), "views": views, "triangulated_samples": len(error)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(native, indent=2) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        pred_uv=pred_uv,
        pred_visible=pred_valid,
        pred_world=pred_world,
        gt_world=world,
        gt_uv=trajectory_np,
        gt_visible=visible_np,
        selected_indices=selected,
    )
    print(json.dumps(native, indent=2))


if __name__ == "__main__":
    main()

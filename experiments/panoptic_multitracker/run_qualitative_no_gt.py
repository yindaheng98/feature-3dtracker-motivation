#!/usr/bin/env python3
"""Run one pretrained tracker on a prepared unannotated sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from qualitative_no_gt_common import (
    load_manifest,
    load_video,
    project_world,
    resize_video_and_camera,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", required=True, choices=["opend4rt", "spatracker", "mvtap", "lapa"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--support-points", type=int, default=64)
    return parser.parse_args()


def only_path(pattern: str) -> Path:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one path for {pattern}, found {len(paths)}")
    return paths[0]


def manifest_values(manifest: dict) -> tuple[Path, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    return (
        Path(str(manifest["scene_root"])),
        manifest["frame_numbers"].astype(np.int64),
        [str(value) for value in manifest["camera_names"]],
        manifest["K"].astype(np.float32),
        manifest["w2c"].astype(np.float32),
        manifest["query_world"].astype(np.float32),
    )


def run_opend4rt(manifest: dict, device: torch.device, query_chunk_size: int) -> dict:
    d4rt_root = ROOT / "Open-d4rt"
    sys.path.insert(0, str(d4rt_root))
    from infer_track_3d import _infer_tracks, _resize_video, _unwrap_state_dict
    from src.core import load_checkpoint, load_yaml_config, seed_everything
    from src.model import build_model

    scene_root, frames, cameras, _, _, _ = manifest_values(manifest)
    video = load_video(scene_root, frames, cameras[0])
    height, width = video.shape[1:3]
    query_uv = manifest["query_uv_ref"].astype(np.float32)
    query_norm = query_uv / np.asarray([max(width - 1, 1), max(height - 1, 1)], np.float32)
    config_path = only_path(
        "checkpoints/huggingface/hub/models--Lijiaxin0111--OpenD4RT/"
        "snapshots/*/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
    )
    checkpoint_path = config_path.with_name("opend4rt.ckpt")
    config = load_yaml_config(config_path)
    seed_everything(int(config.get_path("experiment.seed", 42)), deterministic=True)
    model = build_model(config["model"]).eval()
    state = _unwrap_state_dict(load_checkpoint(checkpoint_path, map_location="cpu"))
    loaded = model.load_state_dict(state, strict=False)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Open-d4rt checkpoint mismatch")
    model.to(device).eval()
    model_height, model_width = [
        int(value) for value in config.get_path("model.input.image_size", [256, 256])
    ]
    prediction = _infer_tracks(
        model=model,
        video_model_rgb=_resize_video(video, (model_height, model_width)),
        native_aspect_ratio=width / height,
        query_uv_norm=query_norm,
        query_chunk_size=query_chunk_size,
    )
    pred_uv = prediction["tracks_uv_norm"].transpose(1, 0, 2)
    pred_uv *= np.asarray([max(width - 1, 1), max(height - 1, 1)], np.float32)
    return {
        "pred_uv": pred_uv,
        "pred_visible": prediction["tracks_visibility"].T,
        "pred_xyz": prediction["tracks_xyz_ref0"].transpose(1, 0, 2),
    }


def run_spatracker(
    manifest: dict, device: torch.device, support_points: int
) -> dict:
    spa_root = ROOT / "SpaTrackerV2"
    sys.path.insert(0, str(spa_root))
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

    scene_root, frames, cameras, _, _, _ = manifest_values(manifest)
    video_np = load_video(scene_root, frames, cameras[0])
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()
    height, width = video.shape[-2:]
    new_width = 518
    new_height = round(height * (new_width / width) / 14) * 14
    crop_y = max((new_height - 518) // 2, 0)
    query_uv = manifest["query_uv_ref"].astype(np.float32).copy()
    query_uv[:, 0] *= new_width / width
    query_uv[:, 1] = query_uv[:, 1] * new_height / height - crop_y
    queries = np.concatenate([np.zeros((len(query_uv), 1), np.float32), query_uv], axis=-1)

    processed = preprocess_image(video)
    front = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        front_out = front(processed[None].to(device) / 255.0)
    depth = front_out["points_map"][..., 2].squeeze(0).float().cpu().numpy()
    intrinsics = front_out["intrs"].squeeze(0).float().cpu().numpy()
    del front, front_out
    torch.cuda.empty_cache()

    tracker = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    tracker.spatrack.track_num = support_points
    tracker.eval()
    tracker.to(device)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], len(frames), axis=0)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = tracker.forward(
            processed,
            depth=depth,
            intrs=intrinsics,
            extrs=extrinsics,
            queries=queries,
            full_point=True,
            iters_track=4,
            query_no_BA=True,
            fixed_cam=True,
            stage=1,
            support_frame=len(frames) - 1,
        )
    pred_uv = result[5][..., :2].float().cpu().numpy()
    pred_uv[..., 0] *= width / new_width
    pred_uv[..., 1] = (pred_uv[..., 1] + crop_y) * height / new_height
    return {
        "pred_uv": pred_uv,
        "pred_visible": result[6][..., 0].float().cpu().numpy() > 0.5,
        "pred_xyz": result[4][..., :3].float().cpu().numpy(),
    }


def triangulate_tracks(
    uv: np.ndarray, projection: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    views, frames, points, _ = uv.shape
    world = np.full((frames, points, 3), np.nan, np.float32)
    for frame in range(frames):
        for point in range(points):
            chosen = np.flatnonzero(valid[:, frame, point])
            if len(chosen) < 2:
                continue
            rows = []
            for view in chosen:
                x, y = uv[view, frame, point]
                matrix = projection[view]
                rows.extend((x * matrix[2] - matrix[0], y * matrix[2] - matrix[1]))
            _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
            homogeneous = vh[-1]
            if abs(homogeneous[3]) > 1e-8:
                world[frame, point] = homogeneous[:3] / homogeneous[3]
    return world


def prepare_multiview(manifest: dict) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    scene_root, frames, cameras, K, w2c, query_world = manifest_values(manifest)
    videos = []
    scaled_K = []
    query_uv = []
    for view, camera in enumerate(cameras):
        video = load_video(scene_root, frames, camera)
        resized, camera_K = resize_video_and_camera(video, K[view])
        uv, depth = project_world(query_world, camera_K, w2c[view])
        if np.any(depth <= 0):
            raise RuntimeError(f"Initial queries behind camera {camera}")
        videos.append(resized)
        scaled_K.append(camera_K)
        query_uv.append(uv)
    return videos, np.stack(scaled_K), w2c, np.stack(query_uv)


def run_mvtap(manifest: dict, device: torch.device) -> dict:
    mvtap_root = ROOT / "MV-TAP"
    sys.path.insert(0, str(mvtap_root))
    from models.mvtap import MVTAP

    videos, K, w2c, query_uv = prepare_multiview(manifest)
    views, points = query_uv.shape[:2]
    frame_count = len(manifest["frame_numbers"])
    video = torch.from_numpy(np.stack(videos)).permute(0, 1, 4, 2, 3).float()
    queries = np.concatenate(
        [np.zeros((views, points, 1), np.float32), query_uv.astype(np.float32)], axis=-1
    )
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
    payload = torch.load(ROOT / "checkpoints/MVTAP.ckpt", map_location="cpu", weights_only=False)
    state = {key.removeprefix("model."): value for key, value in payload["state_dict"].items()}
    loaded = model.load_state_dict(state, strict=False)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("MV-TAP checkpoint mismatch")
    model.eval().to(device)
    intrinsic = torch.from_numpy(np.repeat(K[:, None], frame_count, axis=1))[None].to(device)
    extrinsic = torch.from_numpy(np.repeat(w2c[:, None], frame_count, axis=1))[None].to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            video[None].to(device),
            torch.from_numpy(queries)[None].to(device),
            intrinsic,
            extrinsic,
            iters=4,
            is_train=False,
        )
    pred_views = output[0][0].float().cpu().numpy()
    visible_views = (output[1][0] * output[2][0]).float().cpu().numpy() > 0.6
    projection = np.stack([K[view] @ w2c[view, :3] for view in range(views)])
    pred_world = triangulate_tracks(pred_views, projection, visible_views)
    native_height, native_width = int(manifest["heights"][0]), int(manifest["widths"][0])
    pred_ref = pred_views[0].copy()
    pred_ref[..., 0] *= native_width / 512
    pred_ref[..., 1] *= native_height / 384
    return {
        "pred_uv": pred_ref,
        "pred_visible": visible_views[0],
        "pred_world": pred_world,
        "pred_uv_views_512x384": pred_views,
        "pred_visible_views": visible_views,
    }


def run_lapa(manifest: dict, device: torch.device) -> dict:
    lapa_root = ROOT / "Look-Around-and-Pay-Attention-LAPA-"
    sys.path.insert(0, str(lapa_root))
    from lapa.features.precompute import DINOv2FeatureExtractor, get_cotracker, run_cotracker
    from lapa.models.lapa import LAPA, build_w2c_normalized

    videos, K, w2c, query_uv = prepare_multiview(manifest)
    query_world = manifest["query_world"].astype(np.float32)
    views, points = query_uv.shape[:2]
    frame_count = len(manifest["frame_numbers"])

    cotracker = get_cotracker(device)
    tracks, visibility = [], []
    for view in range(views):
        queries = np.concatenate(
            [query_uv[view], np.zeros((points, 1), np.float32)], axis=-1
        )
        track, visible = run_cotracker(videos[view], queries, device, model=cotracker)
        track[0] = query_uv[view]
        visible[0] = True
        tracks.append(track)
        visibility.append(visible)
    del cotracker
    torch.cuda.empty_cache()
    tracks = np.stack(tracks)
    visibility = np.stack(visibility)
    for view in range(views):
        visibility[view] &= (
            (tracks[view, ..., 0] >= 0)
            & (tracks[view, ..., 0] < 512)
            & (tracks[view, ..., 1] >= 0)
            & (tracks[view, ..., 1] < 384)
            & np.isfinite(tracks[view]).all(axis=-1)
        )

    dino = DINOv2FeatureExtractor(device)
    features = []
    for view in range(views):
        tokens = dino.extract_video_features(videos[view])
        feature = dino.sample_at_points(tokens, tracks[view], (512, 384)).astype(np.float32)
        feature[~visibility[view]] = 0
        features.append(feature)
        del tokens
    del dino
    torch.cuda.empty_cache()

    projection = np.stack([K[view] @ w2c[view, :3] for view in range(views)])
    cotracker_world = triangulate_tracks(tracks, projection, visibility)
    cloud = cotracker_world[np.isfinite(cotracker_world).all(axis=-1)]
    cloud = np.concatenate([cloud, query_world], axis=0)
    lower, upper = np.percentile(cloud, [2, 98], axis=0)
    center = ((lower + upper) / 2).astype(np.float32)
    half = np.maximum((upper - lower) / 2 * 1.2, 0.5).astype(np.float32)
    queries_norm = (query_world - center) / half

    model = LAPA(volume_size=16).to(device)
    checkpoint_path = only_path(
        "checkpoints/huggingface/hub/models--bishoygaloaa--LAPA-Joint/snapshots/*/lapa.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    loaded = model.load_state_dict(checkpoint["model"], strict=False)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("LAPA checkpoint mismatch")
    model.eval()
    center_tensor = torch.from_numpy(center).to(device)
    half_tensor = torch.from_numpy(half).to(device)
    K_device = [torch.from_numpy(value).float().to(device) for value in K]
    w2c_device = [
        build_w2c_normalized(torch.from_numpy(value).float().to(device), center_tensor, half_tensor)
        for value in w2c
    ]
    points_device = [
        [torch.from_numpy(tracks[view, frame]).float().to(device) for frame in range(frame_count)]
        for view in range(views)
    ]
    features_device = [
        [torch.from_numpy(features[view][frame]).float().to(device) for frame in range(frame_count)]
        for view in range(views)
    ]
    valid_device = [torch.from_numpy(visibility[view]).to(device) for view in range(views)]
    with torch.no_grad():
        output = model(
            points_device,
            features_device,
            K_device,
            w2c_device,
            torch.from_numpy(queries_norm).float().to(device),
            (512, 384),
            view_valid=valid_device,
        )
    pred_world = output["points_3d"].float().cpu().numpy() * half + center
    pred_uv, depth = project_world(pred_world, manifest["K"][0], manifest["w2c"][0])
    visible = output["vis_logits"].float().cpu().numpy() > 0
    native_height, native_width = int(manifest["heights"][0]), int(manifest["widths"][0])
    visible &= (
        (depth > 1e-5)
        & (pred_uv[..., 0] >= 0)
        & (pred_uv[..., 0] < native_width)
        & (pred_uv[..., 1] >= 0)
        & (pred_uv[..., 1] < native_height)
    )
    input_ref = tracks[0].copy()
    input_ref[..., 0] *= native_width / 512
    input_ref[..., 1] *= native_height / 384
    return {
        "pred_uv": pred_uv,
        "pred_visible": visible,
        "pred_world": pred_world,
        "input_cotracker_uv_ref": input_ref,
        "input_cotracker_visible_ref": visibility[0],
        "aabb_center": center,
        "aabb_half": half,
    }


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("These pretrained qualitative runners require CUDA")
    if args.model == "opend4rt":
        result = run_opend4rt(manifest, device, args.query_chunk_size)
    elif args.model == "spatracker":
        result = run_spatracker(manifest, device, args.support_points)
    elif args.model == "mvtap":
        result = run_mvtap(manifest, device)
    else:
        result = run_lapa(manifest, device)

    pred_uv = np.asarray(result["pred_uv"], dtype=np.float32)
    pred_visible = np.asarray(result["pred_visible"], dtype=bool)
    expected = (len(manifest["frame_numbers"]), len(manifest["query_world"]))
    if pred_uv.shape != (*expected, 2) or pred_visible.shape != expected:
        raise RuntimeError(
            f"Unexpected output shapes: uv={pred_uv.shape}, visible={pred_visible.shape}, expected={expected}"
        )
    finite = np.isfinite(pred_uv).all(axis=-1)
    anchor_error = np.linalg.norm(pred_uv[0] - manifest["query_uv_ref"], axis=-1)
    metadata = {
        "model": args.model,
        "scene": str(manifest["scene"]),
        "frames": expected[0],
        "points": expected[1],
        "finite_fraction": float(np.mean(finite)),
        "visible_fraction": float(np.mean(pred_visible & finite)),
        "query_anchor_median_px": float(np.median(anchor_error)),
        "query_anchor_max_px": float(np.max(anchor_error)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **result, query_uv_ref=manifest["query_uv_ref"])
    temporary.replace(args.output)
    write_json(args.output.with_suffix(".json"), metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

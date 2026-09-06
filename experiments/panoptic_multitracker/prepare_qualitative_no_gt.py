#!/usr/bin/env python3
"""Prepare shared queries and camera geometry for an unannotated sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from qualitative_no_gt_common import (
    load_video,
    project_world,
    read_colmap_text,
    sample_frame_numbers,
    unproject_reference,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--cameras", nargs=3, default=["0.png", "1.png", "2.png"])
    parser.add_argument("--max-depth", type=float, default=15.0)
    return parser.parse_args()


def motion_mask(video: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = video.shape[1:3]
    small_width = 480
    small_height = round(height * small_width / width)
    small = np.stack(
        [cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_AREA) for frame in video]
    )
    median = np.median(small, axis=0).astype(np.uint8)
    difference = np.mean(cv2.absdiff(small[0], median), axis=2)
    threshold = max(12.0, float(np.percentile(difference, 75)))
    mask = (difference >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_border = (
            x <= 1
            or y <= 1
            or x + component_width >= small_width - 1
            or y + component_height >= small_height - 1
        )
        if area >= 20 and not touches_border:
            cleaned[labels == label] = 255
    full_mask = cv2.resize(cleaned, (width, height), interpolation=cv2.INTER_NEAREST)
    full_difference = cv2.resize(
        difference.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    return full_mask, full_difference, threshold


def verify_fixed_calibration(
    scene_root: Path, frame_numbers: np.ndarray, camera_names: list[str], baseline: dict
) -> None:
    for frame in frame_numbers:
        cameras = read_colmap_text(scene_root / f"frame{int(frame)}" / "sparse/0")
        for name in camera_names:
            if name not in cameras:
                raise RuntimeError(f"Camera {name} absent from frame {frame}")
            for key in ("K", "w2c"):
                if not np.allclose(cameras[name][key], baseline[name][key], atol=1e-5):
                    raise RuntimeError(f"Camera {name} {key} changes at frame {frame}")


def select_queries(
    video: np.ndarray,
    inverse_depth_path: Path,
    cameras: dict,
    camera_names: list[str],
    point_count: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    mask, difference, threshold = motion_mask(video)
    inverse_depth = np.asarray(Image.open(inverse_depth_path), dtype=np.float32)
    if inverse_depth.shape != mask.shape:
        raise RuntimeError(
            f"Inverse-depth shape {inverse_depth.shape} does not match image {mask.shape}"
        )
    depth = np.divide(
        1.0,
        inverse_depth,
        out=np.full_like(inverse_depth, np.nan),
        where=inverse_depth > 1e-8,
    )
    mask[(depth < 2.0) | (depth > max_depth) | ~np.isfinite(depth)] = 0
    gray = cv2.cvtColor(video[0], cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=1000,
        qualityLevel=0.003,
        minDistance=10,
        mask=mask,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None:
        raise RuntimeError("No Shi-Tomasi corners found in the motion mask")
    candidates = corners[:, 0].astype(np.float32)
    rounded = np.rint(candidates).astype(np.int64)
    candidate_depth = depth[rounded[:, 1], rounded[:, 0]]
    reference = cameras[camera_names[0]]
    world = unproject_reference(candidates, candidate_depth, reference["K"], reference["w2c"])

    valid = np.isfinite(world).all(axis=-1)
    projected = []
    for name in camera_names:
        camera = cameras[name]
        uv, z = project_world(world, camera["K"], camera["w2c"])
        in_bounds = (
            (z > 1e-5)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < camera["width"])
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < camera["height"])
        )
        valid &= in_bounds
        projected.append(uv)
    candidates, world = candidates[valid], world[valid]
    projected = np.stack([uv[valid] for uv in projected])
    if len(candidates) < point_count:
        raise RuntimeError(f"Only {len(candidates)} cross-view-valid corner candidates")

    rounded = np.rint(candidates).astype(np.int64)
    scores = difference[rounded[:, 1], rounded[:, 0]]
    order = np.argsort(-scores, kind="stable")
    selected = []
    for minimum_distance in (35.0, 25.0, 15.0, 8.0):
        for index in order:
            if index in selected:
                continue
            if all(np.linalg.norm(candidates[index] - candidates[other]) >= minimum_distance for other in selected):
                selected.append(int(index))
            if len(selected) >= point_count:
                break
        if len(selected) >= point_count:
            break
    if len(selected) < point_count:
        raise RuntimeError(f"Could spatially distribute only {len(selected)} points")
    selected = np.asarray(selected[:point_count], dtype=np.int64)
    diagnostics = {
        "motion_threshold": threshold,
        "motion_mask_fraction": float(np.mean(mask > 0)),
        "corner_candidates": int(len(candidates)),
    }
    return candidates[selected], world[selected], mask, diagnostics


def render_initialization(
    scene_root: Path,
    output_dir: Path,
    camera_names: list[str],
    cameras: dict,
    query_world: np.ndarray,
) -> None:
    colors = [
        tuple(round(value * 255) for value in cv2.cvtColor(
            np.uint8([[[round(index * 179 / max(len(query_world), 1)), 220, 255]]]),
            cv2.COLOR_HSV2RGB,
        )[0, 0] / 255.0)
        for index in range(len(query_world))
    ]
    panels = []
    for name in camera_names:
        image = Image.open(scene_root / "frame1/images" / name).convert("RGB")
        uv, _ = project_world(query_world, cameras[name]["K"], cameras[name]["w2c"])
        draw = ImageDraw.Draw(image)
        for index, (xy, color) in enumerate(zip(uv, colors)):
            x, y = map(float, xy)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="black", width=2)
            draw.text((x + 8, y - 7), str(index), fill="cyan", stroke_width=1, stroke_fill="black")
        image.thumbnail((640, 360))
        panel = Image.new("RGB", (640, 385), "black")
        panel.paste(image, (0, 25))
        ImageDraw.Draw(panel).text((8, 7), f"frame 1 | camera {name}", fill="white")
        panels.append(panel)
    sheet = Image.new("RGB", (640 * len(panels), 385), "black")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * 640, 0))
    sheet.save(output_dir / "query_initialization.png")


def main() -> None:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_numbers = sample_frame_numbers(scene_root, args.frames)
    cameras = read_colmap_text(scene_root / "frame1/sparse/0")
    for name in args.cameras:
        if name not in cameras:
            raise RuntimeError(f"Camera {name} is unavailable")
    verify_fixed_calibration(scene_root, frame_numbers, args.cameras, cameras)
    reference_video = load_video(scene_root, frame_numbers, args.cameras[0])
    query_uv, query_world, mask, diagnostics = select_queries(
        reference_video,
        scene_root / "frame1/depths" / f"{Path(args.cameras[0]).stem}.tiff",
        cameras,
        args.cameras,
        args.points,
        args.max_depth,
    )
    selected_cameras = [cameras[name] for name in args.cameras]
    manifest_path = output_dir / "manifest.npz"
    temporary = manifest_path.with_suffix(".partial.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            scene=np.asarray(scene_root.name),
            scene_root=np.asarray(str(scene_root)),
            frame_numbers=frame_numbers,
            camera_names=np.asarray(args.cameras),
            K=np.stack([camera["K"] for camera in selected_cameras]),
            w2c=np.stack([camera["w2c"] for camera in selected_cameras]),
            widths=np.asarray([camera["width"] for camera in selected_cameras]),
            heights=np.asarray([camera["height"] for camera in selected_cameras]),
            query_uv_ref=query_uv,
            query_world=query_world,
        )
    temporary.replace(manifest_path)
    Image.fromarray(mask).save(output_dir / "motion_query_mask.png")
    render_initialization(scene_root, output_dir, args.cameras, cameras, query_world)
    write_json(
        output_dir / "manifest.json",
        {
            "scene": scene_root.name,
            "scene_root": str(scene_root),
            "frame_numbers": frame_numbers.tolist(),
            "camera_names": args.cameras,
            "frames": len(frame_numbers),
            "points": len(query_uv),
            "query_rule": (
                "Shi-Tomasi corners in non-border frame-1 versus temporal-median motion "
                "components, gated by 2-15m inverse depth and three-view projection"
            ),
            **diagnostics,
        },
    )
    print(f"Prepared {scene_root.name}: {len(frame_numbers)} frames, {len(query_uv)} points")


if __name__ == "__main__":
    main()

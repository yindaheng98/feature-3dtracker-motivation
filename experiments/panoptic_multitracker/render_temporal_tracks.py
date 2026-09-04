#!/usr/bin/env python3
"""Render temporal-stride 3D tracking predictions in the reference camera."""

from __future__ import annotations

import argparse
import colorsys
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MODEL_ORDER = ("opend4rt", "spatracker", "mvtap", "lapa")
MODEL_LABELS = {
    "opend4rt": "Open-d4rt",
    "spatracker": "SpaTrackerV2",
    "mvtap": "MV-TAP",
    "lapa": "LAPA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/panoptic_multitracker/temporal_stride"),
    )
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--trail-length", type=int, default=8)
    parser.add_argument("--gif-duration-ms", type=int, default=350)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def project(xyz: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    z = xyz[..., 2]
    valid = np.isfinite(xyz).all(axis=-1) & (z > 1e-6)
    safe_z = np.where(valid, z, 1.0)
    uv = np.stack(
        [fx * xyz[..., 0] / safe_z + cx, fy * xyz[..., 1] / safe_z + cy],
        axis=-1,
    )
    uv[~valid] = np.nan
    return uv, valid


def world_to_camera(world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    w2c = np.asarray(w2c, dtype=np.float64)
    return np.asarray(world, dtype=np.float64) @ w2c[:3, :3].T + w2c[:3, 3]


def load_source(path: Path) -> tuple[list[Image.Image], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        images = [
            Image.open(io.BytesIO(bytes(blob))).convert("RGB")
            for blob in data["images_jpeg_bytes"]
        ]
        gt_uv, depth_valid = project(data["tracks_XYZ"], data["fx_fy_cx_cy"])
        gt_visible = np.asarray(data["visibility"], dtype=bool) & depth_valid
        intrinsics = np.asarray(data["fx_fy_cx_cy"], dtype=np.float64)
    return images, gt_uv, gt_visible, intrinsics


def load_prediction(path: Path, fallback_intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        intrinsics = data["intrinsics"] if "intrinsics" in data else fallback_intrinsics
        if "pred_xyz" in data:
            pred_camera = data["pred_xyz"]
        elif "pred_world" in data and "w2c_ref" in data:
            pred_camera = world_to_camera(data["pred_world"], data["w2c_ref"])
        else:
            raise KeyError(f"Cannot find a renderable 3D prediction in {path}")

        pred_uv, depth_valid = project(pred_camera, intrinsics)
        predicted_visible = np.asarray(data["pred_visible"], dtype=bool)
        if predicted_visible.ndim == 3:
            predicted_visible = predicted_visible[0]
        predicted_visible &= depth_valid
    return pred_uv, predicted_visible


def load_metrics(path: Path) -> dict[tuple[int, str], dict[str, float]]:
    metrics: dict[tuple[int, str], dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            metrics[(int(row["stride"]), row["model"])] = {
                "aj3d": float(row["aj3d"]),
                "epe_m": float(row["epe_m"]),
            }
    return metrics


def choose_points(
    uv: np.ndarray,
    visible: np.ndarray,
    width: int,
    height: int,
    count: int,
) -> np.ndarray:
    in_bounds = (
        (uv[..., 0] >= 0)
        & (uv[..., 0] < width)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < height)
    )
    valid = visible & in_bounds & np.isfinite(uv).all(axis=-1)
    scores = np.zeros(uv.shape[1], dtype=np.float64)
    for frame in range(1, uv.shape[0]):
        pairs = valid[frame - 1] & valid[frame]
        scores[pairs] += np.linalg.norm(uv[frame, pairs] - uv[frame - 1, pairs], axis=-1)
    scores[valid.sum(axis=0) < max(4, uv.shape[0] // 2)] = -1.0
    order = np.argsort(-scores, kind="stable")
    selected = order[scores[order] >= 0][:count]
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} sufficiently visible points available")
    return selected


def make_colors(count: int) -> list[tuple[int, int, int]]:
    colors = []
    for index in range(count):
        hue = (index * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 1.0)
        colors.append((round(red * 255), round(green * 255), round(blue * 255)))
    return colors


def draw_segmented_trail(
    draw: ImageDraw.ImageDraw,
    uv: np.ndarray,
    valid: np.ndarray,
    start: int,
    end: int,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    for frame in range(max(start + 1, 1), end + 1):
        if valid[frame - 1] and valid[frame]:
            draw.line(
                [tuple(uv[frame - 1]), tuple(uv[frame])],
                fill=fill,
                width=width,
            )


def marker_cross(draw: ImageDraw.ImageDraw, xy: np.ndarray, radius: int = 5) -> None:
    x, y = map(float, xy)
    draw.line((x - radius, y - radius, x + radius, y + radius), fill="black", width=4)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill="black", width=4)
    draw.line((x - radius, y - radius, x + radius, y + radius), fill="white", width=2)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill="white", width=2)


def render_panel(
    image: Image.Image,
    pred_uv: np.ndarray,
    pred_visible: np.ndarray,
    gt_uv: np.ndarray,
    gt_visible: np.ndarray,
    selected: np.ndarray,
    colors: list[tuple[int, int, int]],
    frame: int,
    trail_length: int,
    title: str,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    start = max(0, frame - trail_length + 1)
    for color, point in zip(colors, selected):
        draw_segmented_trail(
            draw, gt_uv[:, point], gt_visible[:, point], start, frame, (0, 0, 0), 4
        )
        draw_segmented_trail(
            draw, gt_uv[:, point], gt_visible[:, point], start, frame, (255, 255, 255), 2
        )
        draw_segmented_trail(
            draw, pred_uv[:, point], pred_visible[:, point], start, frame, (0, 0, 0), 5
        )
        draw_segmented_trail(
            draw, pred_uv[:, point], pred_visible[:, point], start, frame, color, 3
        )
        if gt_visible[frame, point]:
            marker_cross(draw, gt_uv[frame, point])
        if pred_visible[frame, point]:
            x, y = map(float, pred_uv[frame, point])
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="black")
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    font = ImageFont.load_default()
    draw.rectangle((0, 0, canvas.width, 23), fill=(0, 0, 0))
    draw.text((7, 6), title, fill=(255, 255, 255), font=font)
    return canvas


def compose_grid(panels: list[Image.Image], columns: int, panel_size: tuple[int, int]) -> Image.Image:
    rows = (len(panels) + columns - 1) // columns
    grid = Image.new("RGB", (panel_size[0] * columns, panel_size[1] * rows), "black")
    for index, panel in enumerate(panels):
        panel = panel.resize(panel_size, Image.Resampling.LANCZOS)
        grid.paste(panel, ((index % columns) * panel_size[0], (index // columns) * panel_size[1]))
    return grid


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.root / "renderings"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.root / "manifest.json").read_text())
    metrics = load_metrics(args.root / "common_metrics.csv")
    unknown = set(args.strides) - {int(value) for value in manifest["strides"]}
    if unknown:
        raise ValueError(f"Strides absent from manifest: {sorted(unknown)}")

    runs: dict[int, dict[str, object]] = {}
    for stride in args.strides:
        run_root = args.root / f"stride_{stride}"
        source_path = (
            run_root / "source/tapvid3d_minival/tap3d_juggle/juggle_7.npz"
        )
        images, gt_uv, gt_visible, intrinsics = load_source(source_path)
        predictions = {}
        for model in MODEL_ORDER:
            predictions[model] = load_prediction(
                run_root / "results" / model / "prediction.npz", intrinsics
            )
        runs[stride] = {
            "images": images,
            "gt_uv": gt_uv,
            "gt_visible": gt_visible,
            "predictions": predictions,
            "raw_frames": manifest["runs"][str(stride)]["raw_frame_indices"],
        }

    base = runs[args.strides[0]]
    width, height = base["images"][0].size
    selected = choose_points(
        base["gt_uv"], base["gt_visible"], width, height, args.points
    )
    colors = make_colors(len(selected))
    point_ids = [manifest["point_indices"][int(index)] for index in selected]
    (output_dir / "render_manifest.json").write_text(
        json.dumps(
            {
                "scene": manifest["scene"],
                "reference_camera": manifest["reference_camera"],
                "strides": args.strides,
                "compact_point_slots": selected.tolist(),
                "original_point_indices": point_ids,
                "point_selection": (
                    "highest accumulated GT pixel motion among points visible in at "
                    "least half of the first rendered stride's frames"
                ),
                "trail_length_in_sampled_frames": args.trail_length,
                "legend": "colored circle/trail = projected 3D prediction; white cross/trail = GT",
            },
            indent=2,
        )
        + "\n"
    )

    for stride in args.strides:
        run = runs[stride]
        animation = []
        for frame, image in enumerate(run["images"]):
            panels = []
            for model in MODEL_ORDER:
                pred_uv, pred_visible = run["predictions"][model]
                metric = metrics[(stride, model)]
                title = (
                    f"{MODEL_LABELS[model]} | stride {stride} | raw frame "
                    f"{run['raw_frames'][frame]} | AJ3D {metric['aj3d']:.3f} | "
                    f"EPE {metric['epe_m']:.3f} m"
                )
                panels.append(
                    render_panel(
                        image,
                        pred_uv,
                        pred_visible,
                        run["gt_uv"],
                        run["gt_visible"],
                        selected,
                        colors,
                        frame,
                        args.trail_length,
                        title,
                    )
                )
            animation.append(compose_grid(panels, columns=2, panel_size=(480, 270)))
        animation[0].save(
            output_dir / f"tracks_stride_{stride}.gif",
            save_all=True,
            append_images=animation[1:],
            duration=args.gif_duration_ms,
            loop=0,
            optimize=True,
        )

        contact_panels = []
        for model in MODEL_ORDER:
            pred_uv, pred_visible = run["predictions"][model]
            metric = metrics[(stride, model)]
            for frame in [0, 5, 10, 15]:
                title = (
                    f"{MODEL_LABELS[model]} | s={stride}, raw={run['raw_frames'][frame]} | "
                    f"AJ3D {metric['aj3d']:.3f}, EPE {metric['epe_m']:.3f}m"
                )
                contact_panels.append(
                    render_panel(
                        run["images"][frame],
                        pred_uv,
                        pred_visible,
                        run["gt_uv"],
                        run["gt_visible"],
                        selected,
                        colors,
                        frame,
                        args.trail_length,
                        title,
                    )
                )
        compose_grid(contact_panels, columns=4, panel_size=(480, 270)).save(
            output_dir / f"tracks_stride_{stride}_contact_sheet.png"
        )

    overview_panels = []
    for model in MODEL_ORDER:
        for stride in args.strides:
            run = runs[stride]
            frame = len(run["images"]) - 1
            pred_uv, pred_visible = run["predictions"][model]
            metric = metrics[(stride, model)]
            title = (
                f"{MODEL_LABELS[model]} | s={stride}, raw={run['raw_frames'][frame]} | "
                f"AJ3D {metric['aj3d']:.3f}, EPE {metric['epe_m']:.3f}m"
            )
            overview_panels.append(
                render_panel(
                    run["images"][frame],
                    pred_uv,
                    pred_visible,
                    run["gt_uv"],
                    run["gt_visible"],
                    selected,
                    colors,
                    frame,
                    args.trail_length,
                    title,
                )
            )
    compose_grid(overview_panels, columns=len(args.strides), panel_size=(480, 270)).save(
        output_dir / "final_frame_comparison.png"
    )
    print(f"Rendered {len(args.strides)} strides to {output_dir}")


if __name__ == "__main__":
    main()

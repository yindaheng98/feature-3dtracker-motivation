#!/usr/bin/env python3
"""Render four-model trajectory grids for an unannotated sequence."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from qualitative_no_gt_common import load_manifest


MODELS = ("opend4rt", "spatracker", "mvtap", "lapa")
LABELS = {
    "opend4rt": "Open-d4rt",
    "spatracker": "SpaTrackerV2",
    "mvtap": "MV-TAP",
    "lapa": "LAPA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--trail-length", type=int, default=8)
    parser.add_argument("--gif-duration-ms", type=int, default=350)
    return parser.parse_args()


def colors(count: int) -> list[tuple[int, int, int]]:
    result = []
    for index in range(count):
        rgb = colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.85, 1.0)
        result.append(tuple(round(channel * 255) for channel in rgb))
    return result


def draw_trail(
    draw: ImageDraw.ImageDraw,
    uv: np.ndarray,
    valid: np.ndarray,
    start: int,
    end: int,
    color: tuple[int, int, int],
) -> None:
    for frame in range(max(start + 1, 1), end + 1):
        if valid[frame - 1] and valid[frame]:
            points = [tuple(uv[frame - 1]), tuple(uv[frame])]
            draw.line(points, fill="black", width=5)
            draw.line(points, fill=color, width=3)


def render_panel(
    image: Image.Image,
    uv: np.ndarray,
    visible: np.ndarray,
    palette: list[tuple[int, int, int]],
    frame: int,
    trail_length: int,
    title: str,
) -> Image.Image:
    canvas = image.copy()
    width, height = canvas.size
    in_bounds = (
        np.isfinite(uv).all(axis=-1)
        & (uv[..., 0] >= 0)
        & (uv[..., 0] < width)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < height)
    )
    valid = visible & in_bounds
    draw = ImageDraw.Draw(canvas)
    start = max(0, frame - trail_length + 1)
    for point, color in enumerate(palette):
        draw_trail(draw, uv[:, point], valid[:, point], start, frame, color)
        if valid[frame, point]:
            x, y = map(float, uv[frame, point])
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="black")
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    draw.rectangle((0, 0, width, 25), fill="black")
    draw.text((7, 7), title, fill="white", font=ImageFont.load_default())
    return canvas


def grid(panels: list[Image.Image], columns: int, panel_size: tuple[int, int]) -> Image.Image:
    rows = (len(panels) + columns - 1) // columns
    output = Image.new("RGB", (columns * panel_size[0], rows * panel_size[1]), "black")
    for index, panel in enumerate(panels):
        panel = panel.resize(panel_size, Image.Resampling.LANCZOS)
        output.paste(panel, ((index % columns) * panel_size[0], (index // columns) * panel_size[1]))
    return output


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    manifest = load_manifest(scene_dir / "manifest.npz")
    scene_root = Path(str(manifest["scene_root"]))
    frame_numbers = manifest["frame_numbers"].astype(int)
    camera = str(manifest["camera_names"][0])
    frames = [
        Image.open(scene_root / f"frame{frame}" / "images" / camera).convert("RGB")
        for frame in frame_numbers
    ]
    predictions = {}
    metadata = {}
    for model in MODELS:
        with np.load(scene_dir / "predictions" / f"{model}.npz", allow_pickle=False) as data:
            predictions[model] = (
                np.asarray(data["pred_uv"], np.float32),
                np.asarray(data["pred_visible"], bool),
            )
        metadata[model] = json.loads(
            (scene_dir / "predictions" / f"{model}.json").read_text()
        )
    palette = colors(len(manifest["query_uv_ref"]))

    animation = []
    for frame, image in enumerate(frames):
        panels = []
        for model in MODELS:
            uv, visible = predictions[model]
            title = (
                f"{LABELS[model]} | {str(manifest['scene'])} | source frame "
                f"{frame_numbers[frame]} | predicted tracks, no GT"
            )
            panels.append(
                render_panel(image, uv, visible, palette, frame, args.trail_length, title)
            )
        animation.append(grid(panels, 2, (480, 270)))
    animation[0].save(
        scene_dir / "tracks_four_model.gif",
        save_all=True,
        append_images=animation[1:],
        duration=args.gif_duration_ms,
        loop=0,
        optimize=True,
    )

    selected_frames = np.linspace(0, len(frames) - 1, 4).round().astype(int).tolist()
    contact = []
    for model in MODELS:
        uv, visible = predictions[model]
        for frame in selected_frames:
            title = (
                f"{LABELS[model]} | source {frame_numbers[frame]} | "
                f"visible {metadata[model]['visible_fraction']:.1%} | "
                f"anchor {metadata[model]['query_anchor_median_px']:.1f}px | no GT"
            )
            contact.append(
                render_panel(frames[frame], uv, visible, palette, frame, args.trail_length, title)
            )
    grid(contact, 4, (480, 270)).save(scene_dir / "tracks_contact_sheet.png")

    final_panels = []
    frame = len(frames) - 1
    for model in MODELS:
        uv, visible = predictions[model]
        title = (
            f"{LABELS[model]} | final source frame {frame_numbers[frame]} | "
            f"visible {metadata[model]['visible_fraction']:.1%} | "
            f"anchor {metadata[model]['query_anchor_median_px']:.1f}px | no GT"
        )
        final_panels.append(
            render_panel(frames[frame], uv, visible, palette, frame, args.trail_length, title)
        )
    grid(final_panels, 2, (640, 360)).save(scene_dir / "tracks_final_frame.png")
    print(f"Rendered {str(manifest['scene'])} to {scene_dir}")


if __name__ == "__main__":
    main()

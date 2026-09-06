"""Shared data and geometry helpers for unannotated qualitative sequences."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


FRAME_PATTERN = re.compile(r"frame(\d+)$")


def quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(qvec, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_colmap_text(model_dir: Path) -> dict[str, dict[str, np.ndarray | int]]:
    cameras = {}
    for line in (model_dir / "cameras.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[1] != "PINHOLE":
            raise ValueError(f"Expected PINHOLE camera, got {fields[1]}")
        camera_id = int(fields[0])
        width, height = int(fields[2]), int(fields[3])
        fx, fy, cx, cy = map(float, fields[4:8])
        cameras[camera_id] = {
            "width": width,
            "height": height,
            "K": np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64),
        }

    result = {}
    for line in (model_dir / "images.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            continue
        qvec = np.asarray(fields[1:5], dtype=np.float64)
        translation = np.asarray(fields[5:8], dtype=np.float64)
        camera_id = int(fields[8])
        name = fields[9]
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = quaternion_to_rotation(qvec)
        w2c[:3, 3] = translation
        result[name] = {**cameras[camera_id], "w2c": w2c}
    if not result:
        raise RuntimeError(f"No cameras parsed from {model_dir}")
    return result


def available_frame_numbers(scene_root: Path) -> list[int]:
    numbers = []
    for path in scene_root.iterdir():
        match = FRAME_PATTERN.fullmatch(path.name) if path.is_dir() else None
        if match:
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def sample_frame_numbers(scene_root: Path, count: int) -> np.ndarray:
    available = available_frame_numbers(scene_root)
    if len(available) < count:
        raise ValueError(f"{scene_root} has {len(available)} frames, requested {count}")
    indices = np.rint(np.linspace(0, len(available) - 1, count)).astype(np.int64)
    selected = np.asarray([available[index] for index in indices], dtype=np.int64)
    if len(np.unique(selected)) != count:
        raise RuntimeError("Frame sampling produced duplicates")
    return selected


def load_video(scene_root: Path, frame_numbers: np.ndarray, camera_name: str) -> np.ndarray:
    frames = []
    for frame in frame_numbers:
        path = scene_root / f"frame{int(frame)}" / "images" / camera_name
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read {path}")
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise RuntimeError(f"Camera {camera_name} has inconsistent shapes: {shapes}")
    return np.stack(frames)


def project_world(world: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = np.asarray(world, dtype=np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
    depth = camera[..., 2]
    safe_depth = np.where(np.abs(depth) > 1e-8, depth, 1.0)
    uv = np.stack(
        [
            K[0, 0] * camera[..., 0] / safe_depth + K[0, 2],
            K[1, 1] * camera[..., 1] / safe_depth + K[1, 2],
        ],
        axis=-1,
    )
    return uv.astype(np.float32), depth


def unproject_reference(uv: np.ndarray, depth: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    camera = np.stack(
        [
            (uv[:, 0] - K[0, 2]) / K[0, 0] * depth,
            (uv[:, 1] - K[1, 2]) / K[1, 1] * depth,
            depth,
        ],
        axis=-1,
    )
    return ((camera - w2c[:3, 3]) @ w2c[:3, :3]).astype(np.float32)


def resize_video_and_camera(
    video: np.ndarray, K: np.ndarray, size: tuple[int, int] = (512, 384)
) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    native_height, native_width = video.shape[1:3]
    resized = np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in video]
    )
    scaled = np.asarray(K, dtype=np.float32).copy()
    scaled[0] *= width / native_width
    scaled[1] *= height / native_height
    return resized, scaled


def camera_payload(manifest: np.lib.npyio.NpzFile, view: int) -> tuple[np.ndarray, np.ndarray, str]:
    return manifest["K"][view], manifest["w2c"][view], str(manifest["camera_names"][view])


def load_manifest(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def prediction_path(root: Path, scene: str, model: str) -> Path:
    return root / scene / "predictions" / f"{model}.npz"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")

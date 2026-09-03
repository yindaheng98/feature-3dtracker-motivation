#!/usr/bin/env python3
"""Materialize compact, manifest-locked PStudio temporal-stride inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def project(world: np.ndarray, k: np.ndarray, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = world @ w2c[:3, :3].T + w2c[:3, 3]
    uvw = camera @ k.T
    uv = uvw[..., :2] / np.maximum(uvw[..., 2:], 1e-8)
    return uv, camera[..., 2]


def slice_npz(source: Path, destination: Path, frame_idx: np.ndarray, point_idx=None) -> None:
    with np.load(source, allow_pickle=True) as pack:
        payload = {key: pack[key] for key in pack.files}
    for key in ("images_jpeg_bytes", "tracks_XYZ", "visibility", "extrinsics_w2c"):
        if key in payload and payload[key].shape[0] >= int(frame_idx[-1]) + 1:
            payload[key] = payload[key][frame_idx]
    if point_idx is not None:
        if "tracks_XYZ" in payload:
            payload["tracks_XYZ"] = payload["tracks_XYZ"][:, point_idx]
        if "visibility" in payload:
            payload["visibility"] = payload["visibility"][:, point_idx]
        if "queries_xyt" in payload:
            payload["queries_xyt"] = payload["queries_xyt"][point_idx]
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-root", type=Path, required=True)
    parser.add_argument("--mc-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", default="juggle")
    parser.add_argument("--cameras", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--strides", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=64)
    args = parser.parse_args()

    meta_path = args.mc_dir / f"{args.scene}_mc.json"
    meta = json.loads(meta_path.read_text())
    camera_lookup = {int(camera["cam_id"]): camera for camera in meta["cameras"]}
    cameras = [camera_lookup[cam_id] for cam_id in args.cameras]
    ref_id = args.cameras[0]
    ref_track_path = args.mc_dir / args.scene / "tracks_world" / f"cam_{ref_id}.npz"
    with np.load(ref_track_path) as pack:
        ref_payload = {key: pack[key] for key in pack.files}
    world = np.asarray(ref_payload["tracks_world"], dtype=np.float32)
    visible = np.asarray(ref_payload["visibility"], dtype=bool)

    eligible = visible[args.start].copy()
    for camera in cameras:
        uv, depth = project(
            world[args.start],
            np.asarray(camera["K"], dtype=np.float64),
            np.asarray(camera["w2c"], dtype=np.float64),
        )
        eligible &= (
            (depth > 1e-4)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < int(camera["width"]))
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < int(camera["height"]))
        )
    point_idx = np.flatnonzero(eligible)[: args.max_points]
    if not len(point_idx):
        raise RuntimeError("No reference points are valid in every selected view")

    manifest = {
        "scene": args.scene,
        "reference_camera": ref_id,
        "companion_cameras": args.cameras[1:],
        "model_frames": args.frames,
        "start_frame": args.start,
        "strides": args.strides,
        "point_indices": point_idx.tolist(),
        "selection_rule": "first original indices visible in reference and in-bounds with positive depth in all selected views at query frame",
        "runs": {},
    }

    for stride in args.strides:
        frame_idx = args.start + stride * np.arange(args.frames, dtype=np.int64)
        if int(frame_idx[-1]) >= world.shape[0]:
            raise ValueError(f"stride {stride} exceeds {world.shape[0]} source frames")
        run_root = (args.output_root / f"stride_{stride}").resolve()
        source_root = run_root / "source"
        derived_npz_root = source_root / "tapvid3d_minival"

        for cam_id in args.cameras:
            source = args.npz_root / f"tap3d_{args.scene}" / f"{args.scene}_{cam_id}.npz"
            destination = derived_npz_root / f"tap3d_{args.scene}" / source.name
            slice_npz(source, destination, frame_idx, point_idx if cam_id == ref_id else None)
        (source_root / "minival_pstudio.txt").write_text(f"{args.scene}_{ref_id}.npz\n")

        derived_mc = run_root / "mc"
        selected_cameras = []
        for camera in cameras:
            item = dict(camera)
            item["T"] = args.frames
            item["npz_path"] = str(
                derived_npz_root / f"tap3d_{args.scene}" / f"{args.scene}_{item['cam_id']}.npz"
            )
            selected_cameras.append(item)
        derived_meta = dict(meta)
        derived_meta["cameras"] = selected_cameras
        derived_mc.mkdir(parents=True, exist_ok=True)
        derived_meta_path = derived_mc / f"{args.scene}_mc.json"
        derived_meta_path.write_text(json.dumps(derived_meta, indent=2) + "\n")
        index = {
            "scenes": {
                args.scene: {
                    "meta_path": str(derived_meta_path),
                    "n_cameras": len(selected_cameras),
                    "calib_report": meta["calib_report"],
                }
            },
            "calib_ok": True,
        }
        (derived_mc / "index.json").write_text(json.dumps(index, indent=2) + "\n")

        compact_tracks = {}
        for key, value in ref_payload.items():
            if value.ndim >= 2 and value.shape[0] == world.shape[0]:
                compact_tracks[key] = value[frame_idx][:, point_idx]
            elif value.ndim >= 1 and value.shape[0] == world.shape[1]:
                compact_tracks[key] = value[point_idx]
            else:
                compact_tracks[key] = value
        track_out = derived_mc / args.scene / "tracks_world" / f"cam_{ref_id}.npz"
        track_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(track_out, **compact_tracks)

        open_input = run_root / "opend4rt_input" / "pstudio_mini"
        open_input.mkdir(parents=True, exist_ok=True)
        link = open_input / f"{args.scene}_{ref_id}.npz"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(
            derived_npz_root / f"tap3d_{args.scene}" / f"{args.scene}_{ref_id}.npz"
        )
        manifest["runs"][str(stride)] = {
            "raw_frame_indices": frame_idx.tolist(),
            "root": str(run_root.relative_to(Path.cwd().resolve())),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}: {len(point_idx)} points, {len(args.strides)} strides")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare the shared TAPVid-3D PStudio minival and D3G camera data."""

from __future__ import annotations

import argparse
import re
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


MINIVAL_ROOT = (
    "https://storage.googleapis.com/dm-tapnet/tapvid3d/"
    "release_files/minival_v1.0/"
)
SPLITS_URL = (
    "https://raw.githubusercontent.com/google-deepmind/tapnet/main/"
    "tapnet/tapvid3d/splits/tapvid3d_splits.py"
)
D3G_URL = "https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip"
SCENES = {"basketball", "boxes", "football", "juggle", "softball", "tennis"}


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.replace(destination)


def minival_files() -> list[str]:
    with urllib.request.urlopen(SPLITS_URL, timeout=60) as response:
        text = response.read().decode("utf-8")
    start = text.index("MINIVAL_FILES = {")
    end = text.index("\n}", start)
    names = re.findall(r'"([^"/]+\.npz)"', text[start:end])
    return sorted({name for name in names if name.split("_", 1)[0] in SCENES})


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError(f"unsafe archive member: {member.filename}")
        zf.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/panoptic_tracking"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    names = minival_files()
    minival = args.root / "tapvid3d_minival"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(
            pool.map(
                lambda name: download(
                    MINIVAL_ROOT + name,
                    minival / f"tap3d_{name.split('_', 1)[0]}" / name,
                ),
                names,
            )
        )
    (args.root / "minival_pstudio.txt").write_text("\n".join(names) + "\n")

    archive = args.root / "dynamic3dgaussians_data.zip"
    download(D3G_URL, archive)
    extracted = args.root / "d3g"
    if not (extracted / "data").is_dir():
        safe_extract(archive, extracted)

    print(f"PStudio minival: {len(names)} files in {minival}")
    print(f"D3G camera data: {extracted / 'data'}")


if __name__ == "__main__":
    main()

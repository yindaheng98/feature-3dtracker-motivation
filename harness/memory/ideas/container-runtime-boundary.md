---
schema: harness.idea/v1
id: IDEA-CONTAINER-001
status: proposed
scope: Docker runtime and dependency ownership
---

# Separate image dependencies from mounted project state

## Evidence

- The current image starts from Ubuntu 20.04/glibc 2.31 and mounts the root
  `.venv` from the host. Wheels previously installed in a newer-glibc runtime
  caused reproducible cryptography and Taichi loader failures in the container.
- The read-only home bind is currently expected to provide the Python 3.12
  interpreter behind that `.venv`.
- TrackerSplat needs build-essential, CUDA 12.4 nvcc, libGL, FFmpeg, and COLMAP;
  these participate in ABI, linking, or executable discovery.

## Proposed boundary

- Image: CUDA devel toolchain, compiler, system shared libraries/executables,
  Python version, and the sole container-side `.venv` populated only by pip
  commands using Harness constraints.
- Runtime mounts: source repositories, read-only `data/` and `checkpoints/`, writable
  output, and optional named download caches.
- Host injection: GPU devices and compatible driver libraries through NVIDIA
  Container Runtime. Do not manually bind individual driver libraries.
- Do not mount the host `.venv` into an image with a different OS/glibc ABI.

## Validation

After implementing the layout, rebuild from an empty cache and verify the five
project smokes plus TrackerSplat CUDA extension compilation for SM 8.6.

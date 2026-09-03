# TrackerSplat

## Role

Point-tracking-driven dynamic 3D Gaussian reconstruction: dataset extraction,
camera initialization, first-frame training, motion estimation, rendering, and
video assembly.

## Safest experiment shape

Avoid batch shell scripts for initial harness experiments. Prefer a targeted
Python invocation from `TrackerSplat/` for one sequence, a few frames, low batch
size/iteration counts, an explicit pipeline, and a unique destination under
`../output/harness-runs/<experiment-id>/`.

The relevant module is typically:

```bash
mkdir -p ../output/harness-runs/<experiment-id>/taichi-cache
TI_OFFLINE_CACHE_FILE_PATH=../output/harness-runs/<experiment-id>/taichi-cache \
  ../.venv/bin/python -m trackersplat.motionestimation <explicit bounded args>
```

Inspect the current CLI before constructing exact arguments; do not copy a batch
script's full matrix into the main context. The explicit Taichi cache path is
required on hosts where the default `~/.cache/taichi` location is not writable.

## Harness rules

- `tools/extract_*`, `init_*`, `save_*`, render, and merge scripts may contain
  `rm -rf`, overwrite, unzip, move, or all-dataset loops. Do not run them as a
  generic smoke test.
- The nested submodules are initialized at their recorded gitlinks and the
  repository-local target install contains all three CUDA extensions. Run from
  `TrackerSplat/`; the package is intentionally not installed globally into
  `.venv`. Rebuild only when explicitly requested, because the current host has
  no local compiler/nvcc toolchain.
- Existing-result skip behavior can mistake a failed partial output for success.
  Always use a unique run directory and verify expected metrics/artifacts.
- On failure, remove or quarantine only the current unique run output and revert
  only attempt-owned code. Never clean shared root `data/` or `output/`.
- The documented NumPy/CUDA stack conflicts with other submodules; preflight the
  active environment.

## Validated runtime smoke

On Python 3.12.8 and protected Torch 2.6.0+cu124, package import, the
`motionestimation` and `pointtracking` CLIs, and actual GPU calls into the local
KNN, featurefusion, and motionfusion extensions pass. Historical
gaussian-splatting and reduced-3dgs CUDA functions and InstantSplat's CroCo RoPE
extension also load/execute without an ABI error. This validates the installed
code layer, not model weights, datasets, COLMAP preprocessing, or a full
reconstruction.

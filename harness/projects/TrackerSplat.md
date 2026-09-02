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
../.venv/bin/python -m trackersplat.motionestimation <explicit bounded args>
```

Inspect the current CLI before constructing exact arguments; do not copy a batch
script's full matrix into the main context.

## Harness rules

- `tools/extract_*`, `init_*`, `save_*`, render, and merge scripts may contain
  `rm -rf`, overwrite, unzip, move, or all-dataset loops. Do not run them as a
  generic smoke test.
- Several nested CUDA-related submodules may be uninitialized and dependencies
  require compilation. Do not initialize/build/install automatically.
- Existing-result skip behavior can mistake a failed partial output for success.
  Always use a unique run directory and verify expected metrics/artifacts.
- On failure, remove or quarantine only the current unique run output and revert
  only attempt-owned code. Never clean shared root `data/` or `output/`.
- The documented NumPy/CUDA stack conflicts with other submodules; preflight the
  active environment.


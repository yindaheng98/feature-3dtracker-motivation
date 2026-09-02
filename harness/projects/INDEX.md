# Submodule Experiment Index

Read only the row and project detail relevant to the current task. Delegate
cross-project comparison or broad code exploration to `code_explorer`.

| Project | Purpose | Safest first validation | Main risk | Detail |
| --- | --- | --- | --- | --- |
| SpaTrackerV2 | Offline RGB/RGBD 3D point tracking inference | Static checks, then a short/sparse single-video inference | Two large CUDA models, downloads, fixed result path, OOM | [detail](SpaTrackerV2.md) |
| Open-d4rt | 4D reconstruction/tracking and WorldTrack evaluation | `LIMIT_SEQS=1`, one subset, small query chunks | Defaults can require 8 GPUs and OOM; weights/data may be absent | [detail](Open-d4rt.md) |
| MV-TAP | Multi-view 3D point tracking | Static checks, then one explicit dataset/eval target | Full training/eval, W&B, and view-copy operations are expensive | [detail](MV-TAP.md) |
| TrackerSplat | Dynamic Gaussian reconstruction using point tracks | Target one sequence, few frames/iterations, unique output | Batch scripts overwrite/delete/move data and fan out over many runs | [detail](TrackerSplat.md) |

## Shared environment caveat

The root `.venv` (Python 3.12.8, Torch 2.11.0+cu128) is the repository's only
Python environment. The four projects' documented Python, PyTorch, and NumPy
pins conflict, so do not install their raw requirement files. Install through
the manifests under `harness/dependencies/`, preserve the protected stack, and
run an import/version preflight after every dependency change. Do not create a
per-submodule environment.

The shared Python dependency set is installed and healthy. TrackerSplat's
`gaussian-splatting`, `InstantSplat`, `reduced-3dgs`, and consequently
`ExtrinsicInterpolator` remain unavailable until its nested submodules and a
CUDA 12.8-compatible `nvcc`/`CUDA_HOME` are provided. SpaTrackerV2 also has one
source compatibility issue: it imports the removed, unused
`torchvision.io.write_video` symbol; its remaining import graph succeeds when
that unused legacy import is bypassed. Details are in
`../dependencies/native-prerequisites.md` and experiment
`EXP-20260902T035007Z-af34`.

The host exposes four RTX A5000 GPUs, while some scripts assume eight. A command
requiring more than one GPU, model/data downloads, training, full evaluation, or
destructive/overwriting flags must be surfaced as an expensive action before it
is launched.

Root `data/` and `output/` are mainly TrackerSplat-oriented and are not implicitly
mapped to every submodule's expected relative paths. Pass explicit paths and use
a unique `output/harness-runs/<experiment-id>/` directory.

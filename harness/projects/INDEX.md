# Submodule Experiment Index

Read only the row and project detail relevant to the current task. Delegate
cross-project comparison or broad code exploration to `code_explorer`.

| Project | Purpose | Safest first validation | Main risk | Detail |
| --- | --- | --- | --- | --- |
| SpaTrackerV2 | Offline RGB/RGBD 3D point tracking inference | Static checks, then a short/sparse single-video inference | Two large CUDA models, downloads, fixed result path, OOM | [detail](SpaTrackerV2.md) |
| Open-d4rt | 4D reconstruction/tracking and WorldTrack evaluation | `LIMIT_SEQS=1`, one subset, small query chunks | Defaults can require 8 GPUs and OOM; weights/data may be absent | [detail](Open-d4rt.md) |
| MV-TAP | Multi-view 3D point tracking | Static checks, then one explicit dataset/eval target | Full training/eval, W&B, and view-copy operations are expensive | [detail](MV-TAP.md) |
| TrackerSplat | Dynamic Gaussian reconstruction using point tracks | Target one sequence, few frames/iterations, unique output | Batch scripts overwrite/delete/move data and fan out over many runs | [detail](TrackerSplat.md) |
| LAPA | Transformer-based multi-camera 3D point tracking | Entrypoint help, then its built-in synthetic model forward | Real inference needs checkpoint, MC metadata, HDF5 feature cache, and track NPZ | [detail](LAPA.md) |

## Shared environment caveat

The root `.venv` (Python 3.12.8, Torch 2.6.0+cu124) is the repository's only
Python environment. The five projects' documented Python, PyTorch, and NumPy
pins conflict, so do not install their raw requirement files. Install through
the manifests under `harness/dependencies/`, preserve the protected stack, and
run an import/version preflight after every dependency change. Do not create a
per-submodule environment.

The shared Python dependency set is installed. LAPA, MV-TAP, Open-d4rt, and
SpaTrackerV2 pass model-construction or synthetic-forward smokes on the selected
stack. TrackerSplat's historical external packages and nested submodules are
present, but its own three CUDA extensions cannot be built until the host
provides `gcc`, `g++`, and CUDA 12.4 `nvcc`/`CUDA_HOME`. InstantSplat's standard
dense path also needs the system `libGL.so.1`. Details are in
`../dependencies/native-prerequisites.md` and experiment
`EXP-20260902T224607Z-fe07`.

The host exposes four RTX A5000 GPUs, while some scripts assume eight. A command
requiring more than one GPU, model/data downloads, training, full evaluation, or
destructive/overwriting flags must be surfaced as an expensive action before it
is launched.

Root `data/` and `output/` are mainly TrackerSplat-oriented and are not implicitly
mapped to every submodule's expected relative paths. Pass explicit paths and use
a unique `output/harness-runs/<experiment-id>/` directory.

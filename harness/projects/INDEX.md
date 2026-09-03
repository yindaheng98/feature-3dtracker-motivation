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

The shared Python dependency set is installed. All five projects pass bounded
executable smokes on the selected stack: LAPA and Open-d4rt synthetic forwards,
MV-TAP a tiny CUDA forward, SpaTrackerV2 full model construction, and
TrackerSplat imports/CLIs plus real calls into its KNN, featurefusion, and
motionfusion CUDA extensions. TrackerSplat's historical external native
extensions also execute successfully. The sole `pip check` complaint is a
distribution-name mismatch (`opencv-python` versus the working
`opencv-python-headless` provider), disproved with an actual encode/color-convert
test. The durable evidence summary is in
`../dependencies/native-prerequisites.md`.

System `libGL.so.1`, COLMAP 3.6 CPU SIFT, FFmpeg 4.2.7, and the ImageIO bundled
FFmpeg have passed bounded tests. Full dataset workflows may still require
project weights, explicit data paths, and newer/GPU-enabled system tools. See
`../dependencies/native-prerequisites.md` for ownership and escalation rules.

The host exposes four RTX A5000 GPUs, while some scripts assume eight. A command
requiring more than one GPU, model/data downloads, training, full evaluation, or
destructive/overwriting flags must be surfaced as an expensive action before it
is launched.

Root `data/` and `output/` are mainly TrackerSplat-oriented and are not implicitly
mapped to every submodule's expected relative paths. Pass explicit paths and use
a unique `output/harness-runs/<experiment-id>/` directory.

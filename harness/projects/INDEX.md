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

The root `.venv` is Python 3.12.8 and currently does not provide the different
PyTorch/CUDA stacks described by all four projects. Their documented Python,
PyTorch, and NumPy requirements conflict. Always run an import/version preflight;
do not auto-install or switch dependencies during an experiment. A future
environment decision should likely use one environment per submodule.

The host exposes four RTX A5000 GPUs, while some scripts assume eight. A command
requiring more than one GPU, model/data downloads, training, full evaluation, or
destructive/overwriting flags must be surfaced as an expensive action before it
is launched.

Root `data/` and `output/` are mainly TrackerSplat-oriented and are not implicitly
mapped to every submodule's expected relative paths. Pass explicit paths and use
a unique `output/harness-runs/<experiment-id>/` directory.


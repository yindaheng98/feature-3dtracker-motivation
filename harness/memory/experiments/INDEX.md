# Experiment Index

Canonical experiment cards live under `YYYY/MM/EXP-*.md`. The raw run content is
stored separately under `output/harness-runs/<experiment-id>/`.

<!-- experiment-rows:start -->
| ID | Date | Status | Scope | Hypothesis | Headline | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-20260902T232819Z-6ddc | 2026-09-02 | running | root .venv and five submodules; dependency repair if needed; no model/data downloads | The user's historical TrackerSplat dependency installation plus the existing LAPA shared dependencies allow all five projects to pass bound… | started | [EXP-20260902T232819Z-6ddc](2026/09/EXP-20260902T232819Z-6ddc.md) |
| EXP-20260902T224607Z-fe07 | 2026-09-02 | inconclusive | root .venv dependency repair and Look-Around-and-Pay-Attention-LAPA- import validation; no submodule source edits | LAPA and newly installed TrackerSplat dependencies can coexist in the sole root .venv with the user's current Torch 2.6.0+cu124 stack uncha… | LAPA, MV-TAP, Open-d4rt, and SpaTrackerV2 passed executable smokes on protected Torch 2.6.0+cu124; TrackerSplat remains blocked by missing … | [EXP-20260902T224607Z-fe07](2026/09/EXP-20260902T224607Z-fe07.md) |
| EXP-20260902T035007Z-af34 | 2026-09-02 | successful | root .venv dependency resolution and import validation for four submodules; no source changes inside submodules | A current shared set of non-PyTorch dependencies for MV-TAP, Open-d4rt, SpaTrackerV2, and TrackerSplat can be installed in the root .venv w… | Shared Python dependencies installed in root .venv; pip check clean, protected Torch/CUDA stack unchanged, imports and XFormers CUDA kernel… | [EXP-20260902T035007Z-af34](2026/09/EXP-20260902T035007Z-af34.md) |
<!-- experiment-rows:end -->

When this index grows past 20 KiB, split closed rows into monthly archive indexes
and retain routes here.


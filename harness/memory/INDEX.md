# Project Memory Index

Updated: 2026-09-03T07:13:01Z
Schema: memory-v1
Budget: no more than 12 KiB or 200 lines

This is the only long-term memory directory that the main agent reads by
default. Follow links only when they are relevant to the current request.

## Current objective

Build and use a lightweight experiment harness for iterative 3D-tracker research
across the five submodules while keeping evidence reproducible and context small.

## Active threads

| ID | Scope | Status | One-line state | Detail |
| --- | --- | --- | --- | --- |
| HARNESS-001 | repository root | ready | Memory, failure rollback, context delegation, and flexible experiment-code guidance adapt to each actual experiment | [Harness README](../README.md) |

## Recent reusable experiment findings

| Date/topic | Reusable conclusion | Detail |
| --- | --- | --- |
| 2026-09-03 · Panoptic four-project bring-up | SpaTrackerV2, Open-d4rt, MV-TAP, and LAPA all completed real-checkpoint forwards on shared `juggle_7`; adapters and bounded metrics are reproducible. | [result](experiments/panoptic-four-project-bringup.md) |
| 2026-09-02 · shared environment and five-project smoke validation | Earlier setup checks were consolidated into the environment ownership and troubleshooting guide. | [environment guide](../dependencies/native-prerequisites.md) |

Keep at most 12 recent routes here. The full directory is in
[experiments/INDEX.md](experiments/INDEX.md).

## Active decisions

| ID | Decision | Detail |
| --- | --- | --- |
| DEC-HARNESS-001 | Raw artifacts use a task-appropriate layout under `output/`; durable memory stores only compact summaries and relative evidence paths. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-002 | The main agent is the only canonical memory-index writer; subagents are bounded read-only context filters. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-003 | Any attempt that misses the user's acceptance criteria must restore attempt-owned code before exit; an unverified rollback blocks later code attempts. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-004 | The root `.venv` is the only Python environment; dependency changes use explicit pip commands plus protected-stack constraints, never direct environment-file edits. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-005 | Build TrackerSplat with its documented repository-local `pip install --target . --upgrade --no-deps .` layout, invoked through the root `.venv`. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-006 | Agent owns authorized Python/wheel repairs; the user/host owns durable system, toolchain, driver, container-ABI, input, and access prerequisites. | [environment guide](../dependencies/native-prerequisites.md) |
| DEC-HARNESS-008 | Use root `checkpoints/` with acquisition-based routing: shared standard Hugging Face/Torch caches and flat standalone files. | [decision card](decisions/DEC-HARNESS-008.md) |
| DEC-HARNESS-009 | Delete fully abandoned Harness content and all references when no surviving project state requires its history. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-010 | Keep Harness infrastructure in `harness/`; organize experiment code flexibly under `experiments/` using the simplest practical layout. | [decision card](decisions/DEC-HARNESS-010.md) |

## Knowledge routes

| Topic | One-line summary | Route |
| --- | --- | --- |
| Projects | Five submodules have different dependencies, entrypoints, and risk levels. | [project index](../projects/INDEX.md) |
| Setup and operation | The root README documents host/GPU prerequisites, submodules, OverlayFS, the sole `.venv`, constrained shared dependencies, TrackerSplat native builds, validation, Codex startup, and Harness behavior. | [root README](../../README.md) |
| TrackerSplat historical dependencies | External repos are pinned to their last commits before TrackerSplat HEAD on 2025-11-23; nested CUDA submodules use exact gitlinks. | [pin table](../dependencies/trackersplat-historical-pins.md) |
| Host CUDA extension builds | RTX A5000 requires SM 8.6; current Torch is 2.6.0+cu124. The runtime can execute the built TrackerSplat CUDA extensions but still has no `gcc`/`nvcc` for rebuilds. | [code index](code/INDEX.md) |
| Environment troubleshooting | Diagnose Python, wheel ABI, native toolchain, system/image, GPU, permissions, and external inputs separately; route each layer to Agent or user ownership. | [environment guide](../dependencies/native-prerequisites.md) |
| Model checkpoints | The normalized root contains load-validated Spa Front+Offline, Open-d4rt 32-frame, LAPA Joint, and DINOv2 in the HF cache, plus flat MV-TAP/TrackerSplat files. | [checkpoint guide](../dependencies/model-checkpoints.md) |
| Experiment code | Experiment code lives under flexibly organized `experiments/`; reuse or isolate it according to the simplest practical implementation, and keep records/scripts many-to-many. | [workspace policy](../../experiments/README.md) |
| 3D tracking datasets | TAPVid-3D/WorldTrack/LAPA-MC directly assess 3D trajectories; MV-TAP currently reports projected 2D tracking, while TrackerSplat mainly reports reconstruction quality. | [dataset map](data/tracking-evaluation-datasets.md) |
| Panoptic adaptation | Use TAPVid-3D PStudio annotations for arbitrary-point ground truth; export only the project-specific formats needed, while raw Panoptic alone supports a custom skeletal-joint test. | [adaptation map](data/panoptic-studio-adaptation.md) |
| Code state | Record each affected submodule's commit and dirty paths; root status is insufficient. | [code index](code/INDEX.md) |
| Data | `data/`, `output/`, and `checkpoints/` are large overlay trees; inspect only explicit bounded paths. | [data index](data/INDEX.md) |

## Open ideas

See [ideas/INDEX.md](ideas/INDEX.md). Keep only concrete, testable ideas here
when they become immediately relevant.

## Archive routes

See [archive/INDEX.md](archive/INDEX.md).

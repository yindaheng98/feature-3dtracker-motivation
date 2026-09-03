# Project Memory Index

Updated: 2026-09-03T00:13:22Z
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
| HARNESS-001 | repository root | ready | Memory, bounded run capture, failure rollback, context delegation, and the user-facing setup/run guide are installed and self-checked | [Harness README](../README.md) |

## Recent experiments

<!-- experiment-rows:start -->
| ID | Date | Status | Scope | Headline | Detail |
| --- | --- | --- | --- | --- | --- |
| — | 2026-09-02 | consolidated | shared environment and five-project smoke validation | Five setup/compatibility experiments were consolidated into the environment ownership and troubleshooting guide; raw run artifacts remain under `output/harness-runs/`. | [environment guide](../dependencies/native-prerequisites.md) |
<!-- experiment-rows:end -->

Keep at most 12 closed experiments here. The full directory is in
[experiments/INDEX.md](experiments/INDEX.md).

## Active decisions

| ID | Decision | Detail |
| --- | --- | --- |
| DEC-HARNESS-001 | Raw logs and artifacts live in `output/harness-runs/`; durable memory stores only compact summaries and relative evidence paths. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-002 | The main agent is the only canonical memory-index writer; subagents are bounded read-only context filters. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-003 | Any attempt that misses the user's acceptance criteria must restore attempt-owned code before exit; an unverified rollback blocks later code attempts. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-004 | The root `.venv` is the only Python environment; dependency changes use explicit pip commands plus protected-stack constraints, never direct environment-file edits. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-005 | Build TrackerSplat with its documented repository-local `pip install --target . --upgrade --no-deps .` layout, invoked through the root `.venv`. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-006 | Agent owns authorized Python/wheel repairs; the user/host owns durable system, toolchain, driver, container-ABI, input, and access prerequisites. | [environment guide](../dependencies/native-prerequisites.md) |

## Knowledge routes

| Topic | One-line summary | Route |
| --- | --- | --- |
| Projects | Five submodules have different dependencies, entrypoints, and risk levels. | [project index](../projects/INDEX.md) |
| Setup and operation | The root README documents host/GPU prerequisites, submodules, OverlayFS, the sole `.venv`, constrained shared dependencies, TrackerSplat native builds, validation, Codex startup, and Harness behavior. | [root README](../../README.md) |
| TrackerSplat historical dependencies | External repos are pinned to their last commits before TrackerSplat HEAD on 2025-11-23; nested CUDA submodules use exact gitlinks. | [pin table](../dependencies/trackersplat-historical-pins.md) |
| Host CUDA extension builds | RTX A5000 requires SM 8.6; current Torch is 2.6.0+cu124. The runtime can execute the built TrackerSplat CUDA extensions but still has no `gcc`/`nvcc` for rebuilds. | [code index](code/INDEX.md) |
| Environment troubleshooting | Diagnose Python, wheel ABI, native toolchain, system/image, GPU, permissions, and external inputs separately; route each layer to Agent or user ownership. | [environment guide](../dependencies/native-prerequisites.md) |
| Code state | Record each affected submodule's commit and dirty paths; root status is insufficient. | [code index](code/INDEX.md) |
| Data | `data/` and `output/` are multi-terabyte, high-fan-out trees; inspect only explicit bounded paths. | [data index](data/INDEX.md) |

## Open ideas

See [ideas/INDEX.md](ideas/INDEX.md). Keep only concrete, testable ideas here
when they become immediately relevant.

## Archive routes

See [archive/INDEX.md](archive/INDEX.md).

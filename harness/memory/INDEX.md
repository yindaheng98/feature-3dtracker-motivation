# Project Memory Index

Updated: 2026-09-02T02:58:00Z
Schema: memory-v1
Budget: no more than 12 KiB or 200 lines

This is the only long-term memory directory that the main agent reads by
default. Follow links only when they are relevant to the current request.

## Current objective

Build and use a lightweight experiment harness for iterative 3D-tracker research
across the four submodules while keeping evidence reproducible and context small.

## Active threads

| ID | Scope | Status | One-line state | Detail |
| --- | --- | --- | --- | --- |
| HARNESS-001 | repository root | ready | Memory, bounded run capture, failure blocking, and context delegation are installed and self-checked | [Harness README](../README.md) |

## Recent experiments

<!-- experiment-rows:start -->
| ID | Date | Status | Scope | Headline | Detail |
| --- | --- | --- | --- | --- | --- |
<!-- experiment-rows:end -->

Keep at most 12 closed experiments here. The full directory is in
[experiments/INDEX.md](experiments/INDEX.md).

## Active decisions

| ID | Decision | Detail |
| --- | --- | --- |
| DEC-HARNESS-001 | Raw logs and artifacts live in `output/harness-runs/`; durable memory stores only compact summaries and relative evidence paths. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-002 | The main agent is the only canonical memory-index writer; subagents are bounded read-only context filters. | [decisions index](decisions/INDEX.md) |
| DEC-HARNESS-003 | Any attempt that misses the user's acceptance criteria must restore attempt-owned code before exit; an unverified rollback blocks later code attempts. | [decisions index](decisions/INDEX.md) |

## Knowledge routes

| Topic | One-line summary | Route |
| --- | --- | --- |
| Projects | Four submodules have different dependencies, entrypoints, and risk levels. | [project index](../projects/INDEX.md) |
| Code state | Record each affected submodule's commit and dirty paths; root status is insufficient. | [code index](code/INDEX.md) |
| Data | `data/` and `output/` are multi-terabyte, high-fan-out trees; inspect only explicit bounded paths. | [data index](data/INDEX.md) |

## Open ideas

See [ideas/INDEX.md](ideas/INDEX.md). Keep only concrete, testable ideas here
when they become immediately relevant.

## Archive routes

See [archive/INDEX.md](archive/INDEX.md).

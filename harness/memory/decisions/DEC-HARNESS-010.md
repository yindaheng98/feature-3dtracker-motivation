---
schema: harness.decision/v1
id: DEC-HARNESS-010
created_at: 2026-09-03T03:22:57Z
status: active
supersedes: []
---

# Decision: Separate experiment code from Harness infrastructure

## Context

Future work will create experiment-specific launchers, adapters, evaluations,
configurations, and reusable research utilities. Keeping those files inside the
Harness would mix project experiments with the system that records and governs
them. Experiment code may naturally be shared or composed across several tests.

## Decision

Reserve `harness/` for Harness instructions, tools, templates, dependency
guidance, and durable memory. Put newly written experiment-specific code under
root `experiments/`, organized by a coherent individual experiment or family.

At the start of every meaningful experiment, explicitly choose either to reuse
an existing directory or create a new one according to whichever produces the
simplest clear implementation. Experiment memory and experiment code are
many-to-many; folder and script names follow the needs of the code.

## Evidence

- The user selected `experiments/` as the experiment-code workspace and required
  `harness/` to contain only Harness-related content.
- Related experiments benefit from shared utilities, while unrelated assumptions
  and unclear rollback ownership are strong reasons to isolate a new directory.
- A single experiment may compose several scripts, and one reusable script may
  support several experiment records.

## Consequences

- Directory structure, commands, progress descriptions, logs, metrics, and
  artifact formats are selected from the actual experiment and user goal.
- Compact experiment memory notes code paths and entry points when useful for
  reproduction.
- Raw artifacts use a task-appropriate location and format under `output/`.

## Revisit trigger

The workspace grows enough to require a formal package layout, ownership split,
or a searchable code-component registry.

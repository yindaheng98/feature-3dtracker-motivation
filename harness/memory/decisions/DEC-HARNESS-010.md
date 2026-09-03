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
them. Requiring one code directory per record would also duplicate useful code
and misrepresent experiments that compose several scripts.

## Decision

Reserve `harness/` for Harness instructions, tools, templates, dependency
guidance, and durable memory. Put newly written experiment-specific code under
root `experiments/`, organized by a coherent individual experiment or family.

At the start of every meaningful experiment, explicitly choose either to reuse
an existing directory or create a new one, and record the primary directory,
placement rationale, and all scripts/components used. Experiment records and
experiment code are many-to-many; neither folder names nor script names need to
match Harness experiment IDs.

## Evidence

- The user selected `experiments/` as the experiment-code workspace and required
  `harness/` to contain only Harness-related content.
- Related experiments benefit from shared utilities, while unrelated assumptions
  and unclear rollback ownership are strong reasons to isolate a new directory.
- A single experiment may compose several scripts, and one reusable script may
  support several experiment records.

## Alternatives considered

- One directory per Harness experiment ID was rejected because it would force
  duplication and prevent natural reuse.
- Storing runnable experiment code in `harness/` was rejected because it blurs
  infrastructure and research-code ownership.

## Consequences

- `harness/tools/experiment.py start` requires `--code-mode reuse|new` and a
  repository-relative `--code-dir` strictly below `experiments/`.
- `reuse` requires the selected directory to exist; `new` requires it not to
  exist and creates it.
- Generated experiment cards capture the primary directory and provide fields
  for the complete script/component list and placement rationale.
- Raw artifacts remain under `output/harness-runs/<experiment-id>/`.

## Revisit trigger

The workspace grows enough to require a formal package layout, ownership split,
or a searchable code-component registry.

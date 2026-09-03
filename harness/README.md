# Lightweight Experiment Harness

This directory provides the persistent, repository-local part of the harness.
The root `AGENTS.md` is loaded automatically when Codex starts from the repository
root; users do not need to repeat the memory or experiment instructions.

## Use

Start Codex from the repository root as described by the root `README.md`. At the
start of each turn the agent reads only:

- `harness/memory/ACTIVE.md`
- `harness/memory/INDEX.md`

It follows links or delegates detail retrieval only when the current request
requires that history.

Inspect the actual project and use direct commands, progress descriptions, logs,
metrics, and artifact formats suited to the current task. If an experiment
produces large output, store it under an appropriate path in `output/` and
inspect only the relevant portions in the main context. Save a compact memory
afterward only when the result is reusable.

## Experiment code workspace

`harness/` is reserved for the Harness itself. Put new experiment-specific
modules, launch/evaluation scripts, notebooks, and configurations under the root
`experiments/` directory. Keep raw logs, metrics, visualizations, and generated
checkpoints in a suitable location under `output/`, not beside the code.

For each experiment, use the simplest arrangement that fits: reuse coherent
code when useful, or create a directory when isolation makes the implementation
clearer. Let the work determine its directory structure, names, and entry points.
An experiment may use several scripts or shared components, and a script may
serve several experiments.

Note actual code paths and entry points in memory only when useful for
reproduction. See [`experiments/README.md`](../experiments/README.md) for the
workspace policy.

## Single Python environment

The repository root `.venv` is the only allowed Python environment. Do not
activate it, create another environment for a submodule, or edit files inside
it. Invoke its interpreter explicitly and make dependency changes only through
commands from the repository root:

```bash
.venv/bin/python -m pip install --upgrade \
  -c harness/dependencies/protected-stack.txt \
  -r harness/dependencies/shared-requirements.txt

.venv/bin/python -m pip install --upgrade --no-deps \
  -c harness/dependencies/protected-stack.txt \
  -r harness/dependencies/git-requirements.txt

.venv/bin/python -m pip check
```

Do not install the five submodules' original requirement files directly: their
documented Torch, CUDA, NumPy, and legacy API pins conflict. See
`dependencies/native-prerequisites.md` for the environment ownership decision
table, safe repair workflow, known compatible versions, and escalation rules
before attempting TrackerSplat's CUDA extensions or external system tools.

If an attempt does not satisfy the user's acceptance criteria, the agent must
restore only that attempt's code/configuration changes before ending the turn,
then compare each affected repository's status with the recorded baseline. A
failed attempt with unverified cleanup must be described in `memory/ACTIVE.md`
and blocks the next code attempt. Raw evidence may be retained under `output/`;
it must not be used as proof that the failed code should remain.

## Memory layout

- `memory/INDEX.md`: small hot directory, automatically read.
- `memory/ACTIVE.md`: short free-form note for unfinished work or pending
  restoration, automatically read.
- `memory/*/INDEX.md`: category routing indexes, read on demand.
- `memory/experiments/`: compact experiment findings and their index, created
  only as the work requires.
- `memory/inbox/`: unique subagent candidate summaries awaiting main-agent merge.
- `memory/archive/`: cold monthly/topic summaries.
- `output/`: raw and potentially large generated artifacts, arranged according
  to the experiment.

Memory contains observations, evidence paths, interpretations, conclusions, and
decisions—not chat transcripts or raw logs.

When the user requests compaction, obsolete experiment details may be replaced
by one durable topical guide after the main agent verifies that all reusable
facts, decisions, and routes were merged. Update indexes and references in the
same change. Raw artifacts remain under `output/` unless the user separately
asks to delete them.

## Local specialized agents

Project-scoped read-only agent definitions live in `.codex/agents/`:

- `memory_researcher`
- `experiment_analyst`
- `code_explorer`

They are context filters, not extra memory writers. The main agent remains the
single canonical writer.

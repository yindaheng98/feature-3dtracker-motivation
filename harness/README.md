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

For an experiment, the agent should use:

```bash
.venv/bin/python harness/tools/experiment.py start \
  --title "short title" \
  --hypothesis "testable claim" \
  --scope "TrackerSplat"

.venv/bin/python harness/tools/experiment.py run EXP-... \
  --timeout 300 -- <command> <args...>

.venv/bin/python harness/tools/experiment.py finish EXP-... \
  --status successful \
  --summary "measured result" \
  --next "next useful test"
```

The run wrapper stores complete stdout/stderr and a JSON manifest under
`output/harness-runs/<experiment-id>/`. It prints only a bounded tail to the
agent context. The generated Markdown experiment card is the durable summary.

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

Do not install the four submodules' original requirement files directly: their
documented Torch, CUDA, NumPy, and legacy API pins conflict. See
`dependencies/native-prerequisites.md` before attempting TrackerSplat's CUDA
extensions or external system tools.

If an attempt does not satisfy the user's acceptance criteria, the agent must
restore only that attempt's code/configuration changes before ending the turn,
then compare each affected repository's status with the recorded baseline. A
failed attempt with unverified cleanup remains `rollback_pending` and blocks the
next code attempt. Raw evidence in the unique run directory may be retained; it
must not be used as proof that the failed code should remain.

Run a harness self-check with:

```bash
.venv/bin/python harness/tools/experiment.py check
```

## Memory layout

- `memory/INDEX.md`: small hot directory, automatically read.
- `memory/ACTIVE.md`: crash/unfinished-run marker, automatically read.
- `memory/*/INDEX.md`: category routing indexes, read on demand.
- `memory/experiments/YYYY/MM/*.md`: one experiment per file.
- `memory/inbox/`: unique subagent candidate summaries awaiting main-agent merge.
- `memory/archive/`: cold monthly/topic summaries; canonical experiment cards stay.
- `output/harness-runs/`: raw and potentially large generated artifacts.

Memory contains observations, evidence paths, interpretations, conclusions, and
decisions—not chat transcripts or raw logs.

## Local specialized agents

Project-scoped read-only agent definitions live in `.codex/agents/`:

- `memory_researcher`
- `experiment_analyst`
- `code_explorer`

They are context filters, not extra memory writers. The main agent remains the
single canonical writer.

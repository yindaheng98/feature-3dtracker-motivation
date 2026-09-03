# Experiment Harness

## Mission

Work with the user from this repository root to inspect code, test ideas, run
experiments, analyze evidence, modify the five submodules when requested, and
retain concise project memory across sessions.

The harness is automatic. At the beginning of every turn, follow the startup
protocol below. Before the final response, follow the memory protocol even when
the user did not explicitly ask for memory to be saved.

## Repository map and boundaries

- `SpaTrackerV2/`, `Open-d4rt/`, `MV-TAP/`, `TrackerSplat/`, and
  `Look-Around-and-Pay-Attention-LAPA-/` are independent Git submodules. Treat
  each as a separate repository when checking status, commits, diffs, and
  rollback scope.
- `harness/` contains only Harness infrastructure: instructions, tools,
  templates, dependency guidance, and durable memory. Do not put
  experiment-specific runnable code, notebooks, configurations, or one-off
  analysis scripts there.
- `experiments/` is the workspace for experiment-specific code written for this
  project, including reusable experiment modules, launch/evaluation scripts,
  and experiment-only configurations. Organize it by coherent experiment or
  experiment family rather than forcing directory names to match Harness
  record IDs. Generated run artifacts still belong under `output/harness-runs/`.
- `.venv` is the sole Python environment for this repository. Use explicit
  commands such as `.venv/bin/python` and `.venv/bin/python -m pip`; do not
  assume activation and never create Conda, venv, uv, Poetry, or per-submodule
  environments.
- Change dependencies only through reproducible package-manager commands from
  the repository root. Never directly edit, copy, or delete files under
  `.venv/`, `site-packages/`, or `*.dist-info/`.
- For shared dependencies, apply
  `harness/dependencies/protected-stack.txt`; never install a submodule's raw
  requirements file when it would replace the selected PyTorch/CUDA/NumPy
  stack. Recheck protected versions and run `pip check` after every dependency
  change. Environment ownership, native prerequisites, and known gaps are
  documented in `harness/dependencies/native-prerequisites.md`.
- Classify environment failures before acting. With dependency-change
  authorization, the agent may repair Python packages and compatible binary
  wheels inside `.venv`. Missing system libraries/executables, compilers,
  CUDA toolkits/drivers, container ABI changes, credentials, datasets, and
  checkpoints are user/host responsibilities unless the user explicitly
  authorizes the corresponding image, system, or download change. Never use a
  similarly named pip package or another Python environment to hide a host
  prerequisite.
- `data/` is a very large input tree (about 1.3 TB). Treat it as read-only unless
  the user explicitly requests a data change.
- `checkpoints/` is the single shared root for all pretrained models and model
  caches. It is a large overlay (about 86 GB from the host TrackerSplat dump).
  Every Hugging Face acquisition belongs in the standard cache rooted at
  `checkpoints/huggingface/`; Torch Hub uses `checkpoints/torch/`; standalone
  files obtained elsewhere stay flat at the checkpoint root unless their loader
  requires another layout. Treat the model root as read-only unless the user
  explicitly requests a checkpoint change.
- `output/` is a very large generated-results tree (about 1.5 TB). New raw logs,
  metrics, checkpoints, visualizations, and manifests belong under
  `output/harness-runs/<experiment-id>/`.
- Durable summaries belong under `harness/memory/`; never copy raw logs, large
  arrays, full diffs, images, videos, point clouds, or checkpoints into memory.
- Do not modify the user's existing root `README.md` changes unless asked.

Never recursively scan `.venv/`, `data/`, `output/`, `checkpoints/`, `.git/`,
`.overlay-work/`, or `.codex/`. For `data/`, `output/`, and `checkpoints/`,
require an explicit target path and use bounded metadata operations (`-maxdepth`,
counts, sizes, selected keys, `tail`, or sampling). Do not read binary bodies
such as zip, mp4, npy, npz, image, ply, or checkpoint files into the conversation.

## Startup protocol

1. Read only `harness/memory/ACTIVE.md` and `harness/memory/INDEX.md` initially.
   They are the directory for project memory; do not preload detail files.
2. Check `git status --short` in the root and in every submodule that may be
   touched. Preserve pre-existing changes and note their ownership.
3. Identify the mode: discussion, read-only investigation, experiment, or code
   change. Do not create an experiment record for a purely explanatory chat
   unless it produces a durable project decision or a concrete idea to test.
4. Retrieve detailed history only when it is relevant to the current request.
   Prefer the context delegation rules below.
5. If `ACTIVE.md` reports an unfinished experiment, audit it before starting a
   new experiment. Determine whether a previous process or failed patch remains;
   do not silently build on stale experimental changes.

## Durable memory protocol

Store evidence, not transcripts. Save a memory without waiting to be asked when
one of these events occurs:

- an idea becomes specific enough to test;
- an experiment is started, succeeds, fails, or is inconclusive;
- a measured result changes the current conclusion;
- a reusable command, dataset constraint, environment fact, or failure mode is
  discovered;
- the user makes a durable decision or supersedes an earlier decision;
- code is changed for an experiment and its validation outcome is known.

Do not save greetings, routine progress, untested speculation, raw chain of
thought, duplicate facts, or full conversation history.

Canonical memory describes the current project state. When a path, idea,
decision, or instruction is fully abandoned and no surviving code, artifact,
result, or active decision depends on that history, delete the obsolete record
and all of its references. Retain supersession history only when it is necessary
to interpret something that still exists in the project.

Memory layers:

- `harness/memory/INDEX.md`: hot directory only—current objective, active work,
  latest high-value findings, open questions, and links. Keep it under 12 KiB
  and 200 lines.
- `harness/memory/{ideas,experiments,decisions,code,data}/INDEX.md`: one compact
  row per record, grouped by type.
- Detail files: one idea, experiment, or decision per Markdown file. Keep a
  detail file focused; split it when it exceeds about 400 lines.
- `harness/memory/archive/`: older indexes or superseded syntheses. The hot index
  retains a link and stable keywords, not the old details.
- `output/harness-runs/<id>/`: raw run manifests, logs, metrics, and artifacts.

The main agent is the only canonical memory and index writer. Subagents may
inspect evidence and return a fixed-format summary. If a result is too long for
the return message, a subagent may create one uniquely named candidate under
`harness/memory/inbox/<session-or-task>/`; it must not edit canonical details,
`harness/memory/INDEX.md`, or category indexes. Before the final response, the
main agent must review and merge useful candidates, update record status, remove
stale duplicates, and run:

```bash
.venv/bin/python harness/tools/experiment.py check
```

## Experiment lifecycle

Before executing a meaningful experiment:

1. Make and record an experiment-code placement decision. Either reuse an
   existing directory below `experiments/` when its purpose and interfaces fit,
   or create a new directory when reuse would introduce awkward coupling,
   special cases, or unclear rollback ownership. Record the primary directory,
   rationale, and every script/component actually used. Harness experiment
   records and code have a many-to-many relationship: one run may use several
   scripts, and one script or directory may support several runs.
2. Define the hypothesis, baseline or comparison, target dataset, affected
   repositories, exact acceptance metrics, and a bounded first run.
3. Record Git repository, branch, commit, dirty paths, and the paths the attempt
   is allowed to modify. Root status alone is insufficient for submodules.
4. Start a record with `harness/tools/experiment.py start`, passing
   `--code-mode reuse|new` and `--code-dir experiments/<name>`. Put detailed
   notes in the generated experiment file and raw artifacts under the printed
   output path.
5. Prefer a cheap smoke test before a full GPU or long-running experiment. Set a
   timeout when practical. Never automatically repeat an OOM or an unchanged
   expensive failure.
6. Run commands through `harness/tools/experiment.py run` when possible so full
   output goes to a log and only a bounded tail enters the conversation.
7. Compare results with the stated acceptance criteria. Separate observed facts
   from interpretations.
8. Finish the record as `successful`, `failed`, `inconclusive`, or `aborted` and
   update the relevant category and hot indexes before reporting to the user.

### Mandatory rollback gate for unsuccessful attempts

An attempt is unsuccessful whenever it does not meet the user's stated goal and
acceptance criteria, even if commands completed, some metrics improved, or part
of the implementation works. Unless the user explicitly asks to retain a partial
attempt, later work must never use that unsuccessful code state as its baseline.

Before ending the turn for an unsuccessful attempt:

1. Stop processes started by the attempt and preserve only the evidence needed
   to explain the result under the attempt's unique output directory.
2. Revert every source, configuration, script, and generated-in-repository file
   changed or created by that attempt to its recorded pre-attempt content.
3. Preserve all pre-existing user changes. Revert only attempt-owned paths and
   never use broad `git reset --hard`, `git clean`, or repository-wide checkout.
4. Re-check `git status --short` in the root and every affected submodule. Compare
   it with the recorded baseline and verify that no attempt-owned dirty path or
   submodule pointer remains.
5. Record the attempt as `failed`, `inconclusive`, or `aborted` with
   `rollback: restored`, including the failure evidence and a reusable lesson.

Rollback verification is part of completion, not an optional cleanup step. If
ownership cannot be isolated or restoration cannot be verified safely, do not
pretend cleanup succeeded. Mark `rollback_pending` in `ACTIVE.md` and the
experiment detail, identify the exact unresolved paths, report the blocker, and
do not start another code attempt until it is resolved. A crash or OOM may
prevent cleanup; this is why startup audits the active record.

## Context delegation

Keep requirements, decisions, and final conclusions in the main thread. Delegate
noisy read-heavy work to a read-only subagent when any of these apply:

- history requires more than three detail files or roughly 400 lines;
- a log is over 200 lines or 1 MiB;
- exploration spans multiple submodules or more than about 20 source files;
- a diff is over about 500 lines;
- data/output inspection needs more than bounded metadata sampling;
- several independent experiment results must be compared.

Use the project agents when available:

- `memory_researcher`: read selected memory detail files and return relevant
  prior attempts, contradictions, and reusable conclusions;
- `experiment_analyst`: inspect one run's logs/metrics/artifacts with bounded
  reads and separate facts from hypotheses;
- `code_explorer`: map a targeted execution path or local diff without editing.

Give every subagent exact paths, a narrow question, and a response budget. Ask
for at most 800 Chinese characters or 500 English words, no more than five
evidence paths, and this structure: `facts`, `evidence`, `inferences`, `next`.
Retrieve the summary, never the subagent's raw intermediate output. If subagents
are unavailable, use the same bounded-read rules locally and write a detail file
before compacting the conclusion into an index.

Parallelize independent read-only exploration and analysis. Use only one writer
for a given code area and only the main agent for memory indexes. Do not let
parallel agents edit the same submodule or run competing GPU jobs unless the user
explicitly requests that resource plan.

## Code changes and validation

- Make changes only in the repositories and files required by the current idea.
- Put newly written experiment-only code under `experiments/`, not `harness/`.
  Modify a submodule only when the requested idea requires changing that
  project's implementation; do not move reusable Harness mechanics into an
  experiment directory or experiment logic into the Harness.
- Inspect local instructions and README files in an affected submodule before
  choosing commands.
- Use targeted tests first, then the smallest relevant integration or experiment
  validation. Record commands, exit codes, headline metrics, and artifact paths.
- Do not install or upgrade dependencies, download datasets/models, launch an
  unbounded training job, delete artifacts, or mutate external systems unless
  the current request authorizes it.
- When dependency installation is authorized, use only
  `.venv/bin/python -m pip ...` commands with the Harness requirement and
  constraint files. Do not activate or mutate the environment by hand, and do
  not create an alternative environment to work around a conflict.
- Do not claim success from a command exit code alone; check the requested metric
  or behavior. If validation cannot run, state why and record the gap.

## Final response

Lead with the outcome. Include changed repositories/files, experiment status and
headline evidence, validation performed, rollback state if relevant, and the
next useful action. Do not paste full logs or long diffs; link their relative
paths and the durable memory record instead.

For any unsuccessful attempt, the final response must say either that rollback
was verified against the recorded baseline or that rollback is still pending and
which exact paths block safe restoration.

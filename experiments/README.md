# Experiment Code Workspace

This directory contains code written specifically to design, launch, evaluate,
or analyze project experiments. Harness implementation and durable records stay
under `harness/`; raw outputs stay in a task-appropriate location under `output/`.

## Organization

Use flexible, coherent directories such as `experiments/<experiment-or-family>/`.
A directory may hold one isolated experiment or reusable code for a family of
related experiments. Scripts and records deliberately have a many-to-many
relationship:

- one Harness experiment may combine several scripts or shared components;
- one script, module, configuration, or directory may support several Harness
  experiments;
- directory names do not need to contain a Harness experiment ID.

A local README describing purpose, entry points, inputs, and shared assumptions
is recommended once a directory contains more than a trivial script.

## Placement for each experiment

Use the simplest arrangement that fits the current work:

- Reuse an existing directory when the new test shares its purpose, interfaces,
  dependencies, and artifact conventions, and changes remain easy to attribute
  and roll back.
- Create a new directory when assumptions or interfaces are incompatible, reuse
  would require unrelated flags or special cases, or isolation makes the code
  easier to understand and roll back.

Use judgment based on the current code and task. Harness records and code remain
many-to-many. When useful for reproduction, list the actual scripts and entry
points in memory.

## Boundaries

- Put reusable Harness mechanics, memory tools, and templates in `harness/`, not
  here.
- Put experiment-only launchers, adapters, evaluations, ablations, and configs
  here rather than in `harness/`.
- Modify a submodule when the experiment genuinely changes that project's
  implementation; keep disposable orchestration and evaluation code here.
- Use `.venv/bin/python` explicitly. Do not create another environment.
- Treat `data/` and `checkpoints/` as read-only unless the user authorizes a
  change.
- Store logs, metrics, generated checkpoints, and media under `output/` using
  whatever layout and formats fit the actual experiment.

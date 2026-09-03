---
schema: harness.decision/v1
id: DEC-HARNESS-007
created_at: 2026-09-03T00:46:00Z
status: active
supersedes: []
---

# Decision: Overlay host TrackerSplat checkpoints at root `checkpoints/`

## Context

`data/` and `output/` already use OverlayFS named volumes so the container sees
the host TrackerSplat trees without copying them. The host also has an ~86 GB
pretrained dump at `/mnt/minorissd4tb/TrackerSplat/checkpoints`. Copying that
tree into the workspace is unnecessary and was cancelled.

## Decision

Mount `/mnt/minorissd4tb/TrackerSplat/checkpoints` with the same OverlayFS
pattern as `data/` and `output/`:

- `lowerdir=/mnt/minorissd4tb/TrackerSplat/checkpoints`
- `upperdir=${PWD}/checkpoints`
- `workdir=${PWD}/.overlay-work/checkpoints`
- container target `${PWD}/checkpoints`

Keep newly downloaded or project-split weights under `data/checkpoints/`. The
root overlay is the existing TrackerSplat dump, not a replacement for that shared
layout.

## Evidence

- `docker-compose.yml` already used this pattern for `data/` and `output/`.
- Bounded listing of the host dump shows a flat `.pth`/`.pt`/`.npz` tree of
  about 86 GB.
- The workspace already had an empty root `checkpoints/` upperdir.

## Alternatives considered

- Copy the dump into `TrackerSplat/checkpoints` or root `checkpoints/`: rejected
  because the tree is large and already present on the host.
- Overlay onto `TrackerSplat/checkpoints` only: rejected; the requested pattern
  matches root `data/` and `output/`.

## Consequences

- After changing Compose overlay options, recreate named volumes with
  `sudo docker compose down -v` before the new mount is visible.
- TrackerSplat's default loader path is still `TrackerSplat/checkpoints`; use an
  ignored symlink to `../checkpoints` if a hard-coded local path is required.
- Treat root `checkpoints/` like `data/`: large, read-only unless the user asks
  to change it, inspect only with bounded metadata.

## Revisit trigger

Move or rename the host dump, or decide that all five projects should share a
single overlay root instead of `data/checkpoints/` plus `checkpoints/`.

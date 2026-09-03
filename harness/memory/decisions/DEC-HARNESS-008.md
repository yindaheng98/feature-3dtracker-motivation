---
schema: harness.decision/v1
id: DEC-HARNESS-008
created_at: 2026-09-03T02:10:12Z
status: active
supersedes: []
---

# Decision: Use root `checkpoints/` for all pretrained weights

## Context

Root `checkpoints/` already exposes the host TrackerSplat weight dump through a
dedicated Docker OverlayFS. A single shared location keeps model discovery,
cache configuration, and container paths consistent across all five projects.

## Decision

Store all five projects' pretrained inputs under root `checkpoints/`. Preserve
TrackerSplat's existing flat filenames. Store every Hugging Face acquisition in
the standard cache rooted at `checkpoints/huggingface/`, use
`checkpoints/torch/` for Torch Hub, and keep other standalone files flat at the
checkpoint root unless an upstream loader requires another layout.

## Evidence

- The user explicitly selected root `checkpoints/` for all future downloads.
- A bounded check found all 14 TrackerSplat filenames published in its README.
- The minimum SpaTrackerV2, Open-d4rt, MV-TAP, and LAPA model set was downloaded
  and load-validated using the acquisition-based layout.
- SpaTrackerV2's hard-coded Hugging Face repository IDs honor `HF_HOME`, so its
  cache can live under `checkpoints/huggingface` without changing source code.

## Consequences

- Set `MODEL_ROOT=$PWD/checkpoints`, `HF_HOME=$MODEL_ROOT/huggingface`, and
  `TORCH_HOME=$MODEL_ROOT/torch` from the repository root.
- Resolve Open-d4rt and LAPA file paths from the shared Hugging Face snapshot;
  pass flat root paths for MV-TAP and TrackerSplat.
- For SpaTrackerV2, download and run with the same `HF_HOME`; optionally set
  `HF_HUB_OFFLINE=1` after downloads complete.
- Treat the overlay as read-only except during an explicitly authorized model
  download or checkpoint maintenance operation.

## Revisit trigger

The checkpoint overlay moves, becomes capacity-constrained, or plain unpacked
SpaTrackerV2 directories are preferred over the Hugging Face cache.

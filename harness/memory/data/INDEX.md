# Data and Output Context Index

## Stable facts

- `data/` is approximately 1.3 TB and contains high-fan-out frame directories,
  videos, archives, NumPy data, and dataset metadata.
- `output/` is approximately 1.5 TB and contains deep scene × variant × parameter
  experiment matrices, including many arrays, images, JSON files, videos, and
  point clouds.
- Root `data/` and `output/` are not automatically equivalent to the relative
  dataset/output paths expected inside every submodule.
- Root `checkpoints/` is an OverlayFS view of
  `/mnt/minorissd4tb/TrackerSplat/checkpoints` (about 86 GB). Treat it as a large
  input tree; inspect only explicit bounded paths.
- Root `checkpoints/` is the shared model store for all five projects. Preserve
  standalone TrackerSplat and MV-TAP files at the root; route every Hugging Face
  acquisition through `checkpoints/huggingface/` and Torch Hub through
  `checkpoints/torch/`. See
  [model checkpoint guide](../../dependencies/model-checkpoints.md).
- A bounded 2026-09-03 filename and size check found all 14 TrackerSplat files
  published in its README. The minimum SpaTrackerV2, Open-d4rt, MV-TAP, LAPA,
  and DINOv2 set was subsequently downloaded and load-validated.
- The five projects' evaluation datasets differ in what they measure. TAPVid-3D,
  WorldTrack, and LAPA's multi-camera derivatives provide direct 3D tracking
  evidence; MV-TAP's current evaluators report projected 2D TAP metrics; the
  TrackerSplat main benchmark reports reconstruction/rendering quality. See the
  [evaluation dataset map](tracking-evaluation-datasets.md).
- Panoptic Studio can be adapted across the projects, but the official
  TAPVid-3D PStudio annotations are needed for arbitrary-point 3D tracking
  ground truth. Raw CMU video/calibration/skeleton data alone supports a custom
  skeletal-joint benchmark, not an equivalent TAPVid-3D benchmark. See the
  [Panoptic adaptation map](panoptic-studio-adaptation.md).

## Inspection policy

- Index at `scene -> method/variant -> run ID`; never create a file-by-file index.
- Read only an explicit target path with bounded depth/count/sample operations.
- Do not inject binary contents, recursive path lists, or complete metric series
  into the main conversation.
- Store new raw artifacts under `output/` using a layout suited to the task.
- Record dataset paths, split names, sampling rules, and relevant hashes or
  manifests in durable memory when useful for reproduction; do not duplicate
  source data.

Add dataset-specific detail files when evaluation planning or an experiment
produces reusable information.

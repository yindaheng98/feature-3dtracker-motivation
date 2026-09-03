# Panoptic four-project bring-up

Started: 2026-09-03

## Goal

Use TAPVid-3D PStudio `minival` plus the matching Dynamic3DGaussians multi-camera
source to run SpaTrackerV2, Open-d4rt, MV-TAP, and LAPA with real pretrained
weights and real data. Require a meaningful metric/output from each project.

## Baseline

- Root has pre-existing Harness data-memory changes; preserve them.
- SpaTrackerV2 `7e12274`, Open-d4rt `403290a`, MV-TAP `b248aea`, LAPA `30012d6`,
  and out-of-scope TrackerSplat `7e6c485` are clean.
- `.venv` is the sole Python environment. GPUs 1–3 were idle at startup; GPU 0
  had about 8 GiB allocated.

## Scope and evidence

- Experiment code: `experiments/panoptic_multitracker/`
- Downloaded/derived data: `data/panoptic_tracking/`
- Generated outputs: `output/panoptic_multitracker/`
- Submodule edits are avoided unless external adapters cannot express the
  integration.

## Acceptance evidence

For one common PStudio scene per project: exact command and checkpoint, input
sample identity, actual model forward, headline metric/output, and bounded error
evidence if it fails. Import-only checks are insufficient.

## Result

Completed on the common `juggle_7` reference sample without modifying any
submodule. All checkpoints were real pretrained weights and all metrics followed
actual model forwards.

| Project | Evaluated input | Result | Evidence |
| --- | --- | --- | --- |
| Open-d4rt 32CLIP | 32 frames, 221 frame-0-visible queries | 0 missing/unexpected keys; APD 0.9792; EPE 0.0581 m | `output/panoptic_multitracker/opend4rt/eval_juggle7_32f/summary.json` |
| SpaTrackerV2 Front + Offline | 16 frames, 32 queries; Front inferred depth/intrinsics from RGB | OA 0.9844; average 3D threshold accuracy 0.1301; median-scaled EPE 0.1180 m | `output/panoptic_multitracker/spatracker/juggle7_16f_metrics.json` |
| MV-TAP | cameras 7/8/9, 16 frames, 32 queries | 0 missing/unexpected keys; 2D AJ 0.8885; triangulated 3D APD 0.9039; MPJPE 0.01424 m | `output/panoptic_multitracker/mvtap/juggle7_16f_metrics.json` |
| LAPA Joint | cameras 7/8/9, 150 frames, 64 queries | APD 27.11; 3D-AJ 20.24; MPJPE 0.09921 m | `output/panoptic_multitracker/lapa/eval_juggle7_real_metrics.json` |

LAPA's three H5 inputs have `use_cotracker=1`, tracker-visible fractions from
0.7301 to 0.8381, and 99.33% of 2D coordinates differ from GT. The validated
runner explicitly passes the eval-cache directory into the dataset and rejects
missing or GT-like caches. Calling the repository evaluator with only
`feature_dir` does not bind `eval_feature_dir` and can silently use projected GT
2D with zero appearance features. DINOv2 loaded from the shared Hugging Face
cache and CoTracker3 Offline from `checkpoints/torch/`.

The LAPA geometry builder found six `juggle` cameras. Its calibration gate
passed with median camera/world round-trip error `2.57e-8` m, median self
reprojection error `4.93e-6` px, and cross-view in-bounds fraction 0.9793.

## Reusable implementation

- `experiments/panoptic_multitracker/prepare_data.py` downloads only the working
  50-file PStudio minival split plus D3G camera data; it does not request the
  unavailable full-eval objects.
- `run_spatracker.py` performs Front RGB depth/intrinsics inference and Offline
  3D tracking without using GT depth.
- `run_mvtap.py` projects one calibrated reference point set into several views,
  runs MV-TAP, retains its native 2D TAP metrics, and triangulates predictions
  for calibrated 3D metrics.
- The exact commands and bounded result table are in
  `experiments/panoptic_multitracker/README.md`.

Data live in `data/panoptic_tracking/`: 50 official minival NPZ files and the
589,086,921-byte D3G archive/extraction. Raw/derived evidence remains under
`output/panoptic_multitracker/`.

These results validate integration, not a fair leaderboard comparison: frame
and point counts differ by model to keep the first bring-up bounded. A later
benchmark should standardize the common sample set, frame range, query set,
visibility convention, and metric protocol before comparing model quality.

## Validation and repository state

- Both new Python adapters and the data preparer passed `py_compile`.
- The four target submodules and out-of-scope TrackerSplat stayed at their
  recorded baseline commits with clean worktrees.
- No unsuccessful source/configuration attempt remains; the sole early MV-TAP
  path-resolution error was fixed in the adapter and the successful rerun
  verified the resulting state.

# 3D tracking evaluation dataset map

Updated: 2026-09-03

## Direct 3D trajectory evaluation

- **TAPVid-3D**: ADT/Aria Digital Twin, DriveTrack, and Panoptic Studio splits;
  world-space 3D trajectories, visibility, cameras, and projected observations.
  SpaTrackerV2 contains AJ/APD/OA metric code and an example that references
  missing loader/collate modules and internal hard-coded paths; a loader adapter
  is required before use.
- **WorldTrack release in Open-d4rt**: `adt_mini`, `po_mini` (PointOdyssey),
  `pstudio_mini`, and `ds_mini` (reported as Dynamic Replica). NPZ packs contain
  frames, `tracks_XYZ`, intrinsics, visibility, and optional extrinsics. The
  repository directly evaluates scale-aligned APD and EPE, including dynamic
  points.
- **TAPVid-3D-MC in LAPA**: multi-camera construction based on TAPVid-3D
  Panoptic Studio plus Dynamic3DGaussians cameras/frames. `minival` is the public
  evaluation split; metrics are APD, OA, 3D-AJ, 2D-AJ, and MPJPE in metres.
- **PointOdyssey-MC in LAPA**: synthetic multi-camera 3D/2D trajectories,
  visibility, and cameras. It is useful for controlled synthetic generalization;
  the repository has train/validation handling but less standalone benchmark
  tooling than TAPVid-3D-MC.

## Multi-view tracking with current 2D metrics

- **MV-TAP** evaluates DexYCB-multiview, Panoptic Studio, Harmony4D, and
  KubricEval. Inputs include synchronized views, cameras, and dataset-specific
  3D track files, but the shipped evaluation reports TAP-style projected 2D
  metrics: occlusion accuracy, threshold accuracy, and average Jaccard. Kubric is
  also its synthetic training source.

## Indirect end-to-end evidence

- **TrackerSplat** uses Neural 3D Video, ST-NeRF, Meet Room, and Dynamic 3D
  Gaussians for dynamic multi-view reconstruction. Its main results are PSNR,
  SSIM, and LPIPS, so they test the downstream effect of tracking rather than 3D
  trajectory accuracy. RH20T is an additional real robot sequence path without
  a main quantitative tracking benchmark.
- TrackerSplat's embedded DoT project evaluates Kubric-CVO and TAP-Vid DAVIS,
  Kinetics, and RGB-Stacking, but these are optical-flow or 2D point-tracking
  benchmarks rather than 3D trajectory ground truth.

## Practical selection

Start with a small WorldTrack subset for the most direct existing local pipeline.
Use Panoptic Studio/PStudio as the closest common real-data basis across
SpaTrackerV2, Open-d4rt, MV-TAP, and LAPA, while accounting for their different
packaging and metrics. Add PointOdyssey for synthetic controlled tests. Use a
TrackerSplat reconstruction set only when the desired outcome is end-to-end
rendering/reconstruction quality.

The contents already present under root `data/` were not inventoried for this
mapping.

## Evidence routes

- `SpaTrackerV2/models/SpaTrackV2/evaluation/example_multithread_eval.py`
- `SpaTrackerV2/models/SpaTrackV2/evaluation/core/tapvid3d_metrics.py`
- `Open-d4rt/README.md` and `Open-d4rt/eval_track3d_in_worldtrack.py`
- `MV-TAP/README.md` and `MV-TAP/model_utils.py`
- `Look-Around-and-Pay-Attention-LAPA-/README.md` and `evaluate_lapa.py`
- `TrackerSplat/README.md` and `TrackerSplat/submodules/dot/README.md`

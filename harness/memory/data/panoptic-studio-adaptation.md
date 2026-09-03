# Panoptic Studio adaptation map

Updated: 2026-09-03

## Size facts

A 2026-09-03 metadata-only check of the URLs used by LAPA found:

- TAPVid-3D PStudio `minival`: 50 NPZ files, 303,409,446 bytes
  (303.4 MB decimal, 0.283 GiB).
- Dynamic3DGaussians six-scene `data.zip`: 589,086,921 bytes
  (589.1 MB decimal, 0.549 GiB).
- Minimum validation download for both: 892,496,367 bytes
  (892.5 MB decimal, 0.831 GiB), excluding extracted-file overhead and the
  CoTracker/DINOv2 HDF5 caches generated later.
- The official split list names 106 PStudio `full_eval` files, but every tested
  object at the `full_eval_v1.0` base URL currently returns HTTP 404; no reliable
  downloadable size can be stated from that route.

Raw CMU Panoptic Studio has no single download size because its official tooling
selects sequences and any subset of up to 480 VGA and 31 HD cameras. A broad
video download should be treated as multi-terabyte scale; a chosen scene/view
subset must be sized separately.

## Source distinction

- Raw CMU Panoptic Studio provides synchronized multi-camera imagery,
  calibration, and tracked body/hand/face keypoints with confidence. This is
  sufficient for a custom sparse skeletal-joint test after projection and
  visibility handling, but not for reproducing arbitrary surface-point tracks.
- Official TAPVid-3D PStudio NPZ annotations provide the point identities,
  time-varying 3D trajectories, query definition, and visibility needed for the
  standard point-tracking benchmark. The associated Dynamic3DGaussians release
  supplies the multi-camera images and calibration used by LAPA's builder.

## Project adapters

- **Open-d4rt — thin conversion.** Export one NPZ per sequence with
  `images_jpeg_bytes[T]`, camera-space metric `tracks_XYZ[T,N,3]`,
  `fx_fy_cx_cy[4]`, and `visibility[T,N]`; static PStudio can omit
  `extrinsics_w2c`. Official TAPVid-3D PStudio is close to this format and mainly
  needs field renaming/normalization.
- **SpaTrackerV2 — loader implementation required.** Its evaluator expects RGB,
  metric depth, 3D trajectories, visibility, per-frame intrinsics, and
  `(t,y,x)` queries. The example imports loader/collate modules absent from this
  checkout and contains internal paths, so NPZ field conversion alone is not
  enough.
- **MV-TAP — multi-view exporter required.** Each scene needs
  `ims/<view>/*.jpg` and `tapvid3d_annotations.npz` containing
  `trajectories_pixelspace[V,T,N,2]`, `per_view_visibilities[V,T,N]`,
  `intrinsics[V,3,3]`, and world-to-camera `extrinsics[V,4,4]`. Its sampling
  script only selects views from an already processed package.
- **LAPA — existing builder.** Its PStudio path consumes TAPVid-3D per-camera
  tracks plus Dynamic3DGaussians `train_meta.json`, transforms tracks into world
  coordinates, builds the three-view layout, and then requires CoTracker/DINOv2
  HDF5 feature caches for training/evaluation.
- **TrackerSplat — reconstruction conversion.** Undistort Panoptic images, keep
  stable camera names, convert calibration to PINHOLE world-to-camera COLMAP
  models, and arrange a fixed camera set under each time frame. A first-frame
  Gaussian initialization is also required. This enables reconstruction but
  does not create 3D point-track ground truth.

## Reusable intermediate representation

For cross-project work, use synchronized image references, `K[V,3,3]`,
`w2c[V,4,4]`, world tracks `[T,N,3]`, per-view visibility `[V,T,N]`, and derived
pixel tracks `[V,T,N,2]` as the conceptual common representation. Implement only
the exporters needed for the current comparison rather than forcing every
project through one physical file format.

## Evidence routes

- `SpaTrackerV2/models/SpaTrackV2/evaluation/example_multithread_eval.py`
- `SpaTrackerV2/models/SpaTrackV2/evaluation/core/evaluator.py`
- `Open-d4rt/eval_track3d_in_worldtrack.py`
- `MV-TAP/data/panoptic.py` and `MV-TAP/scripts/sample_panoptic_views.py`
- `Look-Around-and-Pay-Attention-LAPA-/lapa/data/mc_builder.py`
- `TrackerSplat/trackersplat/dataset/colmap.py` and
  `TrackerSplat/tools/parse_camera.py`

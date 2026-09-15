# MV-TAP

## Role

Multi-view 3D point tracking for datasets including DexYCB, Panoptic, Kubric, and
Harmony4D, with view-sampling utilities.

## Entrypoints

Run evaluation from `MV-TAP/` with explicit inputs and output:

```bash
../.venv/bin/python experiment.py \
  mode=eval \
  ckpt_path=<checkpoint> \
  datasets.data_root=<explicit-data-root> \
  experiment_path=<explicit-output-path>
```

View-sampling helpers are under `scripts/sample_*_views.py`.

## Known 3D query points

MV-TAP does not consume a world-space query directly. Its model input is one
`[query_frame, x, y]` query per view with shape `[B,V,N,3]`. For a known point
`X_world`, project it into every calibrated camera using that query frame's
scaled intrinsic and world-to-camera extrinsic, preserve the point index across
views, and pass the projected pixels to the model. Intrinsics must match the
resized model images; the repository model expects world-to-camera extrinsics.

Projection proves only positive depth and image bounds, not visibility. MV-TAP
has no query-time visibility-mask input and samples the initialization feature
at every supplied query pixel. A projection hidden by another surface therefore
initializes that view from the occluder's appearance; the model's predicted
visibility/confidence is an output and cannot repair the bad query feature in
advance. Determine query visibility with a depth-buffer/raycast test when depth
or geometry exists. If only RGB exists, select a query frame/view manually or
with a separate visibility estimator, and prefer grouping calls by valid camera
subset over feeding a known-occluded projection. Different views may use
different query frames when the point's 3D position at those frames is known.
Known occlusion after initialization is not an inference input either; combine
it with predicted visibility/confidence when deciding which per-frame views may
be triangulated. If query-time visibility differs per point, group points by
their usable camera subset because the current query tensor has no per-point
view-validity field.

MV-TAP returns per-view 2D coordinates, visibility, and confidence. Recover a
3D trajectory afterward by triangulating frames with at least two reliable
views, preferably with reprojection-error rejection rather than raw DLT alone.
The working implementation is `experiments/panoptic_multitracker/
run_qualitative_no_gt.py` (`prepare_multiview`, `run_mvtap`, and
`triangulate_tracks`).

## Harness rules

- Full training (DDP/W&B) and full evaluation are expensive; create a bounded
  single-dataset or small-sample smoke path before using them.
- Pass explicit absolute/normalized dataset and destination paths. Do not assume
  the submodule's missing default dataset directory maps to root `data/`.
- Do not use a sampling `--overwrite` option without explicit authorization; it
  may delete the destination. Prefer symlinks to copying large image trees.
- Avoid W&B or other network side effects unless requested.
- The documented Python/PyTorch versions differ from the root environment;
  preflight imports before running.

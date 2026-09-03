# PStudio temporal-stride pilot

Started: 2026-09-03
Status: completed

Test `IDEA-TEMPORAL-001` first-stage acceptance gate on `juggle_7`: source
strides `1, 2, 3, 4, 6, 8`, fixed 16 model-input frames, one manifest-defined
point set, reference camera 7, companion cameras 8 and 9, real pretrained
weights, and a common offline TAPVid-3D evaluator. Required result is a valid
prediction from all four pipelines at every stride plus accuracy curves and
bounded per-run evidence under `output/panoptic_multitracker/temporal_stride/`.

Baseline submodules are SpaTrackerV2 `7e12274`, Open-d4rt `403290a`, MV-TAP
`b248aea`, and LAPA `30012d6`, all clean. Experiment-only code belongs under
`experiments/panoptic_multitracker/`; no submodule source change is planned.

## Result

All 24 real model forwards completed. The manifest contains 64 fixed reference
points and raw frame endpoints `15, 30, 45, 60, 90, 120`. Each model emitted
16-frame predictions for all six strides. Open-d4rt, MV-TAP, and LAPA loaded
with zero missing/unexpected checkpoint keys. Every LAPA run used three
stride-specific CoTracker caches; each cache had `use_cotracker=1` and exactly
the expected 93.75% non-GT coordinate fraction after its one GT query frame.

Common evaluation transforms predictions to reference camera 7, estimates one
scale from the query frame, holds it fixed, excludes that frame from scoring,
and applies the same TAPVid-3D metrics.

| Model | AJ3D stride 1 | AJ3D stride 8 | Retention | EPE stride 1→8 (m) |
| --- | ---: | ---: | ---: | ---: |
| Open-d4rt | 0.2253 | 0.1525 | 0.677 | 0.0788 → 0.1076 |
| SpaTrackerV2 | 0.0899 | 0.1627 | 1.810 | 0.1185 → 0.0991 |
| MV-TAP | 0.7247 | 0.3773 | 0.521 | 0.0138 → 0.0396 |
| LAPA Joint | 0.4354 | 0.2803 | 0.644 | 0.0379 → 0.0806 |

Open-d4rt, MV-TAP, and LAPA show overall degradation as the source-frame gap
grows. SpaTrackerV2 is non-monotonic and improves at stride 8 on this clip;
because fixed input count also changes the sampled motion phase and total time
span, this is an observation rather than evidence of general improvement. The
six-scene stage is needed before making a model-level conclusion.

## Evidence

- Manifest: `output/panoptic_multitracker/temporal_stride/manifest.json`
- Metrics: `output/panoptic_multitracker/temporal_stride/common_metrics.csv`
- Absolute curves: `output/panoptic_multitracker/temporal_stride/accuracy_vs_stride.png`
- AJ retention: `output/panoptic_multitracker/temporal_stride/aj3d_retention_vs_stride.png`
- Raw predictions and bounded logs: one `results/<model>/` directory below
  each `output/panoptic_multitracker/temporal_stride/stride_<n>/`.

The pilot also exposed a reusable LAPA integration requirement: the public
evaluator's `feature_dir` argument does not populate the dataset's separate
`eval_feature_dir`. The experiment runner now binds both and validates cache
contents, replacing the earlier optimistic LAPA metric with a real-cache result.

## Next

Repeat the six strides on one reference camera from each of the six actions,
aggregate by scene, and only then decide whether the 50-reference run is worth
the compute cost.

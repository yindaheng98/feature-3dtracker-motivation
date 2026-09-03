# IDEA-TEMPORAL-001: PStudio temporal-stride robustness curves

Status: supported by one-sample pilot; multi-scene evaluation remains open
Created: 2026-09-03

## Question

How quickly do SpaTrackerV2, Open-d4rt, MV-TAP, and LAPA lose 3D point-tracking
accuracy as the interval between observed PStudio frames increases?

## Primary protocol: fixed model-frame count

- Use 16 model-input frames and source strides `1, 2, 3, 4, 6, 8`:
  `raw_idx = start + stride * arange(16)`. All 150-frame clips support this at
  `start=0`; stride 8 spans source frames 0 through 120.
- This keeps SpaTrackerV2 in one window, MV-TAP at its native 16-frame window,
  and Open-d4rt below its 32-frame clip limit. It measures increasing motion and
  time span without changing model-input length or execution branch.
- Select a deterministic set of at most 64 reference-camera points using only
  eligibility at the sampled query frame. Store exact frame, point, reference
  camera, and companion-camera IDs in a manifest and reuse them for every model
  and stride. Do not select using future visibility.
- For MV-TAP/LAPA use a fixed calibrated three-camera group containing the
  reference camera. Mono and multi-view runs must use the same reference GT
  tracks and sampled times.

## Common evaluation

- Convert every prediction into the reference-camera coordinate frame and use
  one shared TAPVid-3D evaluator. Exclude the query frame.
- Align monocular scale once from query-frame points, then hold that scale for
  later frames; do not fit scale using future GT. Apply the same operation to
  all models for the common curve and separately report unaligned metric error
  for calibrated MV-TAP/LAPA.
- Primary curve: common 3D average Jaccard versus source-frame stride. Also plot
  APD, OA, EPE, prediction coverage, and AJ retention relative to stride 1.
  Native project metrics are diagnostics, not the cross-model comparison.
- Aggregate cameras within each scene first and then macro-average the six
  scenes. Show all per-scene curves. The 50 camera-reference samples are
  correlated views of six motions and must not be treated as 50 independent
  sequences. If intervals are shown, use paired scene-level bootstrap and state
  the small six-scene sample size.

## Secondary protocol: fixed physical horizon

Use source interval `[start, start + 120]` and sample it at each stride. Input
length then falls from 121 to 16 frames. Evaluate on the same raw target frames
`24, 48, 72, 96, 120` relative to `start`, which are present in every stride
grid. Keep this result separate: it measures sparse observation over a matched
time horizon and intentionally activates different long-sequence branches.

## Pipeline details

- Slice image bytes, XYZ GT, visibility, and any per-frame camera arrays with
  the same `raw_idx` before model-specific preprocessing. Query time becomes 0
  in the sampled clip.
- SpaTrackerV2 must rerun both Front and Offline on each sampled clip.
- Open-d4rt needs explicit `start/frame_stride` indexing before its loader
  decodes and projects frames; prefix-only `num_frames` is insufficient.
- MV-TAP's existing model `stride=4` is spatial, not temporal. Add an explicitly
  named temporal stride in the experiment adapter.
- End-to-end LAPA must rerun CoTracker on the sparsely sampled video for every
  stride. DINO features are frame-local and may be computed once then indexed.
  Subsampling dense-video CoTracker results is a separate fusion-only ablation,
  not the primary end-to-end result.

## Execution stages

1. `juggle_7`, all six strides: verify indexing, output schemas, non-GT LAPA
   caches, and one curve per metric.
2. One reference camera per action: six-motion trend check and runtime measure.
3. All 50 reference samples only after the first two gates pass. Cache raw
   predictions and recompute metrics/plots offline without rerunning models.

The previous bring-up results cannot be plotted together: they used different
frame counts, point counts, metric implementations, and scale rules.

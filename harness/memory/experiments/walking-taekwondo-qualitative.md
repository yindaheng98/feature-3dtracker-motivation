# Walking/taekwondo qualitative tracking

Started: 2026-09-06
Status: completed

Run Open-d4rt, SpaTrackerV2, MV-TAP, and LAPA on the unannotated `walking` and
`taekwondo` sequences and render predicted trajectories without inventing a
ground-truth metric. The common protocol uses 32 frames sampled uniformly over
each complete sequence, 32 shared frame-1 surface queries, reference camera
`0.png`, and companion cameras `1.png` and `2.png`. The wider temporal sampling
is intentional: the walking action begins late in its 75-frame source. Trails
therefore connect sampled source frames and must not be read as consecutive-frame
motion or a temporal-stride comparison.

The query initializer selects Shi–Tomasi corners in non-border temporal-motion
components, converts reference inverse depth to world points with the COLMAP
camera, and requires positive, in-bounds projections in all three views. A
three-camera initialization sheet was visually checked before inference. The
four models use the same point identities; LAPA's normal CoTracker and DINO
inputs are computed from the sampled videos and are not trajectory labels.

## Result

All eight real-checkpoint forwards completed. Every saved prediction has shape
`32 frames × 32 points × 2`, contains 100% finite coordinates, and every point
marked visible is in the native reference image. Visibility and frame-1 anchor
diagnostics are descriptive runner checks, not accuracy measurements:

| Scene | Model | Visible | Median frame-1 anchor error |
| --- | --- | ---: | ---: |
| walking | Open-d4rt | 92.8% | 5.19 px |
| walking | SpaTrackerV2 | 85.7% | 0.27 px |
| walking | MV-TAP | 92.2% | 0.23 px |
| walking | LAPA | 99.4% | 25.51 px |
| taekwondo | Open-d4rt | 91.8% | 3.77 px |
| taekwondo | SpaTrackerV2 | 66.7% | 0.42 px |
| taekwondo | MV-TAP | 85.9% | 0.35 px |
| taekwondo | LAPA | 76.7% | 11.02 px |

Visual inspection shows that Open-d4rt, SpaTrackerV2, and MV-TAP mostly keep
their rendered tracks attached to the actors. LAPA has a larger first-frame
offset and visibly longer drift, especially on walking. With no trajectory
annotations this is a qualitative observation only, not a model ranking. The
initialization sheets should be inspected alongside the animations when judging
whether the shared points correspond to the intended surfaces.

## Evidence and code

- Walking animation and static summaries:
  `output/panoptic_multitracker/qualitative_no_gt/walking/`
- Taekwondo animation and static summaries:
  `output/panoptic_multitracker/qualitative_no_gt/taekwondo/`
- Per-model bounded metadata: each scene's `predictions/*.json`
- Shared preparation and model adapters:
  `experiments/panoptic_multitracker/prepare_qualitative_no_gt.py` and
  `experiments/panoptic_multitracker/run_qualitative_no_gt.py`
- Renderer and reproducible commands:
  `experiments/panoptic_multitracker/render_qualitative_no_gt.py` and
  `experiments/panoptic_multitracker/README.md`

No submodule source or environment dependency was changed. The root contains
only the new experiment adapters, documentation, and this canonical memory;
all five submodules remain clean.

## Next

If these sequences are used for a model comparison rather than a visual demo,
first add independent annotations or a calibrated consistency protocol. Also
investigate LAPA's query-frame offset before interpreting its later drift.

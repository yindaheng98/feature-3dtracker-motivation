# Experiment Memory Index

Add compact routes only when an experiment produces reusable evidence. Detail
file names, content shape, outcome wording, and raw output layout are chosen for
the actual work.

| Date/topic | Reusable conclusion | Detail |
| --- | --- | --- |
| 2026-09-15 · walking/taekwondo 64-frame memory trace | Eight 64-frame forwards and GIFs completed; Spa Front peaks near 17 GiB allocated, LAPA peaks in CoTracker, and Open-d4rt uses overlapping 32-frame anchor clips. | [result](walking-taekwondo-64f-memory.md) |
| 2026-09-06 · walking/taekwondo qualitative tracking | Four real pretrained trackers produced 32-frame/32-point trajectories for both unannotated scenes; Open/Spa/MV mostly stay on actors, while LAPA shows a larger query-frame offset and more visible drift. | [result](walking-taekwondo-qualitative.md) |
| 2026-09-15 · extended PStudio inference scaling | Frame scaling reaches 150/OOM, while a separate 221-point single-window sweep finds subquadratic point exponents (Open/Spa/MV/LAPA: 0.60/0.02/0.17/0.01); forcing a quadratic wall-time claim is unsupported. | [result](panoptic-inference-scaling.md) |
| 2026-09-03 · PStudio temporal-stride pilot | All 24 forwards completed and strides 1/4/8 have reference-camera GIF/contact-sheet renderings; three models degrade overall, while SpaTrackerV2 is non-monotonic on the single clip. | [result](panoptic-temporal-stride-pilot.md) |
| 2026-09-03 · Panoptic four-project bring-up | Four real pretrained pipelines completed on shared `juggle_7`; Open-d4rt and Spa report 3D tracking, MV-TAP adds calibrated triangulation, and LAPA uses real CoTracker+DINO features. | [result](panoptic-four-project-bringup.md) |
| 2026-09-02 · environment setup and compatibility | Earlier setup checks were consolidated after their reusable evidence and decisions were merged. | [environment guide](../../dependencies/native-prerequisites.md) |

When this index grows past 20 KiB, move older routes into a topical or monthly
archive and retain only the useful directory entries here.

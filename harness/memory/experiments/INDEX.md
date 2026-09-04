# Experiment Memory Index

Add compact routes only when an experiment produces reusable evidence. Detail
file names, content shape, outcome wording, and raw output layout are chosen for
the actual work.

| Date/topic | Reusable conclusion | Detail |
| --- | --- | --- |
| 2026-09-04 · PStudio inference scaling | Frame count dominates all four runtimes; Open-d4rt grows 2.52× from 8→64 points, while Spa/LAPA are flat and MV-TAP grows 1.19× because fixed support/virtual work dominates. | [result](panoptic-inference-scaling.md) |
| 2026-09-03 · PStudio temporal-stride pilot | All 24 forwards completed and strides 1/4/8 have reference-camera GIF/contact-sheet renderings; three models degrade overall, while SpaTrackerV2 is non-monotonic on the single clip. | [result](panoptic-temporal-stride-pilot.md) |
| 2026-09-03 · Panoptic four-project bring-up | Four real pretrained pipelines completed on shared `juggle_7`; Open-d4rt and Spa report 3D tracking, MV-TAP adds calibrated triangulation, and LAPA uses real CoTracker+DINO features. | [result](panoptic-four-project-bringup.md) |
| 2026-09-02 · environment setup and compatibility | Earlier setup checks were consolidated after their reusable evidence and decisions were merged. | [environment guide](../../dependencies/native-prerequisites.md) |

When this index grows past 20 KiB, move older routes into a topical or monthly
archive and retain only the useful directory entries here.

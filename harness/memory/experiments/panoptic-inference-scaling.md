# PStudio inference scaling benchmark

Started: 2026-09-03
Completed: 2026-09-04
Status: completed

Measure how tracked-point count and input-frame count affect inference latency
for the four PStudio-compatible trackers on one idle RTX A5000 (physical GPU
1). Use `juggle_7`, nested input prefixes, points 8/16/32/64 at 16 frames, and
frames 8/16/32 at 32 points. Each unique shape receives one warm-up followed by
three wall-clock measurements with CUDA synchronization; Spa's two stages are
measured separately and their summary statistics are added. Checkpoint loading,
disk I/O, metrics, and saving are outside the timed region.

## Results

Median seconds versus tracked points at 16 frames:

| Model | 8 | 16 | 32 | 64 | 8→64 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open-d4rt | 0.530 | 0.644 | 0.872 | 1.334 | 2.52× |
| SpaTrackerV2 | 1.350 | 1.346 | 1.322 | 1.350 | 1.00× |
| MV-TAP | 0.162 | 0.161 | 0.169 | 0.193 | 1.19× |
| LAPA | 0.243 | 0.232 | 0.225 | 0.237 | 0.97× |

Median seconds versus input frames at 32 points:

| Model | 8 | 16 | 32 | 8→32 ratio |
| --- | ---: | ---: | ---: | ---: |
| Open-d4rt | 0.348 | 0.872 | 2.500 | 7.19× |
| SpaTrackerV2 | 0.740 | 1.322 | 2.794 | 3.78× |
| MV-TAP | 0.111 | 0.169 | 0.409 | 3.67× |
| LAPA | 0.107 | 0.225 | 0.479 | 4.46× |

Open-d4rt has the clearest point-count scaling because queries are decoded in
chunks of 32. SpaTrackerV2 is effectively flat across 8–64 requested points:
its 16-frame Front stage is about 1.00 s and its tracker about 0.33–0.35 s,
while the tracker retains 64 internal support points. MV-TAP also retains 64
virtual tracks, so requested points have a modest effect. LAPA latency is flat
within measurement noise over this range, although its incremental allocated
memory rises from about 17 MiB at 8 points to 76 MiB at 64 points.

Frame count is the dominant cost for every model. SpaTrackerV2, MV-TAP, and
LAPA are near the expected fourfold increase from 8 to 32 frames. Open-d4rt is
more strongly superlinear at 7.19×. Peak allocated memory from 8 to 32 frames
rose from about 4.63 to 4.94 GiB for Open-d4rt, 8.71 to 12.36 GiB for Spa,
2.30 to 2.67 GiB for MV-TAP, and 42 to 97 MiB for LAPA's compact cached-feature
head.

One SpaTrackerV2 16-point tracker-stage sample was an outlier (0.845 s versus
0.323/0.348 s), so its plotted stage-summed quartile range is wide; retaining
the sample does not change the flat median trend.

## Interpretation boundary

These timings support within-model scaling conclusions, not an absolute speed
leaderboard. Open-d4rt's helper includes its internal transfers and CPU result
conversion; Spa reports Front plus Offline Tracker; MV-TAP times the three-view
neural forward but not triangulation; LAPA times only its neural forward over
precomputed CoTracker/DINO features. Internal support/virtual points and image
resolutions also differ.

## Evidence

- Full samples and medians:
  `output/panoptic_multitracker/inference_scaling/results.json`
- Flat table: `output/panoptic_multitracker/inference_scaling/results.csv`
- Point curve:
  `output/panoptic_multitracker/inference_scaling/inference_time_vs_points.png`
- Frame curve:
  `output/panoptic_multitracker/inference_scaling/inference_time_vs_frames.png`
- Reproducible commands and per-shape logs:
  `output/panoptic_multitracker/inference_scaling/commands.json` and `raw/`

The initial 8-frame/8-point compatibility smoke outputs remain under the
separate `inference_scaling/smoke/` directory and are not part of the formal
tables.

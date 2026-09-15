# PStudio inference scaling benchmark

Started: 2026-09-03
Extended: 2026-09-15
Status: completed

Measure how tracked-point count and input-frame count affect inference latency
for the four PStudio-compatible trackers on one idle RTX A5000 (physical GPU
1). Use `juggle_7`, nested input prefixes, points 8/16/32/64 at 16 frames, and
frames 8/16/32/48/64/80/96/112/128/150 at 32 points. The later point extension
uses 8/16/32/64/96/128/160/192/221 points at 16 frames. Each successful unique
shape receives one warm-up followed by three wall-clock measurements with CUDA
synchronization; Spa's two stages are measured separately and their summary
statistics are added. Checkpoint loading, disk I/O, metrics, and saving are
outside the timed region. A model's first real CUDA OOM is retained as a
signature-bound result, and unchanged larger configurations are not attempted.

## Results

Median seconds versus tracked points at 16 frames:

| Model | 8 | 64 | 128 | 221 | 8→221 ratio | Empirical p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open-d4rt | 0.524 | 1.340 | 2.269 | 3.630 | 6.93× | 0.60 |
| SpaTrackerV2 | 1.316 | 1.366 | 1.356 | 1.393 | 1.06× | 0.02 |
| MV-TAP | 0.162 | 0.190 | 0.227 | 0.280 | 1.73× | 0.17 |
| LAPA | 0.229 | 0.237 | 0.250 | 0.235 | 1.03× | 0.01 |

Extended fixed-32-point frame sweep:

| Model | Largest success | Time, 8→largest | Peak allocated, 8→largest | Global empirical p | First OOM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open-d4rt | 150 | 0.348→113.616 s | 4.52→4.95 GiB | 2.16 | none |
| SpaTrackerV2 | 80 | 0.740→9.620 s | 8.50→19.67 GiB | 1.12 | 96 frames |
| MV-TAP | 150 | 0.111→2.221 s | 2.25→8.80 GiB | 1.06 | none |
| LAPA | 150 | 0.107→2.089 s | 0.04→0.36 GiB | 1.01 | none |

Open-d4rt has the clearest point-count scaling because queries are decoded in
independent chunks of 32, but even its high-range fit is sublinear rather than
quadratic. SpaTrackerV2 is effectively flat through 221 requested points because
the 16-frame Front cost is fixed and the tracker uses 64 internal support
points. MV-TAP also uses 64 virtual tracks, so requested points have a modest
effect. LAPA time stays flat within measurement noise while its peak allocation
grows from 0.03 to 0.29 GiB. At 16 frames these runs stay inside Open-d4rt's
32-frame clip, MV-TAP's single 16-frame window, and SpaTrackerV2's 200-frame
outer window. Thus crossing a window is not hiding a point-wise quadratic
trend; this implementation does not have one.

A bounded follow-up isolated Open-d4rt inside its 32-frame clip. The whole
encoder measured about `p=1.10` over 8–32 frames; its real global-attention
operator measured about `p=1.38` over 512–4096 real video tokens. Although the
attention arithmetic contains a quadratic term, GPU utilization and the rest
of the operator prevent wall time from following `T²` over this supported
range. These observations reject, rather than support, a claim of measured
quadratic wall time.

Frame count dominates cost. A global log-log fit gives Open-d4rt `p=2.16`, close
to the plotted quadratic reference. This is an empirical input-range fit, not a
claim that its core is full-sequence quadratic attention: the 32-frame checkpoint
switches to repeated anchored clips above 32 frames, creating a regime change;
the 48–150 segment alone fits about `p=1.58`. MV-TAP (`p=1.06`) and LAPA
(`p=1.01`) remain approximately linear. SpaTrackerV2 reaches 19.67 GiB allocated
at 80 frames, then its Front depth head fails at 96 while attempting another
596 MiB: the process already uses 23.19 of the 23.56 GiB GPU capacity. Although
Spa's Front contains global attention, measured end-to-end successful points fit
`p=1.12` globally (`p≈1.41` over 48–80), not quadratic in this tested range.

In the point-count sweep, absolute peak allocation is flat for Open-d4rt
(~4.62 GiB), SpaTrackerV2 (~9.70 GiB), and MV-TAP (~2.26 GiB); LAPA grows from
about 0.03 to 0.10 GiB. Each memory plot also shows the incremental peak above
the pre-forward allocation as a dashed line. The two curves are alternative
views of the same peak and are not additive.

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
- Peak-memory versus point curve:
  `output/panoptic_multitracker/inference_scaling/peak_memory_vs_points.png`
- Peak-memory versus frame curve:
  `output/panoptic_multitracker/inference_scaling/peak_memory_vs_frames.png`
- Reproducible commands and per-shape logs:
  `output/panoptic_multitracker/inference_scaling/commands.json` and `raw/`
- OOM and skipped configurations:
  `output/panoptic_multitracker/inference_scaling/failures.json`
- Empirical frame exponents:
  `output/panoptic_multitracker/inference_scaling/frame_scaling_fits.json`
- Extended 221-point samples, fits, and figures:
  `output/panoptic_multitracker/point_scaling_extended/`
- Open-d4rt single-clip encoder/attention diagnostic:
  `output/panoptic_multitracker/point_scaling_extended/within_clip_attention_pilot.json`
  and `within_clip_attention_pilot.png`
- Separate real-CoTracker LAPA cache used by the extended point sweep:
  `output/panoptic_multitracker/lapa/features_eval_221/`

The initial 8-frame/8-point compatibility smoke outputs remain under the
separate `inference_scaling/smoke/` directory and are not part of the formal
tables.

The memory curves use `torch.cuda.max_memory_allocated`: they exclude reserved
allocator capacity and non-PyTorch process memory. Spa's two stage peaks are
combined with `max`, not summed, because the Front is released before Tracker.

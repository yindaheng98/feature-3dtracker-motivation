# Walking/taekwondo 64-frame tracks and CUDA memory traces

Started: 2026-09-15
Status: completed

Extend the unannotated walking/taekwondo qualitative test from 32 to 64
uniformly sampled frames while retaining 32 shared frame-1 surface queries, then
record how PyTorch CUDA allocator usage changes during each complete model run.
The 32-frame baseline remains unchanged under `qualitative_no_gt/`; this run has
an independent `qualitative_no_gt_64f/` output root.

The memory sampler runs in a background thread every 50 ms without CUDA
synchronization. It starts before model loading and stops when the model adapter
returns, so the x-axis is process wall time and includes RGB loading, model
setup, and preprocessing. `allocated` means active tensor allocation and
`reserved` means capacity held by PyTorch's caching allocator; neither is total
process memory from `nvidia-smi`. Model-stage events identify which operation
caused each change.

## Result

All eight real-checkpoint runs completed without OOM. Every prediction is
`64×32×2`, all coordinates are finite, every point marked visible lies inside
the native reference image, both GIFs contain exactly 64 frames, and their last
source frames are walking 75 and taekwondo 101.

| Scene | Model | Visible | Median anchor | Peak allocated | Peak reserved |
| --- | --- | ---: | ---: | ---: | ---: |
| walking | Open-d4rt | 93.2% | 3.10 px | 4.89 GiB | 5.25 GiB |
| walking | SpaTrackerV2 | 85.7% | 0.25 px | 16.42 GiB | 18.98 GiB |
| walking | MV-TAP | 92.5% | 0.25 px | 3.48 GiB | 5.46 GiB |
| walking | LAPA | 99.6% | 25.48 px | 5.69 GiB | 7.15 GiB |
| taekwondo | Open-d4rt | 93.4% | 3.75 px | 4.89 GiB | 5.25 GiB |
| taekwondo | SpaTrackerV2 | 52.9% | 0.50 px | 16.96 GiB | 19.50 GiB |
| taekwondo | MV-TAP | 85.6% | 0.21 px | 3.48 GiB | 5.46 GiB |
| taekwondo | LAPA | 72.3% | 11.13 px | 5.69 GiB | 7.15 GiB |

The curves explain the peaks: SpaTrackerV2 ramps to 16–17 GiB in its Front
stage, releases that state, then allocates roughly 4 GiB for Offline Tracker.
LAPA's 5.69 GiB allocated peak belongs to its CoTracker input-generation stage;
DINO and the LAPA head are lower. MV-TAP grows during its multi-window tracking
forward. Open-d4rt holds about 4.5 GiB active while its published anchor-clip
path processes the sequence; its checkpoint is trained for 32-frame clips, so
64 output frames are assembled from overlapping anchored clips rather than one
64-frame forward.

The denser renderings preserve the earlier qualitative pattern: Open-d4rt,
SpaTrackerV2, and MV-TAP mostly remain attached to the actors, while LAPA keeps
the same large initial offset and shows more visible drift. These sequences
still have no trajectory GT, so visibility, anchors, and renderings are
diagnostics rather than accuracy scores.

## Evidence and code

- Walking animation, contact sheet, memory graph, summaries, traces, and raw
  predictions: `output/panoptic_multitracker/qualitative_no_gt_64f/walking/`
- Taekwondo equivalents:
  `output/panoptic_multitracker/qualitative_no_gt_64f/taekwondo/`
- Memory sampler and model-stage markers:
  `experiments/panoptic_multitracker/run_qualitative_no_gt.py`
- Plotter: `experiments/panoptic_multitracker/plot_qualitative_memory.py`
- Reproduction command: `experiments/panoptic_multitracker/README.md`

No dependency or submodule source changed. The initial sampler smoke stopped
before model loading because this PyTorch build requires an initialized CUDA
context before resetting peak stats; initializing CUDA explicitly resolved it,
and no partial prediction was retained.

## Next

If memory scaling versus input length is needed, run independent processes at
fixed counts such as 8/16/32/48/64 and plot peak deltas versus frame count. The
current graph instead answers how memory changes over wall time within the
validated 64-frame end-to-end run.

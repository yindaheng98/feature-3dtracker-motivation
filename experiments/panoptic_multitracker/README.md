# Panoptic multi-tracker bring-up

This directory contains the smallest shared adapters needed to exercise
SpaTrackerV2, Open-d4rt, MV-TAP, and LAPA on the same TAPVid-3D Panoptic Studio
source. It intentionally does not impose one common on-disk format on the four
projects; adapters are added only where the real loaders require them.

Input data lives under `data/panoptic_tracking/`. Generated metrics and other
outputs live under `output/panoptic_multitracker/` in forms chosen for each
project.

Acceptance target: for at least one common PStudio scene, each project loads its
pretrained checkpoint, performs a real-data forward pass, and emits its native
meaningful metric or tracking result. Import-only tests do not count.

Prepare the shared source data from the repository root:

```bash
.venv/bin/python experiments/panoptic_multitracker/prepare_data.py
```

The validated sample is `juggle_7`. Commands below use GPU 1; change
`CUDA_VISIBLE_DEVICES` or `--device` together when needed.

## Open-d4rt

The official minival NPZ already matches Open-d4rt's WorldTrack input contract,
so only a one-file subset directory is needed:

```bash
mkdir -p output/panoptic_multitracker/opend4rt/input/pstudio_mini
ln -sfn ../../../../../data/panoptic_tracking/tapvid3d_minival/tap3d_juggle/juggle_7.npz \
  output/panoptic_multitracker/opend4rt/input/pstudio_mini/juggle_7.npz
cd Open-d4rt
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python eval_track3d_in_worldtrack.py \
  --model-config ../checkpoints/huggingface/hub/models--Lijiaxin0111--OpenD4RT/snapshots/7099b1fc760475de3b7409acd6b63f801b015d07/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --ckpt-path ../checkpoints/huggingface/hub/models--Lijiaxin0111--OpenD4RT/snapshots/7099b1fc760475de3b7409acd6b63f801b015d07/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --data-root ../output/panoptic_multitracker/opend4rt/input \
  --subsets pstudio_mini --limit-seqs 1 --num-frames 32 \
  --query-chunk-size 32 --device cuda --save-per-sequence \
  --output-dir ../output/panoptic_multitracker/opend4rt/eval_juggle7_32f
cd ..
```

## SpaTrackerV2

This adapter runs the published Front model to infer depth and intrinsics, then
passes them to the Offline tracker. It does not use GT depth.

```bash
HF_HOME="$PWD/checkpoints/huggingface" HF_HUB_OFFLINE=1 \
MPLCONFIGDIR="$PWD/output/panoptic_multitracker/matplotlib" \
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  experiments/panoptic_multitracker/run_spatracker.py \
  data/panoptic_tracking/tapvid3d_minival/tap3d_juggle/juggle_7.npz \
  --output output/panoptic_multitracker/spatracker/juggle7_16f_metrics.json \
  --device cuda:0 --frames 16 --max-points 32 --support-points 64
```

## LAPA

Build the calibrated multi-camera representation, then compute real CoTracker
and DINOv2 features before evaluating the Joint checkpoint. The one-line split
file in `lapa_juggle7/` keeps this validation bounded to one reference camera.

```bash
cd Look-Around-and-Pay-Attention-LAPA-
PYTHONPATH="$PWD" ../.venv/bin/python -m lapa.data.mc_builder \
  --npz_root ../data/panoptic_tracking/tapvid3d_minival \
  --d3g_root ../data/panoptic_tracking/d3g/data \
  --out_dir ../output/panoptic_multitracker/lapa/mc --scenes juggle
HF_HOME="../checkpoints/huggingface" TORCH_HOME="../checkpoints/torch" \
HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" ../.venv/bin/python \
  -m lapa.features.precompute_canonical --mode eval \
  --mc_dir ../output/panoptic_multitracker/lapa/mc \
  --out_dir ../output/panoptic_multitracker/lapa/features_eval \
  --data_root ../experiments/panoptic_multitracker/lapa_juggle7 \
  --device cuda:1 --use_cotracker --max_points 64
HF_HOME="../checkpoints/huggingface" HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" \
../.venv/bin/python evaluate_lapa.py \
  --checkpoint ../checkpoints/huggingface/hub/models--bishoygaloaa--LAPA-Joint/snapshots/0cce9285a629fb05a843edd690d38ca4107de177/lapa.pt \
  --mc_dir ../output/panoptic_multitracker/lapa/mc \
  --feature_dir ../output/panoptic_multitracker/lapa/features_eval \
  --data_root ../experiments/panoptic_multitracker/lapa_juggle7 \
  --device cuda:1 --num_views 3 --max_points 64 \
  --output ../output/panoptic_multitracker/lapa/eval_juggle7_metrics.json
cd ..
```

## MV-TAP

MV-TAP predicts calibrated 2D tracks in three views. The adapter additionally
triangulates them with the same PStudio cameras to report a 3D error.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  experiments/panoptic_multitracker/run_mvtap.py \
  --mc-dir output/panoptic_multitracker/lapa/mc \
  --npz-root data/panoptic_tracking/tapvid3d_minival \
  --checkpoint checkpoints/MVTAP.ckpt \
  --output output/panoptic_multitracker/mvtap/juggle7_16f_metrics.json \
  --scene juggle --cameras 7 8 9 --frames 16 --max-points 32 --device cuda:0
```

## Validated results

| Project | Input | Headline result |
| --- | --- | --- |
| Open-d4rt | 32 frames, 221 queries | APD 0.9792; EPE 0.0581 m |
| SpaTrackerV2 | 16 frames, 32 queries | OA 0.9844; scaled EPE 0.1180 m |
| MV-TAP | 3 views, 16 frames, 32 queries | 2D AJ 0.8885; triangulated MPJPE 0.01424 m |
| LAPA Joint | 3 views, 150 frames, 64 queries | 3D-AJ 43.17; MPJPE 0.02620 m |

All four runs loaded their real pretrained weights. Open-d4rt and MV-TAP had
zero missing and zero unexpected checkpoint keys. LAPA's three feature caches
have `use_cotracker=1`; 99.33% of their coordinates differ from GT, so the
reported run did not take the silent GT fallback path.

# Model checkpoint routing and commands

This is the canonical checkpoint guide. Use the root `checkpoints/` overlay for
all pretrained inputs and keep project source trees free of model binaries.

## Storage policy

Route files by acquisition mechanism:

```text
checkpoints/
  huggingface/       # HF_HOME: every model obtained from Hugging Face
  torch/             # TORCH_HOME: Torch Hub repositories and downloads
  MVTAP.ckpt         # direct Google Drive file
  *.pth, *.pt, *.npz # direct TrackerSplat/DOT/initializer files
```

Hugging Face's `hub/models--ORG--NAME/snapshots/<revision>/` hierarchy is the
model's local path; project-named copies beside that cache are unnecessary.
Direct non-Hub files stay flat at the checkpoint root unless an upstream loader
requires a different layout.

Set these variables from the repository root for every download or run:

```bash
export HF_HOME="$PWD/checkpoints/huggingface"
export TORCH_HOME="$PWD/checkpoints/torch"
```

The container's default Home is read-only. Hub tools and project code must use
the paths above. After downloads complete, `HF_HUB_OFFLINE=1` makes missing
cache entries fail immediately instead of accessing the network.

## Minimal download commands

The currently selected minimum set can be reproduced with:

```bash
# SpaTrackerV2
.venv/bin/hf download Yuxihenry/SpatialTrackerV2_Front
.venv/bin/hf download Yuxihenry/SpatialTrackerV2-Offline

# Open-d4rt 32-frame
.venv/bin/hf download Lijiaxin0111/OpenD4RT \
  --include 'checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/*'

# MV-TAP
.venv/bin/gdown --no-cookies 1sCml0BL6VQGy-MGgpidz2-BdymAJhboU \
  -O checkpoints/MVTAP.ckpt

# LAPA Joint and its feature backbone
.venv/bin/hf download bishoygaloaa/LAPA-Joint lapa.pt
.venv/bin/hf download facebook/dinov2-base
```

`gdown --no-cookies` is required here because its cookie path otherwise points
into the read-only Home. Hugging Face may transparently retry temporary DNS
failures; an incomplete download is never considered validation.

## How each project loads models

### SpaTrackerV2

`SpaTrackerV2/inference.py` hard-codes Hugging Face repository IDs. With
`HF_HOME` set, `from_pretrained` resolves them from the shared cache. The front
model is always loaded; `--track_mode offline` selects the Offline tracker.

```bash
cd SpaTrackerV2
HF_HUB_OFFLINE=1 ../.venv/bin/python inference.py \
  --track_mode offline --data_type RGBD \
  --data_dir <data-dir> --video_name <video-name>
```

The Online tracker is optional:

```bash
.venv/bin/hf download Yuxihenry/SpatialTrackerV2-Online
```

`moge_as_base=True` additionally uses `Ruicheng/moge-vitl`. The optional web
demo has separate SAM requirements; neither is needed by the validated minimum
offline loader.

### Open-d4rt

Open-d4rt accepts explicit `--model-config` and `--ckpt-path` values. Re-running
the cached download command with offline mode returns its snapshot directory:

```bash
OPEN_SNAPSHOT="$(HF_HUB_OFFLINE=1 .venv/bin/hf download Lijiaxin0111/OpenD4RT \
  --include 'checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/*')"
OPEN_MODEL="$OPEN_SNAPSHOT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG"

cd Open-d4rt
../.venv/bin/python eval_track3d_in_worldtrack.py \
  --model-config "$OPEN_MODEL/model.yaml" \
  --ckpt-path "$OPEN_MODEL/opend4rt.ckpt" \
  <bounded-data-and-output-args>
```

The 48-frame checkpoint and training-only VideoMAEv2 initialization are not
part of the current minimum set.

### MV-TAP

MV-TAP restores the flat checkpoint through Lightning's `ckpt_path`:

```bash
cd MV-TAP
../.venv/bin/python experiment.py \
  mode=eval ckpt_path="../checkpoints/MVTAP.ckpt"
```

### LAPA

LAPA accepts an explicit `--checkpoint`. Resolve the cached Hub file with the
same short command used to download it:

```bash
LAPA_CKPT="$(HF_HUB_OFFLINE=1 .venv/bin/hf download \
  bishoygaloaa/LAPA-Joint lapa.pt)"

cd Look-Around-and-Pay-Attention-LAPA-
../.venv/bin/python inference_lapa.py \
  --checkpoint "$LAPA_CKPT" <scene-feature-output-args>
```

Feature precomputation calls
`AutoModel.from_pretrained("facebook/dinov2-base")`, which uses `HF_HOME`.
Optional `--use_cotracker` calls Torch Hub and therefore uses `TORCH_HOME`.

### TrackerSplat

TrackerSplat's published dependencies are direct flat files. Its default DOT
estimator accepts explicit paths:

```bash
-o "tracker_path='../checkpoints/movi_f_cotracker2_patch_4_wind_8.pth'" \
-o "estimator_path='../checkpoints/cvo_raft_patch_8.pth'" \
-o "refiner_path='../checkpoints/movi_f_raft_patch_4_alpha.pth'"
```

For loaders that insist on `TrackerSplat/checkpoints`, an ignored compatibility
link may point to the root overlay:

```bash
test ! -e TrackerSplat/checkpoints
ln -s ../checkpoints TrackerSplat/checkpoints
```

## Validated inventory

Validated on 2026-09-03:

- TrackerSplat: all 14 weight filenames published in its README are present.
- SpaTrackerV2 Front and Offline load from `HF_HOME` with
  `HF_HUB_OFFLINE=1`.
- Open-d4rt 32-frame loads from its Hub snapshot with zero missing and zero
  unexpected keys.
- MV-TAP loads from `checkpoints/MVTAP.ckpt` through Lightning.
- LAPA Joint loads from its Hub snapshot with strict state-dict matching.
- DINOv2 Base loads from `HF_HOME` through Transformers offline mode.

Optional variants not downloaded: SpaTrackerV2 Online, Open-d4rt 48-frame, and
the TAPVid-3D-MC/PointOdyssey-MC-specific LAPA checkpoints.

Validated model identities:

| Component | Revision or source | Weight SHA-256 |
| --- | --- | --- |
| SpaTrackerV2 Front | `2e7164a93770cb934861d7de579bb81f58ad0b03` | `28c4377fd8bedfa1f43d4e486dfdce84813b8ce3af57ecce27a93f8a5f22b788` |
| SpaTrackerV2 Offline | `76e275b00f9c57dab71d46544df5255d4538106d` | `f1236958b274867ca9a743303eb2cf48a9d217a7d005e163b45a9ab87ed2e723` |
| Open-d4rt 32-frame | `7099b1fc760475de3b7409acd6b63f801b015d07` | `1f63305422fdc2000b057fbbc1d37459ac1a8063bbfcd0e3b7d473f5485943f5` |
| MV-TAP | Drive file `1sCml0BL6VQGy-MGgpidz2-BdymAJhboU` | `1294f213611acbf0807b3395e5e72d78d72bf429c6dfbb7cb15b9bf0fc05a894` |
| LAPA Joint | `0cce9285a629fb05a843edd690d38ca4107de177` | `57ee94f1e76811a2145d3a6c715419d3e2ce12151de744820e955acf27e1b76a` |
| DINOv2 Base safetensors | `f9e44c814b77203eaa57a6bdbbd535f21ede1415` | `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841` |

## Integrity rules

1. Record repository/file source, resolved revision, local relative path, byte
   size, and SHA-256 after each download.
2. Validate the real project loader; a successful HTTP request alone is not
   sufficient.
3. Keep credentials outside the repository and check model terms before
   redistribution.
4. Use only `.venv` commands and leave dependency versions unchanged during
   checkpoint acquisition.

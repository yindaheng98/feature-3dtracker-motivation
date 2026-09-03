# Model checkpoint inventory and storage

This is the durable routing guide for pretrained weights. Check official pages
again immediately before a download, because model files and access terms can
change. Do not download weights during read-only investigation; downloads into
`data/` require explicit user authorization.

## Recommended shared layout

Use `data/checkpoints/` as the shared store for newly downloaded or project-split
weights. It is ignored by the root Git repository, visible inside the Docker data
overlay, separate from raw experiment output, and can be treated as read-only
after population.

The existing host TrackerSplat dump is a separate OverlayFS mount: Compose uses
`/mnt/minorissd4tb/TrackerSplat/checkpoints` as `lowerdir` and project
`checkpoints/` as writable `upperdir`, matching `data/` and `output/`. Do not
copy that tree into the repository. TrackerSplat's hard-coded local path can
point at it with `TrackerSplat/checkpoints -> ../checkpoints` when no path
already exists.

```text
data/checkpoints/
  huggingface/                 # HF_HOME; SpaTrackerV2 and DINOv2 snapshots
  torch/                       # TORCH_HOME; Torch Hub repositories/weights
  Open-d4rt/checkpoints/
    OpenD4RT_32CLIP_9Dataset_NoAUG/
    OpenD4RT_48CLIP_9Mix_NoCropAUG/
  MV-TAP/MVTAP.ckpt
  LAPA/
    tapvid3d/lapa.pt
    pointodyssey/lapa.pt
    joint/lapa.pt
  TrackerSplat/                # flat names expected by its default loaders
```

From the root, use these variables for commands that may access model hubs:

```bash
export MODEL_ROOT="$PWD/data/checkpoints"
export HF_HOME="$MODEL_ROOT/huggingface"
export TORCH_HOME="$MODEL_ROOT/torch"
```

The home directory is read-only in the project container, so never rely on the
default `~/.cache/huggingface` or `~/.cache/torch`. After all required snapshots
exist, `HF_HUB_OFFLINE=1` can prevent accidental network access during runs.

## SpaTrackerV2

`SpaTrackerV2/inference.py` uses Hugging Face `from_pretrained` directly:

| Required for | Repository | Approximate size |
| --- | --- | ---: |
| RGB and RGBD front model | `Yuxihenry/SpatialTrackerV2_Front` | 4.63 GB |
| Offline tracking | `Yuxihenry/SpatialTrackerV2-Offline` | 276 MB |
| Online tracking | `Yuxihenry/SpatialTrackerV2-Online` | 264 MB |

The front model is always loaded. Select Offline or Online with `--track_mode`.
With `HF_HOME` set, the existing code downloads once and reuses the shared
cache; no code/path override is needed.

```bash
HF_HOME="$HF_HOME" .venv/bin/hf download Yuxihenry/SpatialTrackerV2_Front
HF_HOME="$HF_HOME" .venv/bin/hf download Yuxihenry/SpatialTrackerV2-Offline
# Only if online mode is needed:
HF_HOME="$HF_HOME" .venv/bin/hf download Yuxihenry/SpatialTrackerV2-Online

cd SpaTrackerV2
HF_HOME="../data/checkpoints/huggingface" \
  ../.venv/bin/python inference.py --track_mode offline <bounded input args>
```

The repositories are currently public and ungated, but their model cards do not
declare a weight license; verify usage/redistribution terms. `moge_as_base=True`
would additionally download `Ruicheng/moge-vitl`. The optional SAM demo needs a
separate SAM checkpoint and is not required by `inference.py`.

Official sources:

- <https://huggingface.co/Yuxihenry/SpatialTrackerV2_Front>
- <https://huggingface.co/Yuxihenry/SpatialTrackerV2-Offline>
- <https://huggingface.co/Yuxihenry/SpatialTrackerV2-Online>

## Open-d4rt

Two evaluation checkpoints are released. Each is about 14 GB and includes a
small `model.yaml`; download only one initially. The 32-frame variant is the
smallest sensible first choice and is also the initialization checkpoint used
by the published 48-frame training recipe.

```bash
.venv/bin/hf download Lijiaxin0111/OpenD4RT \
  --include "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt" \
  --include "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml" \
  --local-dir "$MODEL_ROOT/Open-d4rt"
```

Load it through explicit paths:

```bash
cd Open-d4rt
EXP="../data/checkpoints/Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG"
../.venv/bin/python eval_track3d_in_worldtrack.py \
  --model-config "$EXP/model.yaml" \
  --ckpt-path "$EXP/opend4rt.ckpt" \
  <bounded data/output args>
```

The alternative released variant is
`OpenD4RT_48CLIP_9Mix_NoCropAUG` with the same two filenames. Training, unlike
evaluation, also requires VideoMAEv2 `vit_g_hybrid_pt_1200e.pth`; its official
model zoo requires a download request. Pass it as `VIDEOMAE2_CKPT=<path>`.

Official sources:

- <https://huggingface.co/Lijiaxin0111/OpenD4RT/tree/main/checkpoints>
- <https://github.com/Lijiaxin0111/Open-d4rt>
- <https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md>

## MV-TAP

The official repository currently exposes one Google Drive file named
`MVTAP.ckpt`. It does not publish the size or access/license details, and no
third-party backbone download is present in the model path.

```bash
mkdir -p "$MODEL_ROOT/MV-TAP"
.venv/bin/gdown 1sCml0BL6VQGy-MGgpidz2-BdymAJhboU \
  -O "$MODEL_ROOT/MV-TAP/MVTAP.ckpt"

cd MV-TAP
../.venv/bin/python experiment.py \
  mode=eval ckpt_path="../data/checkpoints/MV-TAP/MVTAP.ckpt"
```

Lightning receives the path through `ckpt_path` and restores it in
`Trainer.test`. If Google Drive requires sign-in or confirmation, the user must
complete that access step rather than storing credentials in the repository.

Official sources:

- <https://github.com/cvlab-kaist/MV-TAP>
- <https://drive.google.com/file/d/1sCml0BL6VQGy-MGgpidz2-BdymAJhboU/view>

## LAPA

Three public task variants each contain a file named `lapa.pt`. Keep them in
separate directories to avoid overwriting one another:

| Use | Hugging Face repository | Local path |
| --- | --- | --- |
| TAPVid-3D-MC benchmark | `bishoygaloaa/LAPA-TAPVid-3D-MC` | `LAPA/tapvid3d/lapa.pt` |
| PointOdyssey-MC benchmark | `bishoygaloaa/LAPA-PointOdyssey-MC` | `LAPA/pointodyssey/lapa.pt` |
| General/mixed starting point | `bishoygaloaa/LAPA-Joint` | `LAPA/joint/lapa.pt` |

For initial general use, download only Joint; use a benchmark-specific model
when reproducing that benchmark.

```bash
.venv/bin/hf download bishoygaloaa/LAPA-Joint lapa.pt \
  --local-dir "$MODEL_ROOT/LAPA/joint"

cd Look-Around-and-Pay-Attention-LAPA-
../.venv/bin/python inference_lapa.py \
  --checkpoint "../data/checkpoints/LAPA/joint/lapa.pt" \
  <scene/feature/output args>
```

The loader expects `torch.load(path)["model"]`. Feature precomputation also
downloads `facebook/dinov2-base` through Hugging Face; `--use_cotracker` invokes
Torch Hub's `facebookresearch/co-tracker` `cotracker3_offline`. Set both
`HF_HOME` and `TORCH_HOME` during preprocessing. Inference reads the generated
HDF5 feature cache and does not fetch those backbones itself.

Official sources:

- <https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA->
- <https://huggingface.co/collections/bishoygaloaa/lapa-6a79dcf4bbedf556ad7da964>
- <https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-/releases>

## TrackerSplat

TrackerSplat does not publish a TrackerSplat neural checkpoint. Its learned
inputs are third-party point-tracker/initializer weights; reconstruction starts
from a scene-specific `point_cloud.ply` generated by initialization.

The default `dot-cotracker3` estimator needs exactly these three DOT files for a
minimal first run:

```text
movi_f_cotracker2_patch_4_wind_8.pth
cvo_raft_patch_8.pth
movi_f_raft_patch_4_alpha.pth
```

A bounded filename check on 2026-09-03 found none of these three files in the
existing host TrackerSplat checkpoint dump, so the minimal DOT set still needs
to be downloaded.

Download from the official DOT Hugging Face repository:

```bash
mkdir -p "$MODEL_ROOT/TrackerSplat"
for file in \
  movi_f_cotracker2_patch_4_wind_8.pth \
  cvo_raft_patch_8.pth \
  movi_f_raft_patch_4_alpha.pth
do
  .venv/bin/hf download 16lemoing/dot "$file" \
    --local-dir "$MODEL_ROOT/TrackerSplat"
done
```

The loaders do not auto-download missing files. Prefer the root OverlayFS dump
when it is mounted:

```bash
test ! -e TrackerSplat/checkpoints
ln -s ../checkpoints TrackerSplat/checkpoints
```

If using only newly downloaded files under `data/checkpoints/TrackerSplat`,
create this ignored compatibility link instead (only when no path already
exists):

```bash
test ! -e TrackerSplat/checkpoints
ln -s ../data/checkpoints/TrackerSplat TrackerSplat/checkpoints
```

or pass explicit values through repeatable `-o/--option_estimation` options:

```bash
-o "tracker_path='../data/checkpoints/TrackerSplat/movi_f_cotracker2_patch_4_wind_8.pth'" \
-o "estimator_path='../data/checkpoints/TrackerSplat/cvo_raft_patch_8.pth'" \
-o "refiner_path='../data/checkpoints/TrackerSplat/movi_f_raft_patch_4_alpha.pth'"
```

Optional alternatives, not needed for the default first run:

- Direct CoTracker3: `facebook/cotracker3/scaled_offline.pth`; pass
  `-o "checkpoint='<path>'"` with estimator `cotracker3`.
- DOT alternatives from `16lemoing/dot`:
  `movi_f_cotracker_patch_4_wind_8.pth`, `panning_movi_e_tapir.pth`, and
  `panning_movi_e_plus_bootstapir.pth`.
- InstantSplat initializer choices listed by TrackerSplat: three DUSt3R files,
  one MASt3R metric file, and Depth-Anything-V2 Small/Base/Large. Do not
  download all seven until the selected initializer is known. The default
  `nodepth-colmap-dense` path should be validated separately before selecting a
  learned initializer.

Official sources:

- <https://github.com/yindaheng98/TrackerSplat>
- <https://huggingface.co/16lemoing/dot>
- <https://huggingface.co/facebook/cotracker3>
- <https://download.europe.naverlabs.com/ComputerVision/DUSt3R/>
- <https://huggingface.co/depth-anything>

## Download and integrity policy

1. Download only the variant needed by the next bounded experiment.
2. Keep partial downloads and hub caches under `data/checkpoints/`, never in
   submodule source trees or `output/harness-runs/`.
3. Record source repository, revision, local relative path, byte size, and
   SHA-256 in a manifest after download. Never trust filename alone.
4. Check the model/data license and gated access before redistribution.
5. Validate loading with the smallest CPU parse or GPU model construction, then
   one bounded inference. A successful HTTP download is not model validation.
6. Do not commit checkpoint binaries or tokens. Authentication belongs in the
   user's external credential store.

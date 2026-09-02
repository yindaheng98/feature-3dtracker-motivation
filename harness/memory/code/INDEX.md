# Code Context Index

| Repository | Role | Persistent context rule | Project detail |
| --- | --- | --- | --- |
| `SpaTrackerV2/` | Offline monocular/RGBD 3D point tracking inference | Record its own commit/dirty paths; avoid loading model code and inference logs together in the main thread. | [SpaTrackerV2](../../projects/SpaTrackerV2.md) |
| `Open-d4rt/` | 4D reconstruction/tracking and WorldTrack evaluation | Default scripts may be multi-GPU; begin with bounded evaluation or dry-run checks. | [Open-d4rt](../../projects/Open-d4rt.md) |
| `MV-TAP/` | Multi-view 3D point tracking | Dataset/view generation and full evaluation can be large; use explicit paths and unique output. | [MV-TAP](../../projects/MV-TAP.md) |
| `TrackerSplat/` | Tracking-driven dynamic Gaussian reconstruction | Batch scripts contain destructive/overwriting steps; use targeted Python entrypoints for smoke tests. | [TrackerSplat](../../projects/TrackerSplat.md) |
| `Look-Around-and-Pay-Attention-LAPA-/` | Multi-camera transformer point tracking | Real runs require checkpoint, MC metadata, HDF5 features, and track NPZ; keep synthetic and real-data validation separate. | [LAPA](../../projects/LAPA.md) |

Per-experiment cards record exact commit, branch, dirty paths, changed symbols,
and diff stats. Never place full diffs in this index.

## Host build facts

| Date | Scope | Reusable fact |
| --- | --- | --- |
| 2026-09-02 | Root `.venv` CUDA extensions | The host GPUs are RTX A5000 (SM 8.6) and protected PyTorch is 2.6.0+cu124. The current runtime has no `gcc`, `g++`, `nvcc`, or `CUDA_HOME`; TrackerSplat cannot build its three custom extensions until a CUDA 12.4 build toolchain is provided. The similarly named pip `nvidia-cuda-nvcc-cu12` wheel contains `ptxas`, not the `nvcc` driver. |

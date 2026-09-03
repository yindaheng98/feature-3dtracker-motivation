# Code Context Index

| Repository | Role | Persistent context rule | Project detail |
| --- | --- | --- | --- |
| `SpaTrackerV2/` | Offline monocular/RGBD 3D point tracking inference | Record its own commit/dirty paths; avoid loading model code and inference logs together in the main thread. | [SpaTrackerV2](../../projects/SpaTrackerV2.md) |
| `Open-d4rt/` | 4D reconstruction/tracking and WorldTrack evaluation | Default scripts may be multi-GPU; begin with bounded evaluation or dry-run checks. | [Open-d4rt](../../projects/Open-d4rt.md) |
| `MV-TAP/` | Multi-view 3D point tracking | Dataset/view generation and full evaluation can be large; use explicit paths and unique output. | [MV-TAP](../../projects/MV-TAP.md) |
| `TrackerSplat/` | Tracking-driven dynamic Gaussian reconstruction | Batch scripts contain destructive/overwriting steps; use targeted Python entrypoints for smoke tests. | [TrackerSplat](../../projects/TrackerSplat.md) |
| `Look-Around-and-Pay-Attention-LAPA-/` | Multi-camera transformer point tracking | Real runs require checkpoint, MC metadata, HDF5 features, and track NPZ; keep synthetic and real-data validation separate. | [LAPA](../../projects/LAPA.md) |

When useful for reproducing a result, its compact memory records the relevant
commit, branch, dirty paths, changed symbols, and diff stats. Never place full
diffs in this index.

## Host build facts

| Date | Scope | Reusable fact |
| --- | --- | --- |
| 2026-09-02 | Root `.venv` CUDA extensions | The host GPUs are RTX A5000 (SM 8.6) and protected PyTorch is 2.6.0+cu124. TrackerSplat's three repository-local extensions and historical dependency extensions are built and execute successfully. The runtime still has no `gcc`, `g++`, `nvcc`, or `CUDA_HOME`, so it can use but not rebuild them; the pip `nvidia-cuda-nvcc-cu12` wheel contains `ptxas`, not the `nvcc` driver. |
| 2026-09-02 | Ubuntu 20.04 runtime ABI and native tools | Runtime glibc is 2.31. Pin Taichi 1.7.3 because the nominally manylinux_2_27 Taichi 1.7.4 CPython 3.12 wheel references GLIBC through 2.34; reinstalling cryptography 50.0.1 selected a compatible manylinux_2_28 wheel. System libGL/Open3D, COLMAP 3.6 CPU SIFT, and FFmpeg 4.2.7 H.264 encode all pass bounded tests. |

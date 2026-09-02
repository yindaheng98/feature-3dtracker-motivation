# Native and system prerequisites

The shared `.venv` contains the mutually compatible Python dependencies. The
historical `gaussian-splatting`, `InstantSplat`, `reduced-3dgs`,
`ExtrinsicInterpolator`, and CoTracker packages are installed, and
TrackerSplat's nested submodules are initialized. The remaining host blockers
are:

- TrackerSplat itself builds three custom CUDA extensions. The current runtime
  has no `gcc`, `g++`, `nvcc`, or `CUDA_HOME`, so package metadata generation
  fails before compilation. Use a CUDA 12.4 toolkit and target the RTX A5000
  explicitly with `TORCH_CUDA_ARCH_LIST=8.6`.
- The PyPI `nvidia-cuda-nvcc-cu12` wheel was tested and removed: it contains
  `ptxas` and headers but not the required `nvcc` driver.
- Open3D and InstantSplat's standard dense initializer require system
  `libGL.so.1`. Both `open3d` and `open3d-cpu` were tested; neither bundles it.
- The `colmap` and `ffmpeg` executables are not currently on `PATH`.

Do not solve these prerequisites by creating another Python environment or by
editing `.venv` files. After a compatible CUDA 12.4 compiler toolchain and the
system libraries/executables have been provided, install TrackerSplat with
`.venv/bin/python -m pip`, while applying `protected-stack.txt` and checking the
protected package versions afterward.
Use the pre-TrackerSplat-update commit set in
`trackersplat-historical-pins.md`; do not install the moving `main`/`master`
branches.

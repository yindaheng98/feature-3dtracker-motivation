# Native and system prerequisites

The shared `.venv` contains the mutually compatible Python dependencies. The
following TrackerSplat components cannot be installed on this host yet:

- `gaussian-splatting`, `InstantSplat`, and `reduced-3dgs` build custom CUDA
  extensions. Their sources are TrackerSplat nested submodules, which are not
  initialized, and installation currently fails because `CUDA_HOME` and
  `nvcc` are unavailable.
- `ExtrinsicInterpolator` imports `gaussian_splatting` unconditionally, so it
  must be installed only after that native package succeeds.
- The `colmap` and `ffmpeg` executables are not currently on `PATH`.

Do not solve these prerequisites by creating another Python environment or by
editing `.venv` files. After a compatible CUDA 12.8 compiler toolchain and the
system executables have been provided, initialize the exact nested submodules
and install each package with `.venv/bin/python -m pip`, while applying
`protected-stack.txt` and checking the protected package versions afterward.
Use the pre-TrackerSplat-update commit set in
`trackersplat-historical-pins.md`; do not install the moving `main`/`master`
branches.

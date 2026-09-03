# Environment ownership and troubleshooting

Use this file as the durable environment guide. Diagnose the layer before
changing packages; do not reconstruct environment history from old experiment
logs.

## Invariants

- The repository root `.venv` is the only Python environment. Invoke it as
  `.venv/bin/python`; never create or activate another environment.
- Never edit, copy, or delete `.venv`, `site-packages`, or `*.dist-info` files
  directly. Use `.venv/bin/python -m pip` from the repository root.
- Preserve every version in `protected-stack.txt`, especially PyTorch, CUDA,
  NumPy, and Pillow. Never install a submodule's raw requirements file.
- Use `shared-requirements.txt`, then `git-requirements.txt`, and use the exact
  historical TrackerSplat pins in `trackersplat-historical-pins.md`.
- Dependency changes require authorization from the current user request. A
  diagnostic request alone permits inspection and smoke tests, not installs.

## Ownership decision table

| Evidence or failure | Layer | Agent action | Owner/escalation |
| --- | --- | --- | --- |
| `ModuleNotFoundError`, Python API mismatch, or resolvable pip conflict | Python packages | With authorization, dry-run and install/reinstall a non-protected package through the Harness constraints; then run a real import or bounded behavior test. | Agent can resolve in `.venv`. |
| `GLIBC_x.y not found` from a package `.so` | Binary-wheel ABI | Confirm runtime glibc and wheel provenance; try the newest compatible wheel/version without changing protected packages. | Agent can repair a package wheel. If the interpreter itself or every viable wheel requires a newer ABI, the user must align/rebuild the image or host runtime. |
| `CUDA_HOME ... not set`, missing `gcc`/`g++`/`nvcc`, or extension compile failure before compilation | Native build toolchain | Use an already-built extension when its kernel passes. Do not substitute the PyPI `nvidia-cuda-nvcc-cu12` package for a real compiler. | User supplies a matching CUDA development toolkit/compiler in the host or image; Agent may build afterward if explicitly authorized. |
| Missing `libGL.so.1`, `ffmpeg`, `ffprobe`, `colmap`, or another system command/library | System/image dependency | Report the exact missing name and a bounded post-install validation command. Do not create a second Python environment as a workaround. | User installs it durably in the host/image or authorizes a Dockerfile change and rebuild. |
| CUDA driver/device unavailable or incompatible with the runtime | GPU host boundary | Collect `torch.cuda`/driver evidence without replacing PyTorch. | User fixes the host driver, NVIDIA Container Runtime, GPU exposure, or image compatibility. |
| Home cache is read-only | Runtime path/permissions | Redirect caches to a writable task-specific directory under `output/`. | Agent can resolve per run; user changes global ownership only if desired. |
| Missing datasets, checkpoints, credentials, licenses, or private access | External input/access | Identify the exact expected path/artifact and stop before downloading unless authorized. | User supplies or authorizes access/download. |

When escalating, tell the user: the missing component, direct evidence, why pip
cannot fix it, the required host/image action, and the exact command the Agent
will run afterward to verify it.

## Safe dependency workflow

Before a change, record repository status and protected versions. Inspect the
actual runtime rather than assuming the host and container have the same ABI:

```bash
.venv/bin/python -c "import platform,sys; print(sys.executable); print(platform.python_version()); print(platform.libc_ver())"
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.venv/bin/python -m pip check
```

Resolve only through the Harness manifests. Start with a dry run when the
change may affect many packages:

```bash
.venv/bin/python -m pip install --dry-run \
  -c harness/dependencies/protected-stack.txt \
  -r harness/dependencies/shared-requirements.txt

.venv/bin/python -m pip install --upgrade \
  -c harness/dependencies/protected-stack.txt \
  -r harness/dependencies/shared-requirements.txt

.venv/bin/python -m pip install --upgrade --no-deps \
  -c harness/dependencies/protected-stack.txt \
  -r harness/dependencies/git-requirements.txt
```

For one bad binary wheel, first dry-run and then reinstall only that package in
the runtime where it will execute. Pin the working version in a Harness manifest
when the latest release is incompatible:

```bash
.venv/bin/python -m pip install --dry-run \
  -c harness/dependencies/protected-stack.txt PACKAGE==VERSION
.venv/bin/python -m pip install --force-reinstall \
  -c harness/dependencies/protected-stack.txt PACKAGE==VERSION
```

After every change, run `pip check`, compare every protected version exactly,
and perform the smallest real behavior test. An import alone is insufficient
for CUDA/native code; execute one tiny kernel. A resolver warning alone is not
proof of failure; test the requested behavior. If the attempt fails, uninstall
or restore only the attempt-owned package changes and verify the prior state.

## Known-good state and reusable lessons

- Runtime: Python 3.12.8 on glibc 2.31; RTX A5000 (SM 8.6); protected PyTorch
  2.6.0+cu124. All five projects pass bounded executable smoke tests.
- `xformers==0.0.29.post3`, `moviepy==1.0.3`, `decord2`, and
  `taichi==1.7.3` are intentional compatibility selections.
- Taichi 1.7.4's CPython 3.12 wheel is tagged manylinux_2_27 but references
  GLIBC through 2.34. Taichi 1.7.3 loads on glibc 2.31. Cryptography 50.0.1
  works after reinstalling its manylinux_2_28 wheel inside this runtime.
- `gaussian-splatting` reports that `opencv-python` is missing because the
  working provider is `opencv-python-headless`. `cv2` color conversion and PNG
  encoding pass; do not install both OpenCV distributions merely to silence
  the metadata warning.
- TrackerSplat's repository-local KNN, featurefusion, and motionfusion CUDA
  extensions and the historical external extensions are already built and
  execute. A compiler is needed only to rebuild them.
- The current runtime has no `gcc`, `g++`, `nvcc`, or `CUDA_HOME`. The tested
  PyPI `nvidia-cuda-nvcc-cu12` wheel contains `ptxas` and headers, not the
  required `nvcc` driver. Rebuilding for this host needs a CUDA 12.4-compatible
  development toolkit and `TORCH_CUDA_ARCH_LIST=8.6`.
- System `libGL.so.1` works with Open3D 0.19.0. System COLMAP 3.6 performs CPU
  SIFT but has no CUDA. System FFmpeg/FFprobe 4.2.7 encode/probe H.264, and the
  ImageIO-FFmpeg bundled 7.0.2 executable also works.
- For Taichi on a read-only home directory, use a unique writable cache:

  ```bash
  mkdir -p output/<chosen-path>/taichi-cache
  TI_OFFLINE_CACHE_FILE_PATH="$PWD/output/<chosen-path>/taichi-cache" \
    .venv/bin/python <command>
  ```

- Passing these smokes validates the environment and code layer, not the
  presence of model weights/data or correctness of a full reconstruction.

## Optional FFmpeg and COLMAP upgrade

Ubuntu 20.04's installed FFmpeg 4.2.7 and COLMAP 3.6 pass the current bounded
tests but are not current feature releases. Upgrade only for a demonstrated
project need, and re-run the bounded encoder/probe and COLMAP feature/database
tests afterward.

For FFmpeg, use a numbered Linux static release from
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases/tag/latest),
not an unpinned master build. For a newer COLMAP, prefer a container/image
solution with its required compiler, Qt, CUDA, and glibc instead of turning a
COLMAP-only tool prefix into another project Python environment. Changing the
image OS requires reconsidering the ABI of the mounted `.venv`.

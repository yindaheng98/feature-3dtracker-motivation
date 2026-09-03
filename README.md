# 3D Tracker Experiment Environment

本仓库把五个 3D tracking 项目、一个共享 Python 环境和 Codex Experiment
Harness 放在同一个工作目录中，用于对话式代码修改、可复现实验、结果分析和跨会话
记忆。

当前包含以下 Git submodule：

- `SpaTrackerV2/`
- `Open-d4rt/`
- `MV-TAP/`
- `TrackerSplat/`
- `Look-Around-and-Pay-Attention-LAPA-/`

Docker 主要用于限制 Codex 能访问的宿主机目录，并提供少量系统工具。代码、唯一的
`.venv`、Codex CLI、`data/` 和 `output/` 均来自宿主机挂载；不要在各项目内另建
Conda、venv、uv 或 Poetry 环境。

## 已验证基线

当前工作区已在以下组合上通过五项目有界 smoke test：

| 组件 | 已验证状态 |
| --- | --- |
| Python | 3.12.8，容器运行时 glibc 2.31 |
| GPU | RTX A5000，SM 8.6 |
| PyTorch | 2.6.0+cu124；受 `harness/dependencies/protected-stack.txt` 保护 |
| Open3D/libGL | Open3D 0.19.0 可导入并执行基础几何操作 |
| COLMAP | 系统版 3.6，CPU SIFT 可用，无 CUDA |
| FFmpeg | 系统版 4.2.7 可编码/探测 H.264；ImageIO bundled 7.0.2 也可用 |
| TrackerSplat native extensions | KNN、featurefusion、motionfusion CUDA kernel 均可执行 |

这只能证明环境和最小代码路径可运行，不代表模型权重、数据集或完整训练/重建流程
已经就绪。

## 1. 宿主机前置条件

需要准备：

- Linux 宿主机；当前 Compose/Dockerfile 以 Ubuntu 20.04 ABI 为基线。
- Docker Engine、Docker Compose plugin 和 NVIDIA Container Toolkit。
- 可用的 NVIDIA 驱动与 GPU。
- 安装在当前用户目录下的 Node.js/NVM、Codex CLI 和 Python 3.12 解释器。
- 仅在重新编译 TrackerSplat CUDA 扩展时：与 PyTorch CUDA 兼容的 CUDA
  development toolkit、`gcc`、`g++` 和真正的 `nvcc`。

先在宿主机检查：

```bash
nvidia-smi
docker compose version
source "$HOME/.nvm/nvm.sh"
node --version
codex --version
```

当前 Docker image 没有 `gcc`、`g++` 或 `nvcc`。PyPI 包
`nvidia-cuda-nvcc-cu12` 也不包含可用于构建扩展的 `nvcc` driver，不能代替宿主机
CUDA toolkit。

## 2. 获取仓库和 submodule

克隆仓库后，从仓库根目录初始化所有 submodule，包括 TrackerSplat 的嵌套 CUDA
submodule：

```bash
git submodule update --init --recursive
git submodule status --recursive
```

后续所有命令若未特别说明，都应从本仓库根目录运行。

## 3. 配置 OverlayFS 数据目录

`docker-compose.yml` 中下面两个 `lowerdir` 是当前机器的绝对路径：

```text
/mnt/minorissd4tb/TrackerSplat/data
/mnt/minorissd4tb/TrackerSplat/output
```

在其他机器上使用前，必须把它们改成真实存在的只读基础数据/结果目录。项目内
`data/`、`output/` 是 writable upperdir；`.overlay-work/*` 是 OverlayFS workdir。
upperdir 与对应 workdir 必须位于同一文件系统，workdir 必须为空。

```bash
mkdir -p data output .overlay-work/data .overlay-work/output .codex

test -z "$(find .overlay-work/data -mindepth 1 -maxdepth 1 -print -quit)"
test -z "$(find .overlay-work/output -mindepth 1 -maxdepth 1 -print -quit)"
test "$(stat -c %d data)" = "$(stat -c %d .overlay-work/data)"
test "$(stat -c %d output)" = "$(stat -c %d .overlay-work/output)"

sudo docker compose config >/dev/null
```

Harness 将 `data/` 视为只读输入。新日志、指标、checkpoint 和可视化统一写到
`output/harness-runs/<experiment-id>/`，不要让测试脚本覆盖共享结果。

## 4. 安装并登录 Codex

使用当前用户的 NVM/Node.js 环境安装，不要使用 `sudo npm install`：

```bash
source "$HOME/.nvm/nvm.sh"
npm install -g @openai/codex
codex --version
```

Compose 将 `CODEX_HOME` 设置为项目内的 `.codex`。首次运行前用相同目录登录：

```bash
CODEX_HOME="$PWD/.codex" codex login
```

## 5. 创建唯一 Python 环境

不要激活环境，始终显式调用 `.venv/bin/python`。当前容器不自带 Python，而是通过
只读 `$HOME` 挂载使用宿主机解释器。因此基础解释器必须位于宿主机 `$HOME` 下，
并且在容器内能以相同绝对路径访问；不要依赖只存在于宿主机 `/usr/bin` 的解释器。

例如，当前机器可使用用户目录下的 Python 3.12：

```bash
"$HOME/miniconda3/bin/python3.12" -m venv .venv
.venv/bin/python --version
```

这里的 Miniconda 只提供基础 Python；`.venv` 仍是项目唯一环境。不要为各 submodule
创建 Conda environment。

本仓库假定 `.venv` 已经包含合适的 PyTorch 栈。不要用各项目的 requirements
文件重新安装或升级 PyTorch。当前受保护版本以
`harness/dependencies/protected-stack.txt` 为准；已有环境至少应确认：

```bash
.venv/bin/python -c \
  "import torch,torchvision,torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.version.cuda)"
```

若是全新机器，应先由用户在最终容器 ABI 下安装选定的 PyTorch/CUDA 组合，再更新并
确认 `protected-stack.txt`；不要让 Agent 猜测或替换该栈。

## 6. 构建容器

Dockerfile 当前安装 `git`、`ripgrep`、`libgl1`、COLMAP 和 FFmpeg 等系统工具，
但不安装 Python 或 CUDA development toolkit：

```bash
sudo docker compose build --pull
```

验证挂载的 Python 和 GPU 在最终运行时可见：

```bash
sudo docker compose run --rm dev \
  .venv/bin/python -c \
  "import platform,torch; print(platform.libc_ver()); print(torch.__version__, torch.cuda.is_available())"
```

如果这里出现 `GLIBC_x.y not found`，应在该容器运行时中重新选择/安装兼容 wheel，
而不是继续在 ABI 不同的宿主环境中安装二进制包。

## 7. 安装共享 Python 依赖

进入一次性安装 shell；仍然不要激活 `.venv`：

```bash
sudo docker compose run --rm dev bash
```

在容器内、仓库根目录依次执行：

```bash
.venv/bin/python -m pip install --upgrade pip

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

不要直接安装五个项目原始的 requirements 文件：其中的旧版 PyTorch、CUDA、NumPy
和 API pins 彼此冲突。Harness 当前有意选择了包括以下兼容版本：

- `xformers==0.0.29.post3`
- `moviepy==1.0.3`
- `taichi==1.7.3`
- `decord2`（提供 `decord` import）
- `opencv-python-headless`，不要再并装 `opencv-python`

详细原因见
[`harness/dependencies/native-prerequisites.md`](harness/dependencies/native-prerequisites.md)。

## 8. 安装 TrackerSplat native 依赖

仅在相应扩展尚未构建或需要重建时执行本节。由于当前容器没有编译工具链，这些命令
应由用户在具有 CUDA 12.4-compatible toolkit、`gcc/g++` 和 `nvcc` 的宿主机运行。
先检查：

```bash
command -v gcc
command -v g++
command -v nvcc
.venv/bin/python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"
```

如果任一项缺失，先在宿主机/镜像安装真正的开发工具链，不要尝试用 pip 包代替。
然后安装 TrackerSplat HEAD 之前的历史提交：

```bash
TORCH_CUDA_ARCH_LIST=8.6 .venv/bin/python -m pip install \
  --no-deps \
  --no-build-isolation \
  -c harness/dependencies/protected-stack.txt \
  git+https://github.com/yindaheng98/gaussian-splatting.git@017fe9b04015dc71a3eb153840e7937c7fa76f77 \
  git+https://github.com/yindaheng98/InstantSplat.git@303e98cec6180ee7484782c23edef6eb990171bd \
  git+https://github.com/yindaheng98/reduced-3dgs.git@f8d65eb171925d04dace3f68c175d609fd4ccec1 \
  git+https://github.com/yindaheng98/ExtrinsicInterpolator.git@5de703b258d65c39c394a8e7a08fa6391a66155c
```

CoTracker 已由 `git-requirements.txt` 固定到
`82e02e8029753ad4ef13cf06be7f4fc5facdda4d`。完整 pin 表见
[`trackersplat-historical-pins.md`](harness/dependencies/trackersplat-historical-pins.md)。

最后按 TrackerSplat 的 repository-local 布局安装其自身：

```bash
(
  cd TrackerSplat
  TORCH_CUDA_ARCH_LIST=8.6 ../.venv/bin/python -m pip install \
    --target . --upgrade --no-deps .
)
```

这会把 TrackerSplat 安装到 `TrackerSplat/`，而不是全局写入 `.venv`。运行时也应从
该目录调用 `../.venv/bin/python -m trackersplat...`。

## 9. 安装后验证

先检查 Python、系统命令和 Harness：

```bash
.venv/bin/python -c \
  "import torch,torchvision,numpy,cv2,open3d,taichi; print(torch.__version__, torch.cuda.is_available())"

ffmpeg -version
ffprobe -version
colmap -h

.venv/bin/python -m pip check
.venv/bin/python harness/tools/experiment.py check
```

当前 `pip check` 可能只报告：

```text
gaussian-splatting ... requires opencv-python, which is not installed
```

这是发行包名差异：实际使用的 `opencv-python-headless` 已通过 `cv2` 转换和 PNG 编码
测试。不要为了消除此提示同时安装两套 OpenCV。若出现其他错误，不能忽略。

各项目的最小验证入口、资源风险和真实数据要求见
[`harness/projects/INDEX.md`](harness/projects/INDEX.md)。CUDA/native 模块不能只验证
import，应至少执行一个有界 kernel；完整训练、全数据评估和模型下载不应作为首次
smoke test。

## 10. 启动 Codex 和使用 Harness

从仓库根目录启动：

```bash
sudo docker compose run --rm dev
```

Compose 默认运行：

```text
codex -a never -s danger-full-access
```

这意味着 Codex 在容器内无需逐次审批且没有额外文件沙箱；安全边界来自 Docker
挂载。宿主机 `$HOME` 为只读，仓库、`data/`/`output/` overlay 和网络可访问。

不需要显式告诉 Agent “保存经验”。从根目录启动后，Codex 会自动读取 `AGENTS.md`
并执行 Harness 协议：

- 每轮只预读 `harness/memory/ACTIVE.md` 和 `harness/memory/INDEX.md`。
- 具体想法、实验、结论、失败模式和用户决策会自动提炼为简短记忆。
- 完整日志与产物写入 `output/harness-runs/<experiment-id>/`。
- 长日志、跨项目代码和大量历史由只读 subagent 摘要，避免主 Agent context 膨胀。
- 未达到用户目标的代码尝试必须回退 attempt-owned 修改并核对各 submodule；无法安全
  回退时会标记 `rollback_pending`，阻止后续尝试继续建立在失败状态上。

Harness 的命令和目录结构见 [`harness/README.md`](harness/README.md)。通常直接描述
目标即可，例如：“在一个小样本上比较两个 tracking 想法，记录指标，失败时回退”。

## 11. Overlay volume 维护

修改 Compose 的 `lowerdir`、`upperdir` 或 `workdir` 后，需要删除旧 volume 定义再
重建：

```bash
sudo docker compose down -v --remove-orphans
sudo docker compose run --rm dev
```

`down -v` 是破坏性操作：它会删除该 Compose project 管理的 named volumes。当前
OverlayFS 配置不会删除作为宿主机 upperdir 的项目内 `data/` 和 `output/`，但执行前
仍应确认 Compose project 中没有其他需要保留的 named volume，并核对两个
`lowerdir` 和两个 workdir 的准确路径。

## 环境问题如何分工

- Agent 在用户授权安装依赖后，可以处理 `.venv` 内的 Python 包冲突、选择兼容
  wheel、做 constrained pip 安装、验证真实 import/kernel，并回退失败的包改动。
- 用户负责宿主机/镜像中的系统库和命令、编译器、CUDA toolkit/driver、NVIDIA
  Container Runtime、容器 ABI 变更、数据、权重、凭据和许可。
- 缺少 `libGL.so.1`、COLMAP、FFmpeg、`gcc/g++/nvcc` 时，Agent 应报告准确证据和
  验证命令，不能通过新建环境或安装同名 pip 包掩盖问题。

完整决策表、已知兼容版本和 ABI 排障流程见
[`harness/dependencies/native-prerequisites.md`](harness/dependencies/native-prerequisites.md)。

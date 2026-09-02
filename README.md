# 3D Tracker Development Environment

Docker 仅用于限制 Codex 对宿主机目录的访问权限。Node.js、Codex CLI、Python
虚拟环境和项目文件均来自宿主机挂载。

## 首次运行前准备

以下命令必须在项目根目录执行。`docker-compose.yml` 使用 `$PWD` 生成
OverlayFS 的绝对路径，因此不要从其他目录启动。

### 1. 安装 Codex CLI

使用当前用户的 NVM/Node.js 环境安装，不要使用 `sudo npm install`：

```bash
source "$HOME/.nvm/nvm.sh"
npm install -g @openai/codex
codex --version
```

### 2. 创建运行目录

```bash
mkdir -p data output .overlay-work/data .overlay-work/output .codex
```

`.overlay-work/data` 和 `.overlay-work/output` 是 OverlayFS 的工作目录，
必须与对应的 `data`、`output` 目录位于同一文件系统，并保持为空。

### 3. 创建 Python 虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

根据需要在该虚拟环境中安装各子项目的依赖。`.venv` 会随项目目录挂载进容器。

### 4. 手动安装 TrackerSplat 依赖

以下依赖需要在项目根目录手动安装。当前宿主机使用 RTX A5000 和 CUDA
12.4 编译器，因此需将 CUDA 扩展的目标架构限制为 SM 8.6，避免
InstantSplat 构建时生成该编译器不支持的目标：

```bash
TORCH_CUDA_ARCH_LIST=8.6 .venv/bin/python -m pip install \
  --no-deps \
  --no-build-isolation \
  -c harness/dependencies/protected-stack.txt \
  git+https://github.com/yindaheng98/gaussian-splatting.git@017fe9b04015dc71a3eb153840e7937c7fa76f77 \
  git+https://github.com/yindaheng98/InstantSplat.git@303e98cec6180ee7484782c23edef6eb990171bd \
  git+https://github.com/yindaheng98/reduced-3dgs.git@f8d65eb171925d04dace3f68c175d609fd4ccec1 \
  git+https://github.com/yindaheng98/ExtrinsicInterpolator.git@5de703b258d65c39c394a8e7a08fa6391a66155c \
  git+https://github.com/facebookresearch/co-tracker.git@82e02e8029753ad4ef13cf06be7f4fc5facdda4d
```

安装后检查共享环境是否仍然一致：

```bash
.venv/bin/python -m pip check
```

### 5. 登录 Codex

Compose 将 `CODEX_HOME` 设置为项目内的 `.codex`。首次运行前使用相同目录登录：

```bash
CODEX_HOME="$PWD/.codex" codex login
```

## 构建和运行

宿主机当前为 Ubuntu 20.04，`Dockerfile` 同样使用 Ubuntu 20.04，以保证挂载进
容器的 Node.js 和 Python 二进制文件具有兼容的系统 ABI。宿主机系统升级后，
应同步修改 `Dockerfile` 的基础镜像并重新构建。

```bash
sudo docker compose build --pull
sudo docker compose run --rm dev
```

Codex 将以 `never` 审批策略和 `danger-full-access` 容器沙箱模式运行。它仍受
Docker 挂载权限限制，但可以访问宿主机网络及所有挂载到容器中的文件。

## Overlay volume 配置变更

修改 `lowerdir`、`upperdir` 或 `workdir` 后，需要删除旧的 Docker volume
定义再重新运行：

```bash
sudo docker compose down -v --remove-orphans
sudo docker compose run --rm dev
```

该操作不会删除作为 OverlayFS `upperdir` 使用的项目内 `data` 和 `output`
目录，但会删除 Compose 管理的其他命名 volume。

## Harness 提问记录

什么是Harness Engineering

如何开始Harness Engineering

我想用 Harness Engineering 帮我基于现有项目写代码，需要为此准备一套简单的 Harness Engineering 。
现有项目位于当前目录下的几个 submodules 中，python运行环境位于 .venv 文件夹，相关数据位于 data 和 output 文件夹。
我想要的 Harness Engineering 主要的运行模式为在当前目录下和用户对话，按照用户的指示做一些实验并记录和分析实验结果、总结经验、根据新的想法修改代码并运行验证等任务，因此需要有记忆功能简要记录已经尝试过的想法和测试结果，看看要怎么写 Harness 能让所有在当前目录下运行的 Agent 都按照指定的格式从对话和实验中提取记忆、保存记忆并在对话中调取记忆。
这个项目可能导致Harness Engineering哪些部分的Context很长？比如多次运行后的记忆可能过长，多次修改代码可能导致Context过长，过长的Context不利于推理，在写Harness Engineering的时候把相关部分的instruction里加上subagent的指令，让模型将这些太长的Context整理为多个文件并留一个目录性质的文件，在主agent中只读取目录性质的文件并按需调用subagent阅读详细内容回报结果，从而节省主agent的Context空间。

我应该如何在 codex 里调用这个 Harness ？是不是直接在当前目录启动对话就可以了？能否实现让我的实验和对话不需要我显式地指定要保存说明经验，Agent自己知道要保存什么？

如果某个尝试无法达到用户所要求的目标，在退出前应该将修改回退，避免后来的程序在一个失败的尝试上修改。将这一指令加入你刚才写好的Harness提示词中。
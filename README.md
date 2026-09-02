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

### 4. 登录 Codex

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

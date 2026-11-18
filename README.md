# LLM Inference Lab

第一阶段使用 `Qwen/Qwen2.5-0.5B-Instruct`，
以 FP16 精度在单张 NVIDIA GPU 上运行 Hugging Face 原生推理。

## Windows 准备

1. 安装兼容 RTX 50 系列的 NVIDIA Windows 驱动，PowerShell 中运行 `nvidia-smi` 能看到显卡。
2. 安装 WSL2；运行 `wsl --list --verbose`，发行版的 VERSION 应为 2。
3. 安装并启动 Docker Desktop，在 Settings → General 中启用 WSL 2 based engine，使用 Linux containers。
4. PowerShell 中运行 `docker info --format '{{.OSType}}'`，输出应为 `linux`。

如需从 Ubuntu 终端执行 Docker 命令，还需要在 Docker Desktop 的 WSL Integration 中启用对应 Ubuntu。
Windows 驱动和容器内 CUDA 是不同层：`nvidia-smi` 显示的 CUDA Version 是驱动支持上限，
不要求它与镜像的 CUDA 12.8 数字相同。

## 第一次运行

在仓库根目录执行下面的 PowerShell 命令，每步成功后再继续。

```powershell
docker compose build
docker compose run --rm dev python scripts/check_gpu.py
docker compose run --rm dev python scripts/infer.py
```

首次构建会下载较大的 PyTorch 镜像并安装 Python 依赖；首次推理会从 Hugging Face 下载模型。
需要分别能够访问 Docker Hub、PyPI 和 Hugging Face 的下载服务。

GPU 检查成功时末尾输出 `PASS`；推理成功时会输出回答、生成 token 数和 PyTorch 峰值分配显存。
该显存指标不等于整张显卡的总占用，当前脚本用于验证环境，暂不作为性能基准。

自定义输入：

```powershell
docker compose run --rm dev python scripts/infer.py --prompt "什么是 KV Cache？" --max-new-tokens 128
```

## 日常开发

```powershell
docker compose run --rm dev bash
```

进入容器后执行 `python scripts/infer.py`，输入 `exit` 退出。
代码通过挂载同步到 `/workspace`，修改 Python 文件后直接重新运行即可，不需要重建镜像。
修改 `requirements.txt` 或 `Dockerfile` 后重新执行 `docker compose build`。

模型缓存存放在每台电脑自己的 Docker named volume 中，退出容器后保留。
共用 Git 仓库中的代码与环境配置，各自构建镜像、下载模型。
不要使用 `docker compose down -v`，除非确实想删除包括模型缓存在内的 Compose 卷。
本阶段不启动常驻服务，因此不需要 `docker compose up`。

## 常见问题

- `docker info` 连接失败：先启动 Docker Desktop，等引擎准备好。
- 拉取镜像时出现 Docker Hub `timeout`：检查网络；如使用代理，在 Docker Desktop 设置中配置可用的代理。
  浏览器可以访问网站，不代表 Docker 引擎自动使用了同一个代理。
- `CUDA unavailable` 或 GPU runtime 报错：确认 Linux containers、WSL2 和 Windows NVIDIA 驱动，重启 Docker Desktop 后再检查。
- 模型下载连接失败：检查 Hugging Face 下载服务在容器中的可达性；宿主机网络代理不会自动成为容器的代理。
- 显存不足：先关闭其他占用 GPU 的程序，将输出长度减小后重试；任务管理器中查看专用 GPU 内存。

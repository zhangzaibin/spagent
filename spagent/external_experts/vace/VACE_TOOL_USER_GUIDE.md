# 在 SPAgent 中使用 VACE 视频生成工具

VACE 在本地以 **独立 HTTP 服务** 运行，SPAgent 通过 `VaceTool`（`video_generation_vace_tool`）把「单张参考图 + 文本」交给服务，返回生成的 `.mp4` 路径。除安装 `requirements-vace.txt` 中的包外，还需要完成下面步骤。

## 环境要求

- **NVIDIA GPU**：显存需能跑 Wan2.1-VACE（与分辨率、帧数等设置有关；不足时可能出现进程被系统杀掉等情况）。
- **Python**：建议使用与 PyTorch / CUDA 匹配的版本；**启动 `vace_server` 的解释器**应与 `pip install -r requirements-vace.txt` 的环境**一致**。

## 1. 安装依赖（含 `requirements-vace.txt`）

在仓库根目录执行：

```bash
python -m pip install -r requirements-vace.txt
```

**关于 `flash-attn`**：若直接安装失败，请在**已安装 PyTorch** 的同一环境中尝试：

```bash
python -m pip install flash-attn --no-build-isolation
```

或使用与当前 CUDA / torch 版本匹配的预编译 wheel。

## 2. 下载模型权重（不包含在 Git 中）

权重体积大，需自行下载到仓库根目录下的 `checkpoints/vace/`，例如：

```bash
python -m pip install -U "huggingface_hub[cli]"
mkdir -p checkpoints/vace

huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B \
  --local-dir checkpoints/vace/Wan2.1-VACE-1.3B \
  --local-dir-use-symlinks False
```

## 3. 启动 VACE 服务

在已激活上述环境的终端中（**需要能访问 GPU 的机器上**）：

```bash
python spagent/external_experts/vace/vace_server.py \
  --checkpoint_path checkpoints/vace/Wan2.1-VACE-1.3B \
  --port 20034
```

默认 `--vace_root` 为 `vace_server.py` 所在目录（即 `spagent/external_experts/vace`），默认 `--checkpoint_path` 为 `checkpoints/vace/Wan2.1-VACE-1.3B`。若使用其他 Python，可加 `--python_exec /path/to/python`。

健康检查示例：

```bash
curl -s http://127.0.0.1:20034/health
```

响应中的 `runtime_deps_ok` 可帮助确认依赖是否齐全。

## 4. 在 SPAgent / 评测脚本里指向服务

- 使用 `VaceTool(use_mock=False, server_url="http://127.0.0.1:20034", mode="inference")`。
- 若在 `examples/evaluation/evaluate_img.py` 等处配置 `TOOL_SERVERS["vace"]`，请使用 **`http://127.0.0.1:<端口>`**（或本机实际 IP），**不要**把客户端写成 `http://0.0.0.0:...`（`0.0.0.0` 仅用于服务端监听）。

# Depth Anything 3 安装步骤（clone 在 /data/sjq）

在 `/data/sjq` 下 clone 并安装，以下命令按顺序执行即可。

## 1. 进入目录并 clone

```bash
cd /data/sjq
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
```

## 2. 激活你的 Python 环境（如 spagent 的 conda/venv）

```bash
# 若用 conda，例如：
conda activate spagent

# 确保当前环境是要装 Depth Anything 3 的环境
which python
pip --version
```

## 3. 安装 PyTorch 与 xformers（若尚未安装）

```bash
pip install "torch>=2" torchvision xformers
```

如已有合适版本的 PyTorch，可跳过或只补装 xformers：

```bash
pip install xformers
```

## 4. 以可编辑方式安装 Depth Anything 3

**仅基础功能（推荐先试）：**

```bash
# 当前目录应为 /data/sjq/Depth-Anything-3
pwd   # 应输出 .../Depth-Anything-3
pip install -e .
```

**需要 Gradio Web UI（Python>=3.10）：**

```bash
pip install -e ".[app]"
```

**需要全部依赖（3D 高斯、gsplat 等）：**

```bash
pip install -e ".[all]"
```

## 5. 验证安装

```bash
python -c "
from depth_anything_3.api import DepthAnything3
print('depth_anything_3 OK')
"
```

若无报错即安装成功。

## 6. 在 SPAgent 里用真实模型跑 Depth Anything 3 Tool

- 官方模型通过 Hugging Face 下载，例如单目深度模型：  
  `depth-anything/DA3MONO-LARGE`  
  或通用：`depth-anything/DA3-LARGE-1.1`

- 首次使用时会自动下载权重；若网络受限可设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

- 我们的 Tool 当前通过 `external_experts.depth_anything3.depth_anything3_client` 调用：  
  client 会先尝试官方 `depth_anything_3` API，因此安装好上述步骤后，在 Tool 里设置 `use_mock=False` 并传入合适的 `checkpoint_path`（或 client 支持的其他配置）即可跑真实推理。

- 使用 HuggingFace 模型 id 时，在 Tool 或环境变量里把 `checkpoint_path` 设为 `depth-anything/DA3MONO-LARGE` 等即可，client 会自动用官方 API 下载并推理。

---

## 安装好之后怎么做（然后呢）

### 1. 跑一次真实模型测试（推荐）

在项目根目录下，用 HuggingFace 模型 id 跑测试（首次会下载权重，可能较慢）：

```bash
cd /data/sjq/spagent
export DEPTH_ANYTHING3_CHECKPOINT="depth-anything/DA3MONO-LARGE"
python test/depth_anything3/test_depth_anything3_tool.py --real
```

或使用 pytest：

```bash
DEPTH_ANYTHING3_CHECKPOINT="depth-anything/DA3MONO-LARGE" pytest test/depth_anything3/test_depth_anything3_tool.py::test_depth_anything3_tool_real -v -s
```

成功后会生成 `test/depth_anything3/outputs/example_depth.png` 和 `example_depth.npy`。

### 2. 在代码里用 Tool（真实模型）

```python
from spagent.tools import DepthAnything3Tool

tool = DepthAnything3Tool(
    use_mock=False,
    checkpoint_path="depth-anything/DA3MONO-LARGE",  # 或本地 .pth 路径
    device="cuda",
    save_dir="./depth_outputs",
)
result = tool.call(image_path="path/to/your/image.png", output_format="both")
# result["success"], result["result"]["depth_png_path"], ...
```

### 3. 网络不好时用 HF 镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

再执行上面的测试或推理即可。

---

## 命令汇总（复制一整段执行）

```bash
cd /data/sjq
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
# conda activate spagent   # 如需要
pip install "torch>=2" torchvision xformers
pip install -e .
python -c "from depth_anything_3.api import DepthAnything3; print('OK')"
```

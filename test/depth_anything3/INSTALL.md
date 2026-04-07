# Depth Anything 3 依赖安装（用于真实模型测试与 Tool 真实推理）

本说明用于：在 **test/depth_anything3** 跑真实模型测试，或在 **spagent/tools/depth_anything3_tool.py** 中 `use_mock=False` 时所需的环境。  
Tool 的调用与注册始终在 `spagent/tools/depth_anything3_tool.py`，见 [README.md](README.md) 与 [docs/ADDING_NEW_TOOLS.md](../../docs/ADDING_NEW_TOOLS.md)。

以下步骤使用相对路径（以 spagent 项目根为基准）；也可在任意工作目录 clone，将路径替换为你的实际路径即可。

## 1. Clone 仓库（示例：项目根上一级，相对路径 ../Depth-Anything-3）

在 spagent 项目根目录执行：

```bash
cd ..
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
# 当前目录应为 clone 下来的 Depth-Anything-3 根目录
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

## 6. 安装后：启动 Server 与 使用 Tool（与 Pi3X 一致）

采用 **Server/Client 架构**：模型在独立进程中运行，Tool 通过 HTTP 调用。

### 启动 Depth Anything 3 Server

在项目根或任意目录（已安装 depth-anything-3 的环境）执行：

```bash
python -m spagent.external_experts.depth_anything3.depth_anything3_server \
  --checkpoint_path depth-anything/DA3MONO-LARGE \
  --port 20032
```

`--checkpoint_path` 可为 HuggingFace 模型 id 或本地 .pth 路径。国内可设 `export HF_ENDPOINT=https://hf-mirror.com`。

### 跑本目录真实模型测试

先启动 Server（见上），再在项目根执行：

```bash
python test/depth_anything3/test_depth_anything3_tool.py --real
# 或
pytest test/depth_anything3/test_depth_anything3_tool.py::test_depth_anything3_tool_real -v -s
```

可选：`export DEPTH_ANYTHING3_SERVER_URL=http://localhost:20032`（默认即此）。

### 在业务代码里使用 Tool（真实模型）

先启动 Server，再在代码中：

```python
from spagent.tools import DepthAnything3Tool

tool = DepthAnything3Tool(
    use_mock=False,
    server_url="http://localhost:20032",
    save_dir="test/depth_anything3/outputs",
)
result = tool.call(image_path="assets/example.png", output_format="both")
```

---

## 命令汇总（在项目根上一级 clone，相对路径）

```bash
cd ..   # 从 spagent 根目录到上一级
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install "torch>=2" torchvision xformers
pip install -e .
cd ../spagent
python -c "from depth_anything_3.api import DepthAnything3; print('OK')"
```

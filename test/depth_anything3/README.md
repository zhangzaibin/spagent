# Depth Anything 3 Tool — 真实部署与运行

本文档说明如何从 clone 仓库到运行真实模型的全流程。**所有路径均以相对路径给出**，默认当前工作目录为 **spagent 项目根目录**（即与 `test`、`assets`、`spagent` 同级）。

---

## 前置要求

- Python 3.9–3.13，已激活你的环境（如 `conda activate spagent`）
- CUDA（若用 GPU）
- Git

---

## 1. Clone Depth Anything 3 仓库

建议 clone 在 **项目根目录外**（避免把第三方仓库纳入 spagent 的 git），使用相对路径时即为「项目根目录的上一级」：

```bash
# 从 spagent 项目根目录执行
cd ..
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
```

若希望 clone 在项目内（并加入 .gitignore），可放在 `external_experts` 下：

```bash
# 从 spagent 项目根目录执行
cd spagent/external_experts
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd ../../..
```

以下步骤假设 clone 在 **上一级目录**，即相对项目根为 `../Depth-Anything-3`。

---

## 2. 安装依赖（PyTorch、xformers）

在 **Depth-Anything-3** 目录下执行（或任意已激活环境的终端）：

```bash
# 若当前在 Depth-Anything-3 目录
pip install "torch>=2" torchvision xformers
```

若本机已有合适版本的 PyTorch，可只补装 xformers：

```bash
pip install xformers
```

---

## 3. 以可编辑方式安装 Depth Anything 3

仍在 **Depth-Anything-3** 目录下：

```bash
# 基础安装（推荐）
pip install -e .
```

可选：

```bash
pip install -e ".[app]"   # 含 Gradio Web UI，需 Python>=3.10
pip install -e ".[all]"   # 含 3D 高斯等全部功能
```

安装完成后，任意目录下 Python 都能 `import depth_anything_3`，无需再关心 clone 的绝对路径。

---

## 4. 验证安装

回到 **spagent 项目根目录** 再验证，确保在真实使用场景下能导入：

```bash
cd path/to/spagent   # 换成你的 spagent 根目录相对或绝对路径，例如从 Depth-Anything-3 回来：cd ../spagent
python -c "from depth_anything_3.api import DepthAnything3; print('depth_anything_3 OK')"
```

无报错即表示安装成功。

---

## 5. 运行真实模型测试

所有以下命令均在 **spagent 项目根目录** 下执行，使用相对路径。

### 5.1 使用 HuggingFace 模型 id（推荐，无需本地权重）

首次运行会自动从 Hugging Face 下载权重；国内可先设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后执行：

```bash
export DEPTH_ANYTHING3_CHECKPOINT="depth-anything/DA3MONO-LARGE"
python test/depth_anything3/test_depth_anything3_tool.py --real
```

或使用 pytest：

```bash
DEPTH_ANYTHING3_CHECKPOINT="depth-anything/DA3MONO-LARGE" pytest test/depth_anything3/test_depth_anything3_tool.py::test_depth_anything3_tool_real -v -s
```

测试会读取 `assets/example.png`，输出写入 `test/depth_anything3/outputs/`：

- `test/depth_anything3/outputs/example_depth.png`
- `test/depth_anything3/outputs/example_depth.npy`

### 5.2 使用本地 checkpoint（相对路径）

若已将权重下载到本目录或子目录，可用相对路径指定，例如：

```bash
# 假设权重在 test/depth_anything3/checkpoints/depth_anything_v3_vitl.pth
export DEPTH_ANYTHING3_CHECKPOINT="test/depth_anything3/checkpoints/depth_anything_v3_vitl.pth"
python test/depth_anything3/test_depth_anything3_tool.py --real
```

或在 `test/depth_anything3/` 下放置任意 `.pth` 文件，测试脚本会自动选用（见 `test_depth_anything3_tool.py` 中的逻辑）。

---

## 6. 在代码中真实部署运行 Tool

在 **spagent 项目根目录** 下运行你的脚本，或保证工作目录为项目根，以便相对路径一致。

### 6.1 使用 HuggingFace 模型 id

```python
from pathlib import Path
from spagent.tools import DepthAnything3Tool

# 建议先切到项目根，或使用基于项目根的相对路径
ROOT = Path(__file__).resolve().parents[1]   # 若脚本在 test/depth_anything3/ 下
# 或 ROOT = Path(".").resolve()  若在根目录运行

tool = DepthAnything3Tool(
    use_mock=False,
    checkpoint_path="depth-anything/DA3MONO-LARGE",
    device="cuda",
    save_dir="test/depth_anything3/outputs",
)
result = tool.call(
    image_path="assets/example.png",
    output_format="both",
    colormap="inferno",
    normalize=True,
)
print(result["success"], result.get("result", {}).get("depth_png_path"))
```

### 6.2 使用本地 checkpoint（相对路径）

```python
tool = DepthAnything3Tool(
    use_mock=False,
    checkpoint_path="test/depth_anything3/checkpoints/your_model.pth",  # 相对项目根
    device="cuda",
    save_dir="test/depth_anything3/outputs",
)
result = tool.call(image_path="assets/example.png", output_format="both")
```

### 6.3 Mock 模式（不依赖真实模型）

不 clone、不装 Depth Anything 3 也可跑 Tool，仅用于联调/CI：

```python
tool = DepthAnything3Tool(use_mock=True, save_dir="test/depth_anything3/outputs")
result = tool.call(image_path="assets/example.png", output_format="both")
```

---

## 7. 路径与工作目录约定（相对路径汇总）

| 用途           | 相对路径（均相对 spagent 项目根） |
|----------------|-----------------------------------|
| 测试输入图     | `assets/example.png`              |
| 测试输出目录   | `test/depth_anything3/outputs`    |
| 测试脚本       | `test/depth_anything3/test_depth_anything3_tool.py` |
| 本地 checkpoint 示例 | `test/depth_anything3/checkpoints/xxx.pth` |
| Depth-Anything-3 clone（在根目录外） | `../Depth-Anything-3` |

运行测试或调用 Tool 时，请保证 **当前工作目录为 spagent 项目根**，或自行将上述相对路径改为相对当前工作目录的路径。

---

## 8. 可选：HF 镜像与其它模型

- 国内下载 HuggingFace 权重可设：`export HF_ENDPOINT=https://hf-mirror.com`
- 其它官方模型 id 示例：`depth-anything/DA3-LARGE-1.1`、`depth-anything/DA3-BASE` 等，替换 `DEPTH_ANYTHING3_CHECKPOINT` 或 `checkpoint_path` 即可。

---

## 9. 故障排除

- **ModuleNotFoundError: depth_anything_3**  
  未安装或未激活环境：在 Depth-Anything-3 目录下执行 `pip install -e .`，并确保运行脚本时使用同一环境。

- **FileNotFoundError: assets/example.png**  
  当前工作目录不是 spagent 项目根：先 `cd` 到项目根再运行，或把 `image_path` 改为绝对路径/相对当前目录的路径。

- **Checkpoint not found**  
  使用 HF id 时不要传本地路径；使用本地路径时确保该相对路径在当前工作目录下存在（如 `test/depth_anything3/checkpoints/xxx.pth`）。

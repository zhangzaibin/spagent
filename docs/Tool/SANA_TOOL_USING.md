# Sana Tool 使用说明

本文档说明如何在 SPAgent 中使用 `SanaTool`，包括：

- 新增了什么功能
- 相比原版项目修改了什么
- 涉及哪些文件
- 如何启动 Sana 服务
- 如何进行快速测试、Agent 调用与评测

---

## 1. 功能概述

本次为 SPAgent 新增了一个图像生成工具 `SanaTool`，用于把文本描述转成图像，并接入到 Agent 的工具调用链中。

它的定位不是“看图理解”，而是“生成式可视化”：

- 可视化假设场景
- 可视化目标状态
- 可视化计划结果
- 可视化 imagined world state

这意味着：

- `SanaTool` 适合 generation / planning / world-modeling 场景
- 不适合替代 detection / segmentation / depth / 3D reconstruction 这类事实性感知工具

---

## 2. 与原版项目相比新增了什么

原版项目主要聚焦：

- 空间感知
- 视觉分析
- 3D 重建
- 视频生成

本次新增的是：

1. 本地图像生成工具 `SanaTool`
2. Sana 的本地 SGLang 启动脚本
3. Sana 的 client / mock service
4. 专门的 Sana 评测脚本
5. 针对生成任务的 prompt 模板
6. `SPAgent` 的 text-only 推理支持
7. `SPAgent` 的 `workflow_mode="auto"` 自动工作流路由

这使得项目从“以空间感知为主”扩展到了“感知 + 推理 + 生成式可视化”的模式。

---

## 3. 对原版项目的主要改动

### 3.1 新增的文件

#### Sana 服务与工具实现

- [scripts/run_sana_30000.sh](/mnt/st_4t/zhanghao/spagent/scripts/run_sana_30000.sh)
- [spagent/external_experts/Sana/sana_client.py](/mnt/st_4t/zhanghao/spagent/spagent/external_experts/Sana/sana_client.py)
- [spagent/external_experts/Sana/mock_sana_service.py](/mnt/st_4t/zhanghao/spagent/spagent/external_experts/Sana/mock_sana_service.py)
- [spagent/tools/sana_tool.py](/mnt/st_4t/zhanghao/spagent/spagent/tools/sana_tool.py)
- [test/test_sana_tool.py](/mnt/st_4t/zhanghao/spagent/test/test_sana_tool.py)
- [examples/evaluation/evaluate_sana.py](/mnt/st_4t/zhanghao/spagent/examples/evaluation/evaluate_sana.py)
- [dataset/sana_cases_sample.jsonl](/mnt/st_4t/zhanghao/spagent/dataset/sana_cases_sample.jsonl)
- [sana_run.py](/mnt/st_4t/zhanghao/spagent/sana_run.py)

### 3.2 修改的文件

- [spagent/tools/__init__.py](/mnt/st_4t/zhanghao/spagent/spagent/tools/__init__.py)
  作用：导出 `SanaTool`

- [spagent/core/prompts.py](/mnt/st_4t/zhanghao/spagent/spagent/core/prompts.py)
  作用：
  - 增加生成任务 prompt
  - 增加 `GENERATION_SYSTEM_PROMPT`
  - 增加 `GENERATION_CONTINUATION_HINT`
  - 增加 Sana 的 tool usage policy

- [spagent/core/spagent.py](/mnt/st_4t/zhanghao/spagent/spagent/core/spagent.py)
  作用：
  - 增加 text-only 推理支持
  - 增加 `_run_model_inference(...)`
  - 增加 `workflow_mode="auto"`
  - 自动在 `spatial_3d / general_vision / generation` 之间路由

- [examples/evaluation/evaluate_all_tools.py](/mnt/st_4t/zhanghao/spagent/examples/evaluation/evaluate_all_tools.py)
  作用：
  - 将 Sana 纳入统一评测入口
  - 增加 `sana_real / sana_mock`
  - 增加 image generation 结果统计

---

## 4. 架构设计

Sana 接入遵循项目原有工具架构：

```text
SPAgent
  -> ToolRegistry
    -> SanaTool
      -> SanaClient
        -> SGLang /v1/images/generations
```

其中：

- `SanaTool` 负责暴露统一工具接口
- `SanaClient` 负责调用 Sana 服务并保存生成图
- `MockSanaService` 用于本地 mock 测试
- `SPAgent` 负责解析 `<tool_call>` 并执行工具

---

## 5. 工作流自动路由

本次改动给 `SPAgent` 增加了：

```python
workflow_mode="auto"
```

自动路由逻辑大致如下：

- `generation`
  - 只有生成类工具
  - 无输入图且任务是生成
  - prompt 中出现 `generate/create/visualize/imagine/render/synthesize`

- `spatial_3d`
  - 工具中有 `Pi3 / Pi3X / VGGT / MapAnything`
  - 问题中出现 `3d/viewpoint/azimuth/elevation/camera/orientation/top view`

- `general_vision`
  - 默认普通视觉分析

因此 Sana 这类纯文本生成任务现在可以自动走 generation prompt，而不再错误使用默认的 3D workflow。

---

## 6. 如何启动 Sana 服务

### 6.1 启动脚本

使用：

```bash
bash scripts/run_sana_30000.sh
```

默认配置：

- host: `0.0.0.0`
- port: `30000`
- model: `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers`

### 6.2 可选参数

例如：

```bash
bash scripts/run_sana_30000.sh --gpu-device 0 --vae-cpu-offload --text-encoder-cpu-offload
```

如果环境里有代理，脚本默认会尝试避免 SOCKS 代理导致的下载问题。

---

## 7. SanaClient 的用法

示例：

```python
from spagent.external_experts.Sana.sana_client import SanaClient

client = SanaClient(server_url="http://127.0.0.1:30000")
result = client.generate_image(
    prompt="a compact home robot organizing books in a warm study room",
    size="1024x1024",
    num_inference_steps=20,
    guidance_scale=4.5,
    seed=42,
)
print(result)
```

返回结果通常包含：

- `success`
- `output_path`
- `image_paths`
- `file_size_bytes`
- `model`
- `size`
- `seed`

---

## 8. SanaTool 的用法

### 8.1 直接调用工具

```python
from spagent.tools import SanaTool

tool = SanaTool(
    use_mock=False,
    server_url="http://127.0.0.1:30000",
)

result = tool.call(
    prompt="a household robot organizing books in a warm study room",
    size="1024x1024",
    num_inference_steps=20,
    guidance_scale=4.5,
    seed=42,
)

print(result)
```

### 8.2 Mock 模式

```python
from spagent.tools import SanaTool

tool = SanaTool(use_mock=True)
result = tool.call(prompt="a mobile robot in an office hallway")
print(result)
```

---

## 9. 在 SPAgent 中使用 Sana

推荐使用自动 workflow：

```python
from spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools import SanaTool

model = GPTModel(model_name="gpt-4o-mini")
tools = [
    SanaTool(
        use_mock=False,
        server_url="http://127.0.0.1:30000",
    )
]

agent = SPAgent(
    model=model,
    tools=tools,
    workflow_mode="auto",
)

result = agent.solve_problem(
    [],
    "Generate an image of a compact household robot organizing books on a wooden shelf in a warm study room."
)

print(result["prompts"]["workflow"])
print(result["tool_calls"])
print(result["additional_images"])
```

说明：

- 输入图为空 `[]` 时，`SPAgent` 现在会走 `text_only_inference`
- `workflow_mode="auto"` 会自动路由到 `generation`

---

## 10. 评测方式

### 10.1 专用 Sana 评测

脚本：

- [examples/evaluation/evaluate_sana.py](/mnt/st_4t/zhanghao/spagent/examples/evaluation/evaluate_sana.py)

运行 mock：

```bash
python3 examples/evaluation/evaluate_sana.py \
  --config sana_mock \
  --data_path dataset/sana_cases_sample.jsonl \
  --model gpt-4o \
  --max_samples 3
```

运行 real：

```bash
python3 examples/evaluation/evaluate_sana.py \
  --config sana_real \
  --data_path dataset/sana_cases_sample.jsonl \
  --model gpt-4o \
  --max_samples 3
```

### 10.2 统一评测入口

脚本：

- [examples/evaluation/evaluate_all_tools.py](/mnt/st_4t/zhanghao/spagent/examples/evaluation/evaluate_all_tools.py)

运行：

```bash
python3 examples/evaluation/evaluate_all_tools.py \
  --config sana_real \
  --data_path dataset/sana_cases_sample.jsonl \
  --image_base_path dataset \
  --model gpt-4o \
  --max_samples 3
```

也支持：

- `sana_mock`
- `all_image`
- `all_generation`

---

## 11. 测试

最小测试文件：

- [test/test_sana_tool.py](/mnt/st_4t/zhanghao/spagent/test/test_sana_tool.py)

说明：

- 默认可以跑 mock 测试
- 可选真实服务测试需要设置环境变量

如果当前环境没有 `pytest`，需要先安装：

```bash
python -m pip install pytest
```

---

## 12. 对原版 SPAgent 的影响

### 正向影响

1. 工具生态更完整
   从空间感知扩展到了生成式可视化

2. 更适合 embodied / planning / world-modeling 场景
   可以把“目标状态”或“假设场景”直接生成出来

3. `SPAgent` 更通用
   现在不仅支持单图、多图，还支持 text-only 任务

4. workflow 更合理
   通过 `workflow_mode="auto"`，Sana 不再误用 3D 工作流

### 需要注意的点

1. 生成图不是事实证据
   它只能作为 synthetic visualization

2. Sana 适合生成任务，不适合代替 detection / segmentation / depth / 3D 工具

3. 生成任务的评估方式和 VQA 不一样
   更适合统计：
   - tool called
   - generation success
   - output path
   - latency

---

## 13. 建议的后续工作

如果继续完善 Sana 接入，建议优先做：

1. 扩展更贴近空间规划的 Sana 数据集
2. 增加更丰富的 generation prompt 模板
3. 记录 Sana 调用前后的决策链
4. 评估 Sana 是否真正提升 planning / world-modeling 任务
5. 视需要支持 image-to-image / conditioned generation

---

## 14. 相关文件索引

### 核心实现

- [spagent/tools/sana_tool.py](/mnt/st_4t/zhanghao/spagent/spagent/tools/sana_tool.py)
- [spagent/external_experts/Sana/sana_client.py](/mnt/st_4t/zhanghao/spagent/spagent/external_experts/Sana/sana_client.py)
- [spagent/external_experts/Sana/mock_sana_service.py](/mnt/st_4t/zhanghao/spagent/spagent/external_experts/Sana/mock_sana_service.py)

### Agent 改动

- [spagent/core/spagent.py](/mnt/st_4t/zhanghao/spagent/spagent/core/spagent.py)
- [spagent/core/prompts.py](/mnt/st_4t/zhanghao/spagent/spagent/core/prompts.py)
- [spagent/tools/__init__.py](/mnt/st_4t/zhanghao/spagent/spagent/tools/__init__.py)

### 脚本与评测

- [scripts/run_sana_30000.sh](/mnt/st_4t/zhanghao/spagent/scripts/run_sana_30000.sh)
- [sana_run.py](/mnt/st_4t/zhanghao/spagent/sana_run.py)
- [examples/evaluation/evaluate_sana.py](/mnt/st_4t/zhanghao/spagent/examples/evaluation/evaluate_sana.py)
- [examples/evaluation/evaluate_all_tools.py](/mnt/st_4t/zhanghao/spagent/examples/evaluation/evaluate_all_tools.py)
- [dataset/sana_cases_sample.jsonl](/mnt/st_4t/zhanghao/spagent/dataset/sana_cases_sample.jsonl)
- [test/test_sana_tool.py](/mnt/st_4t/zhanghao/spagent/test/test_sana_tool.py)

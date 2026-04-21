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

- `scripts/run_sana_30000.sh`
- `spagent/external_experts/Sana/sana_client.py`
- `spagent/external_experts/Sana/mock_sana_service.py`
- `spagent/tools/sana_tool.py`
- `test/test_sana_tool.py`
- `examples/evaluation/evaluate_sana.py`
- `dataset/sana_cases_sample.jsonl`
- `sana_run.py`

### 3.2 修改的文件

- `spagent/tools/__init__.py`
  作用：导出 `SanaTool`

- `spagent/core/prompts.py`
  作用：
  - 增加生成任务 prompt
  - 增加 `GENERATION_SYSTEM_PROMPT`
  - 增加 `GENERATION_CONTINUATION_HINT`
  - 增加 Sana 的 tool usage policy

- `spagent/core/spagent.py`
  作用：
  - 增加 text-only 推理支持
  - 增加 `_run_model_inference(...)`
  - 增加 `workflow_mode="auto"`
  - 自动在 `spatial_3d / general_vision / generation` 之间路由

- `examples/evaluation/evaluate_all_tools.py`
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

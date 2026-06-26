# Quick Eval 快速评测脚本

> **English Version**: [QUICK_EVAL.md](QUICK_EVAL.md) | **中文版本**: 本文档

`scripts/quick_eval.py` 是 SPAgent 的快速评测入口，绕过 vlmeval 的 `infer_data_job` 流程，直接遍历数据集 → `SPAgent.solve_problem` → 写入 xlsx → 调用 `dataset.evaluate()` 打分。

---

## 快速开始

```bash
# 最简运行：5 条样本，不用任何工具
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --datasets VStarBench \
    --limit 5

# 无工具基线，跨 5 个 benchmark
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --datasets MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI \
    --limit 50

# GroundingDINO 组合（zoom 看属性，localize 做空间/计数）
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools zoom localize \
    --datasets MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI \
    --limit 50 \
    --detection-url http://localhost:20022

# 感知组合：zoom + 分割 + 深度
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools zoom segmentation depth \
    --datasets MMStar \
    --limit 50 \
    --detection-url   http://localhost:20022 \
    --segmentation-url http://localhost:20020 \
    --depth-url       http://localhost:20019

# 空间理解组合（感知 + 3D 重建）
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools zoom localize segmentation depth pi3x \
    --datasets VStarBench BLINK \
    --limit 50
```

---

## 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `gpt-4.1-mini` | 推理用的 LLM 模型名称（兼容 OpenAI / LiteLLM） |
| `--tools` | _(无)_ | 启用的工具，空格分隔，可任意组合（见下方工具列表） |
| `--datasets` | `VStarBench` | 要评测的数据集名称，空格分隔，可同时指定多个 |
| `--limit` | _(各数据集默认值)_ | 每个数据集最多取多少条样本，不填则用内置默认值 |

### 推理控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-iterations` | `3` | Agent 每道题最多迭代几轮（含工具调用） |
| `--temperature` | `0.0` | LLM 生成温度 |
| `--seed` | `42` | 随机种子，用于结果复现 |

### 评分与输出

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--judge-model` | `gpt-4o-mini` | 评分时使用的裁判模型 |
| `--nproc` | `4` | 评分并发进程数 |
| `--no-score` | _(关闭)_ | 加此标志则只跑推理，跳过评分步骤 |
| `--work-dir` | `outputs/vlmeval_runs` | xlsx 预测文件和评分摘要的保存目录 |
| `--trace-dir` | `outputs/spagent_traces` | 每条样本的推理 trace（JSON）保存目录 |

---

## 工具列表

`--tools` 支持以下工具名（可任意组合）：

| 工具名 | 对应类 | 功能描述 | URL 参数 | 默认地址 |
|--------|--------|----------|----------|----------|
| `zoom` | ZoomObjectTool | GroundingDINO 裁剪特写，看属性（颜色/纹理/文字） | `--detection-url` | `http://localhost:20022` |
| `localize` | LocalizeObjectTool | GroundingDINO 画检测框，做空间/计数 | `--detection-url` | `http://localhost:20022` |
| `detection` | ZoomObjectTool | `zoom` 的向后兼容别名 | `--detection-url` | `http://localhost:20022` |
| `segmentation` | SegmentationTool | 实例分割（SAM2） | `--segmentation-url` | `http://localhost:20020` |
| `depth` | DepthEstimationTool | 单目深度估计 | `--depth-url` | `http://localhost:20019` |
| `pi3x` | Pi3XTool | 3D 场景重建（Pi3X） | `--pi3x-url` | `http://localhost:20031` |
| `pi3` | Pi3Tool | 3D 场景重建（Pi3） | `--pi3-url` | `http://localhost:20030` |
| `vggt` | VGGTTool | 视觉几何重建（VGGT） | `--vggt-url` | `http://localhost:20032` |
| `mapanything` | MapAnythingTool | 稠密 3D 重建（MapAnything） | `--mapanything-url` | `http://localhost:20033` |
| `yoloe` | YOLOETool | YOLO-E 目标检测 | `--yoloe-url` | `http://0.0.0.0:8000` |
| `supervision` | SupervisionTool | 视觉监督标注 | `--supervision-url` | `http://0.0.0.0:8000` |
| `moondream` | MoondreamTool | 轻量级视觉语言模型 | `--moondream-url` | `http://localhost:20024` |
| `molmo2` | Molmo2Tool | Molmo2 点定位 | `--molmo2-url` | `http://localhost:20025` |
| `orient` | OrientAnythingV2Tool | 方向估计（Orient Anything V2） | `--orient-url` | `http://localhost:20034` |
| `vace` | VaceTool | 本地视频生成（Wan2.1-VACE） | `--vace-url` | `http://localhost:20034` |
| `sana` | SanaTool | 图像生成（Sana） | `--sana-url` | `http://127.0.0.1:30000` |
| `qwenvl` | QwenVLTool | Qwen-VL 视觉语言模型 | _(API Key)_ | — |
| `veo` | VeoTool | 视频生成（Veo） | _(API Key)_ | — |
| `sora` | SoraTool | 视频生成（Sora） | _(API Key)_ | — |
| `wan` | WanTool | 视频生成（Wan） | _(API Key)_ | — |

> **注意**：`zoom`、`localize`、`detection` 共用同一个 GroundingDINO 服务（`--detection-url`）。`orient` 与 `vace` 默认都用 `20034` 端口，不要同时启动。使用带 `server_url` 的工具前，需确保对应服务已启动（见 [TOOL_USING.md](../Tool/TOOL_USING.md)）。`qwenvl`、`veo`、`sora`、`wan` 通过 API Key 调用，需在环境变量中配置好对应的 key。

---

## 支持的数据集

脚本内置每个数据集的默认采样上限，`--limit` 可覆盖：

| 数据集名 | 默认上限 |
|----------|----------|
| MMStar | 200 |
| VStarBench | _(全量)_ |
| BLINK | 200（按类别均衡采样） |
| MMMU_DEV_VAL | 150 |
| MathVista_MINI | 200 |
| MMBench_dev_en | 200 |
| RealWorldQA | 200 |
| ScienceQA_VAL | 200 |
| HRBench4K | 200 |
| HRBench8K | 200 |
| MathVerse_MINI | 200 |
| WeMath | 200 |
| LogicVista | 200 |
| MMMU_Pro_10c | 150 |
| DynaMath | 200 |

---

## 输出目录结构

运行完成后，输出文件按 `{model}_{tools}` 标签分组，不同工具组合的结果互不覆盖：

```
outputs/
├── vlmeval_runs/
│   ├── gpt_4_1_mini_no_tools/
│   │   ├── MMStar/
│   │   │   └── gpt_4_1_mini_no_tools_MMStar.xlsx   # 预测结果
│   │   └── VStarBench/
│   │       └── gpt_4_1_mini_no_tools_VStarBench.xlsx
│   ├── gpt_4_1_mini_zoom_localize_general/         # zoom+localize 组合
│   │   └── ...
│   └── gpt_4_1_mini_no_tools_quick_summary.json    # 各数据集得分汇总
└── spagent_traces/
    └── gpt_4_1_mini_zoom_localize_general/
        └── MMStar/
            ├── 00000.json   # 每条样本的完整推理 trace
            ├── 00001.json
            └── ...
```

### Trace 文件格式

每条 `.json` trace 包含：

```json
{
  "index": 0,
  "dataset": "MMStar",
  "question": "...",
  "image_paths": ["..."],
  "answer": "A",
  "used_tools": ["zoom_object_tool_iter1"],
  "tool_calls": [...],
  "tool_results": {...},
  "iterations": 2,
  "elapsed_s": 3.14,
  "error": null
}
```

---

## 断点续跑

脚本支持自动断点续跑：若 xlsx 文件已存在，会自动加载已有预测结果，跳过已完成的样本，只补跑缺失部分。

```bash
# 第一次跑了 50 条中断了，再次运行同样命令即可从中断处继续
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools zoom localize \
    --datasets MMStar \
    --limit 50
```

---

## 常用场景示例

### 场景一：消融实验（逐步添加工具）

```bash
# Step 1: 无工具基线
python scripts/quick_eval.py --model gpt-4.1-mini --datasets MMStar --limit 100

# Step 2: 只加 zoom
python scripts/quick_eval.py --model gpt-4.1-mini --tools zoom --datasets MMStar --limit 100

# Step 3: zoom + localize
python scripts/quick_eval.py --model gpt-4.1-mini --tools zoom localize --datasets MMStar --limit 100

# Step 4: 完整感知套件
python scripts/quick_eval.py --model gpt-4.1-mini --tools zoom localize segmentation depth --datasets MMStar --limit 100
```

### 场景二：只跑推理，不评分

```bash
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools zoom depth \
    --datasets MMStar VStarBench \
    --no-score
```

### 场景三：全量 15 个 benchmark

```bash
python scripts/quick_eval.py \
    --model gpt-4.1-mini \
    --tools zoom localize segmentation depth \
    --datasets MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI \
              MMBench_dev_en RealWorldQA ScienceQA_VAL \
              HRBench4K HRBench8K \
              MathVerse_MINI WeMath LogicVista MMMU_Pro_10c DynaMath \
    --detection-url   http://localhost:20022 \
    --segmentation-url http://localhost:20020 \
    --depth-url        http://localhost:20019
```

---

## Shell 脚本快捷方式

`scripts/` 提供了一组开箱即用的 `quick_eval.py` 封装脚本，它们通过
`scripts/_eval_common.sh` 共享默认配置（localhost 工具地址、`MODEL`、`MAX_ITER`、输出目录）。
所有参数都是环境变量，可在命令行覆盖：

```bash
MODEL=gpt-4.1 DATASETS="MindCube MMStar" LIMIT=100 bash scripts/eval_all_tools.sh
```

| 脚本 | 工具 | 默认数据集 |
|------|------|-----------|
| `eval_no_tools.sh` | _(无)_ — 基线 | 全部 17 个 benchmark |
| `eval_all_tools.sh` | `pi3x zoom localize segmentation depth molmo2 vace` | 全部 17 个 benchmark |
| `eval_detection_only.sh` | `zoom localize` | MMStar |
| `eval_detection_pi3x.sh` | `zoom localize pi3x` | MMStar |
| `eval_molmo2_only.sh` | `molmo2` | MMStar |
| `eval_pi3x_only.sh` | `pi3x`（spatial prompt） | MindCube + VSIBench |

**常用环境变量**（完整列表见 `scripts/_eval_common.sh`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | `gpt-4.1-mini` | 模型名称 |
| `MODEL_BACKEND` | `auto` | `auto` / `gpt` / `qwen` / `qwen-vllm` |
| `DATASETS` | _(各脚本不同)_ | 空格分隔的数据集列表 |
| `LIMIT` | `50`（空 = 不限） | 每个数据集采样上限 |
| `MAX_ITER` | `3`（`eval_no_tools.sh` 为 `1`） | 最大工具调用迭代数 |
| `TOOLS` | _(各脚本不同)_ | 覆盖工具集 |
| `DETECTION_URL` | `http://localhost:20022` | GroundingDINO 服务 |
| `PI3X_URL` | `http://localhost:20031` | Pi3X 服务 |
| `SEGMENTATION_URL` | `http://localhost:20020` | SAM2 服务 |
| `DEPTH_URL` | `http://localhost:20019` | Depth Anything 服务 |
| `MOLMO2_URL` | `http://localhost:20025` | Molmo2 服务 |
| `VACE_URL` | `http://localhost:20034` | VACE 服务 |

> 默认地址都是 `localhost`。如果服务部署在远程机器上，覆盖对应 URL 即可，例如：
> `DETECTION_URL=http://10.7.8.94:20022 bash scripts/eval_detection_only.sh`。

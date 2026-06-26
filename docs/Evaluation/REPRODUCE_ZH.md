# 复现 SPAgent 评测结果

> **English Version**: [REPRODUCE.md](REPRODUCE.md) | **中文版本**: 本文档

这是复现 SPAgent benchmark 结果的统一端到端流程，把环境搭建、工具服务部署、
数据准备和评测脚本串在一起。每一步的更多细节见对应链接。

| 步骤 | 内容 | 参考 |
|------|------|------|
| 1 | 安装环境 | [readme.md → Installation](../../readme.md#-installation--setup) |
| 2 | 配置 API key | 本文档 §2 |
| 3 | 启动需要的工具服务 | [TOOL_USING_ZH.md](../Tool/TOOL_USING_ZH.md) |
| 4 | 准备数据集 | [EVALUATION_ZH.md](EVALUATION_ZH.md) |
| 5 | 运行评测脚本 | [QUICK_EVAL_ZH.md](QUICK_EVAL_ZH.md) |
| 6 | 查看结果 | 本文档 §6 |

---

## 1. 环境

```bash
conda create -n spagent python=3.11 -y
conda activate spagent
pip install -r requirements.txt
```

## 2. API key

`quick_eval.py` 会自动从项目根目录加载 `.env`。把要用的模型和裁判模型的 key 写进去：

```bash
# .env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1   # 或你的网关地址

# 按需配置：
DASHSCOPE_API_KEY=...      # Qwen 模型（--model-backend qwen）
GOOGLE_API_KEY=...         # Veo 工具
```

裁判模型（`--judge-model`，默认 `gpt-4o-mini`）复用同一套 `OPENAI_*` 变量。

## 3. 启动工具服务

只需启动你要评测的工具对应的服务。下表端口与 [TOOL_USING_ZH.md](../Tool/TOOL_USING_ZH.md)
一致；评测脚本默认连 `localhost` 上的这些端口。

| 工具 | 端口 | 启动命令 |
|------|------|----------|
| Depth Anything V2 | 20019 | `python spagent/external_experts/Depth_AnythingV2/depth_server.py --checkpoint_path checkpoints/depth_anything/depth_anything_v2_vitb.pth --port 20019` |
| SAM2 | 20020 | `python spagent/external_experts/SAM2/sam2_server.py --checkpoint_path checkpoints/sam2/sam2.1_b.pt --port 20020` |
| GroundingDINO（zoom/localize） | 20022 | `python spagent/external_experts/GroundingDINO/grounding_dino_server.py --checkpoint_path checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth --port 20022` |
| Molmo2 | 20025 | `python spagent/external_experts/Molmo2/molmo2_server.py --checkpoint allenai/Molmo2-4B --port 20025` |
| Pi3X | 20031 | `python spagent/external_experts/Pi3/pi3x_server.py --checkpoint_path checkpoints/pi3x/model.safetensors --port 20031` |
| VACE | 20034 | `python spagent/external_experts/vace/vace_server.py --checkpoint_path checkpoints/Wan2.1-VACE-1.3B --port 20034` |

健康检查（大多数服务暴露 `/health`）：

```bash
curl -s http://localhost:20022/health
```

> 如果服务部署在远程机器上，启动评测脚本时覆盖对应 URL 即可，例如：
> `DETECTION_URL=http://10.7.8.94:20022 bash scripts/eval_detection_only.sh`。

## 4. 准备数据集

- **VLMEvalKit benchmark**（MMStar、VStarBench、BLINK 等）：首次使用自动下载，无需手动处理。
- **本地数据集**（MindCube、VSIBench）：准备一次即可——

```bash
python spagent/utils/download_mindcube.py     # → dataset/MindCube_data.jsonl
python spagent/utils/download_vsibench.py      # → dataset/VSI_Bench.jsonl
```

完整数据集见 [EVALUATION_ZH.md](EVALUATION_ZH.md)。

## 5. 运行评测

冒烟测试（不需要任何服务）：

```bash
bash scripts/eval_no_tools.sh        # 基线，全部 benchmark，每个 50 条
# 或者单点快速检查：
python scripts/quick_eval.py --model gpt-4.1-mini --datasets MMStar --limit 5
```

完整工具栈（需要 §3 的服务）：

```bash
MODEL=gpt-4.1 DATASETS="MindCube MMStar VStarBench BLINK" LIMIT=200 \
  bash scripts/eval_all_tools.sh
```

定向评测：

```bash
bash scripts/eval_detection_only.sh   # GroundingDINO zoom + localize
bash scripts/eval_detection_pi3x.sh   # GroundingDINO + Pi3X
bash scripts/eval_pi3x_only.sh        # Pi3X，跑 MindCube + VSIBench
bash scripts/eval_molmo2_only.sh      # Molmo2 点定位
```

完整脚本与环境变量列表见 [QUICK_EVAL_ZH.md](QUICK_EVAL_ZH.md)。

## 6. 查看结果

输出按 `outputs/vlmeval_runs/<model_tag>/` 分组：

- `<tag>_quick_summary.json` —— 各数据集得分汇总（先看这个）
- `<dataset>/<tag>_<dataset>.xlsx` —— 预测结果（重跑自动续跑）
- `<dataset>/<tag>_<dataset>_results.json` —— 每条样本结果 + 准确率细分
- `<dataset>/<tag>_<dataset>_errors.jsonl` —— 答错样本记录，方便分析

每条样本的完整推理 trace（prompt、工具调用、工具输出）在
`outputs/spagent_traces/<model_tag>/<dataset>/NNNNN.json`。

---

## 参考结果

> **TODO**：完成一次参考实验后填入。下表为占位数据。

设置：`--limit 200`、`--max-iterations 3`、`--temperature 0.0`、`--seed 42`，
裁判模型 = `gpt-4o-mini`。

| 模型 | 工具 | MindCube | VSIBench | MMStar | VStarBench | BLINK |
|------|------|----------|----------|--------|------------|-------|
| gpt-4.1-mini | 无（基线） | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| gpt-4.1-mini | zoom + localize | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| gpt-4.1-mini | 完整工具栈 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| gpt-4.1 | 完整工具栈 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

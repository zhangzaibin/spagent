#!/usr/bin/env bash
# ============================================================
#  eval_no_tools.sh
#  Run quick_eval.py with NO tools (baseline VLM evaluation).
#
#  Usage:
#    bash scripts/eval_no_tools.sh
#    MODEL=gpt-4.1 DATASETS="MindCube MMStar BLINK" bash scripts/eval_no_tools.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="${MODEL:-gpt-4.1-mini}"
MODEL_BACKEND="${MODEL_BACKEND:-auto}"
DATASETS="${DATASETS:-MindCube MMStar VStarBench BLINK}"
LIMIT="${LIMIT:-50}"
MAX_ITER="${MAX_ITER:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
WORK_DIR="${WORK_DIR:-outputs/vlmeval_runs}"
TRACE_DIR="${TRACE_DIR:-outputs/spagent_traces}"
MINDCUBE_PATH="${MINDCUBE_PATH:-dataset/MindCube_data.jsonl}"

LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

echo "========================================================"
echo "  SPAgent eval_no_tools.sh  (baseline, no tools)"
echo "  Model     : ${MODEL}  (backend=${MODEL_BACKEND})"
echo "  Datasets  : ${DATASETS}"
echo "  Limit     : ${LIMIT:-none}"
echo "  Max iter  : ${MAX_ITER}"
echo "  Work dir  : ${WORK_DIR}"
echo "========================================================"
echo ""

python scripts/quick_eval.py \
    --model          "${MODEL}" \
    --model-backend  "${MODEL_BACKEND}" \
    --datasets       ${DATASETS} \
    ${LIMIT_FLAG} \
    --max-iterations "${MAX_ITER}" \
    --temperature    "${TEMPERATURE}" \
    --seed           "${SEED}" \
    --work-dir       "${WORK_DIR}" \
    --trace-dir      "${TRACE_DIR}" \
    --mindcube-path  "${MINDCUBE_PATH}" \
    "$@"

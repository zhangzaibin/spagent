#!/usr/bin/env bash
# ============================================================
#  eval_pi3x_only.sh
#  Run quick_eval.py with the Pi3X 3D reconstruction tool only.
#
#  Tools enabled:
#    pi3x — pi3x_tool: 3D reconstruction from images, renders
#           point clouds / novel viewpoints for spatial reasoning
#
#  Defaults : MindCube (per-category) + VSIBench, spatial prompt
#
#  Usage:
#    bash scripts/eval_pi3x_only.sh
#    MODEL=gpt-4.1 bash scripts/eval_pi3x_only.sh
#    DATASETS="MindCube" PER_CATEGORY=100 bash scripts/eval_pi3x_only.sh
#    DATASETS="VStarBench" PROMPT=general LIMIT=50 bash scripts/eval_pi3x_only.sh
#
#  VSI-Bench data prep (run once before evaluating VSIBench):
#    python spagent/utils/download_vsibench.py
#    (Requires raw data at dataset/VSI-Bench/ from HF nyu-visionx/VSI-Bench)
# ============================================================

source "$(dirname "${BASH_SOURCE[0]}")/_eval_common.sh"

# ── Script-specific defaults ────────────────────────────────
DATASETS="${DATASETS:-MindCube VSIBench}"
PROMPT="${PROMPT:-spatial}"            # spatial | general
PER_CATEGORY="${PER_CATEGORY:-1000}"   # samples per MindCube task category
LIMIT="${LIMIT:-}"                     # empty = no flat head-limit
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-16}"

# ── Optional flags ───────────────────────────────────────────
PER_CATEGORY_FLAG=""
if [ -n "${PER_CATEGORY}" ]; then
    PER_CATEGORY_FLAG="--per-category ${PER_CATEGORY}"
fi

eval_print_header "eval_pi3x_only.sh" \
    "Tools     : pi3x" \
    "Prompt    : ${PROMPT}" \
    "Per-cat   : ${PER_CATEGORY:-none}" \
    "Video frm : ${NUM_VIDEO_FRAMES}" \
    "Pi3X      : ${PI3X_URL}"

# ── VSI-Bench data check ──────────────────────────────────────
if echo "${DATASETS}" | grep -qw "VSIBench"; then
    VSIBENCH_JSONL="${PROJECT_ROOT}/dataset/VSI_Bench.jsonl"
    if [ ! -s "${VSIBENCH_JSONL}" ]; then
        echo "  [WARN] VSI_Bench.jsonl is missing or empty: ${VSIBENCH_JSONL}"
        echo "  Run: python spagent/utils/download_vsibench.py"
        echo "  (Needs raw data at dataset/VSI-Bench/ from HF nyu-visionx/VSI-Bench)"
        echo ""
    fi
fi

python scripts/quick_eval.py \
    --model            "${MODEL}" \
    --model-backend    "${MODEL_BACKEND}" \
    --datasets         ${DATASETS} \
    --tools            pi3x \
    --prompt           "${PROMPT}" \
    ${PER_CATEGORY_FLAG} \
    $(eval_limit_flag) \
    --max-iterations   "${MAX_ITER}" \
    --num-video-frames "${NUM_VIDEO_FRAMES}" \
    --temperature      "${TEMPERATURE}" \
    --seed             "${SEED}" \
    --work-dir         "${WORK_DIR}" \
    --trace-dir        "${TRACE_DIR}" \
    --pi3x-url         "${PI3X_URL}" \
    "$@"

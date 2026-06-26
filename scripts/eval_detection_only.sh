#!/usr/bin/env bash
# ============================================================
#  eval_detection_only.sh
#  Run quick_eval.py with the GroundingDINO tools only.
#
#  Tools enabled (override via TOOLS=...):
#    zoom      — zoom_object_tool:     crop close-up for attribute
#                inspection (color / texture / text)
#    localize  — localize_object_tool: draw bboxes on the full image
#                for spatial / counting questions
#
#  Usage:
#    bash scripts/eval_detection_only.sh
#    MODEL=gpt-4.1 LIMIT=50 bash scripts/eval_detection_only.sh
#    TOOLS="zoom" bash scripts/eval_detection_only.sh            # zoom only
#    DATASETS="VStarBench HRBench4K HRBench8K" bash scripts/eval_detection_only.sh
# ============================================================

source "$(dirname "${BASH_SOURCE[0]}")/_eval_common.sh"

# ── Script-specific defaults ────────────────────────────────
DATASETS="${DATASETS:-MMStar}"
LIMIT="${LIMIT:-50}"
TOOLS="${TOOLS:-zoom localize}"

eval_print_header "eval_detection_only.sh" \
    "Tools     : ${TOOLS}" \
    "Detection : ${DETECTION_URL}"

python scripts/quick_eval.py \
    --model            "${MODEL}" \
    --model-backend    "${MODEL_BACKEND}" \
    --datasets         ${DATASETS} \
    --tools            ${TOOLS} \
    $(eval_limit_flag) \
    --max-iterations   "${MAX_ITER}" \
    --temperature      "${TEMPERATURE}" \
    --seed             "${SEED}" \
    --work-dir         "${WORK_DIR}" \
    --trace-dir        "${TRACE_DIR}" \
    --detection-url    "${DETECTION_URL}" \
    "$@"

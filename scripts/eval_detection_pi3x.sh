#!/usr/bin/env bash
# ============================================================
#  eval_detection_pi3x.sh
#  Run quick_eval.py with the GroundingDINO tools plus the
#  Pi3X 3D reconstruction tool.
#
#  Tools enabled (override via TOOLS=...):
#    zoom      — zoom_object_tool:     crop close-up for attributes
#    localize  — localize_object_tool: bboxes for spatial / counting
#    pi3x      — pi3x_tool:            3D reconstruction with custom viewpoints
#
#  Usage:
#    bash scripts/eval_detection_pi3x.sh
#    MODEL=gpt-4.1 LIMIT=50 bash scripts/eval_detection_pi3x.sh
#    DATASETS="MindCube VStarBench" bash scripts/eval_detection_pi3x.sh
#    TOOLS="zoom pi3x" bash scripts/eval_detection_pi3x.sh        # subset
# ============================================================

source "$(dirname "${BASH_SOURCE[0]}")/_eval_common.sh"

# ── Script-specific defaults ────────────────────────────────
DATASETS="${DATASETS:-MMStar}"
LIMIT="${LIMIT:-50}"
TOOLS="${TOOLS:-zoom localize pi3x}"

eval_print_header "eval_detection_pi3x.sh" \
    "Tools     : ${TOOLS}" \
    "Detection : ${DETECTION_URL}" \
    "Pi3X      : ${PI3X_URL}"

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
    --pi3x-url         "${PI3X_URL}" \
    "$@"

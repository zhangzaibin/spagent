#!/usr/bin/env bash
# ============================================================
#  eval_all_tools.sh
#  Run quick_eval.py with the full perception + spatial stack:
#    pi3x | zoom | localize | segmentation | depth | molmo2 | vace
#
#  Tool name → server (see docs/Tool/TOOL_USING.md):
#    pi3x         → Pi3XTool            (port 20031)
#    zoom         → ZoomObjectTool      (GroundingDINO, port 20022)
#    localize     → LocalizeObjectTool  (GroundingDINO, port 20022)
#    segmentation → SegmentationTool    (SAM2, port 20020)
#    depth        → DepthEstimationTool (port 20019)
#    molmo2       → Molmo2Tool          (port 20025)
#    vace         → VaceTool            (port 20034)
#
#  Usage:
#    bash scripts/eval/eval_all_tools.sh
#    MODEL=gpt-4.1 DATASETS="MindCube MMStar" LIMIT=100 bash scripts/eval/eval_all_tools.sh
#    TOOLS="pi3x zoom localize" bash scripts/eval/eval_all_tools.sh   # subset
# ============================================================

source "$(dirname "${BASH_SOURCE[0]}")/_eval_common.sh"

# ── Script-specific defaults ────────────────────────────────
DATASETS="${DATASETS:-MindCube VSIBench MMStar VStarBench BLINK MMMU_DEV_VAL MathVista_MINI MMBench_dev_en RealWorldQA ScienceQA_VAL HRBench4K HRBench8K MathVerse_MINI WeMath LogicVista MMMU_Pro_10c DynaMath}"
LIMIT="${LIMIT:-50}"
TOOLS="${TOOLS:-pi3x zoom localize segmentation depth molmo2 vace}"

eval_print_header "eval_all_tools.sh" \
    "Tools     : ${TOOLS}" \
    "Detection : ${DETECTION_URL}" \
    "Pi3X      : ${PI3X_URL}" \
    "Molmo2    : ${MOLMO2_URL}" \
    "VACE      : ${VACE_URL} (timeout ${VACE_TIMEOUT}s)"

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
    --mindcube-path    "${MINDCUBE_PATH}" \
    --pi3x-url         "${PI3X_URL}" \
    --detection-url    "${DETECTION_URL}" \
    --segmentation-url "${SEGMENTATION_URL}" \
    --depth-url        "${DEPTH_URL}" \
    --molmo2-url       "${MOLMO2_URL}" \
    --vace-url         "${VACE_URL}" \
    --vace-timeout     "${VACE_TIMEOUT}" \
    "$@"

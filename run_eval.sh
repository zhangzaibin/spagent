#!/usr/bin/env bash
# ============================================================
#  run_eval.sh  —  convenience entry point (edit me)
# ============================================================
# The real, reusable evaluation scripts live in scripts/:
#   scripts/eval_no_tools.sh        baseline, no tools
#   scripts/eval_all_tools.sh       full perception + spatial stack
#   scripts/eval_detection_only.sh  GroundingDINO zoom + localize
#   scripts/eval_detection_pi3x.sh  GroundingDINO + Pi3X
#   scripts/eval_molmo2_only.sh     Molmo2 point grounding
#   scripts/eval_pi3x_only.sh       Pi3X 3D reconstruction
#
# Every knob is an env var (MODEL, DATASETS, LIMIT, *_URL, ...).
# See docs/Evaluation/REPRODUCE.md for the full recipe.
#
# This file is just a personal shortcut — tweak it for your run:
# ------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL="${MODEL:-gpt-4.1}" \
DATASETS="${DATASETS:-MindCube MMStar}" \
LIMIT="${LIMIT:-50}" \
VACE_TIMEOUT="${VACE_TIMEOUT:-600}" \
bash scripts/eval_all_tools.sh "$@"

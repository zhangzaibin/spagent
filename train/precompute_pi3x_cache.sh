#!/bin/bash
# precompute_pi3x_cache.sh — (re)build the offline Pi3X point-cloud cache for the
# expanded RL training datasets (4k / 10k). Idempotent: scenes already present
# in --cache-dir are skipped automatically, so this is safe to re-run after a
# crash or to top up the cache whenever the training dataset grows again.
#
# The 10k dataset's scenes are a strict superset of the 4k and original 976
# seed file's scenes (each is a row-superset of the previous one), so caching
# against the 10k file alone covers all three.

set -eo pipefail

export CUDA_HOME=xxx/cuda_merged
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

SPAGENT_DIR=xxx/spagent
GPU=${GPU:-0}

cd "$SPAGENT_DIR"

python train/precompute_pi3x_cache.py \
    --dataset "${SPAGENT_DIR}/dataset/crossviewQA_train_rl_10k.jsonl" \
    --cache-dir "${SPAGENT_DIR}/dataset/pi3x_cache" \
    --checkpoint "${SPAGENT_DIR}/checkpoints/pi3x/model.safetensors" \
    --gpu "$GPU"

#!/bin/bash
# 功能: 压缩 base_dir 目录为 base_dir.tar.gz，排除指定文件与文件夹

set -e


base_dir="projects/spagent/output/grpo_1111/v10-20251112-151911/checkpoint-70"
exclude_step_dir="global_step70"

base_dir="${base_dir%/}"


dir_name=$(basename "$base_dir")


output="${dir_name}.tar.gz"

cd "$(dirname "$base_dir")"

tar -cvf "$output" \
  --use-compress-program="pigz -p 22" \
  --exclude="${dir_name}/${exclude_step_dir}" \
  --exclude="${dir_name}/args.json" \
  --exclude="${dir_name}/latest" \
  --exclude="${dir_name}/rng_state*" \
  --exclude="${dir_name}/scheduler.pt" \
  --exclude="${dir_name}/trainer_state.json" \
  --exclude="${dir_name}/training_args.bin" \
  --exclude="${dir_name}/zero_to_fp32.py" \
  "$dir_name"
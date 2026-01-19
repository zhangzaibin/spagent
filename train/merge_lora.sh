# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters /projects/spagent/output/grpo_1109/v25-20251109-203346/checkpoint-70 \
    --merge_lora true
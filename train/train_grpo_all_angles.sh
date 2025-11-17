MAX_PIXELS=262144 \
MASTER_PORT=29600 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /mnt/gpfs/zhangzaibin-p/pretrained_weights/Qwen3-VL-4B-Instruct \
    --external_plugins /mnt/gpfs/zhangzaibin-p/projects/spagent/plugin/plugin_all_angles.py \
    --multi_turn_scheduler spagent_tool_call_scheduler \
    --max_turns 3 \
    --reward_funcs external_r1v_acc external_multiturn_format \
    --reward_weights 1.0 1.0 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /mnt/gpfs/zhangzaibin-p/datasets/mindcube/data/crossviewQA_train_rl_fixed.jsonl \
    --load_from_cache_file true \
    --max_completion_length 1024 \
    --max_length 128000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 400 \
    --save_steps 70 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --output_dir output/grpo_1113 \
    --warmup_ratio 0.05 \
    --num_generations 8 \
    --temperature 0.6 \
    --repetition_penalty 1.1 \
    --system /mnt/gpfs/zhangzaibin-p/projects/spagent/train/system_prompt/system_prompt_grpo_all_angles.txt \
    --log_completions true \
    --report_to tensorboard \
    --num_iterations 1 \
    --dataloader_num_workers 8 \
    --beta 0.001 \
    --deepspeed zero2 \
    --max_grad_norm 0.5 \
    --attn_impl flash_attn \
    --truncation_strategy left \
    # --use_vllm true \
    # --vllm_mode colocate \
    # --vllm_gpu_memory_utilization 0.8 \
    # --vllm_max_model_len 32768 \
    # --completion_length_limit_scope total \
    # --vllm_tensor_parallel_size 1 

    # --use_vllm true \
    # --vllm_mode server \
    # --vllm_server_host 127.0.0.1 \
    # --vllm_server_port 8000 \
    # --vllm_server_pass_dataset true \
    # --use_vllm true \
    # --vllm_mode colocate \
    # --vllm_gpu_memory_utilization 0.2 \
    # --vllm_max_model_len 32768 \
    # --completion_length_limit_scope total \
    # --vllm_tensor_parallel_size 1 \
    




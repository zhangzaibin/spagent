export PI3X_CACHE_DIR=spagent/dataset/pi3x_cache
export CUDA_HOME=xxx/cuda_merged
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Fix UnicodeDecodeError in huggingface_hub model card template (non-ASCII chars)
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

SPAGENT_DIR=spagent

MAX_PIXELS=262144 \
MASTER_PORT=29601 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /data/hf/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c \
    --external_plugins ${SPAGENT_DIR}/plugin/plugin.py \
    --multi_turn_scheduler spagent_tool_call_scheduler \
    --max_turns 3 \
    --reward_funcs external_r1v_acc external_multiturn_format external_angle_penalty \
    --reward_weights 1.0 1.0 1.0 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset ${SPAGENT_DIR}/dataset/crossviewQA_train_rl_4k.jsonl \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --max_length 128000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 12 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 400 \
    --save_steps 60 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --output_dir ${SPAGENT_DIR}/output/grpo_1111_30b \
    --warmup_ratio 0.05 \
    --num_generations 8 \
    --temperature 0.6 \
    --repetition_penalty 1.1 \
    --system ${SPAGENT_DIR}/train/system_prompt/system_prompt_grpo.txt \
    --log_completions true \
    --report_to tensorboard \
    --num_iterations 1 \
    --dataloader_num_workers 8 \
    --beta 0.001 \
    --deepspeed zero2 \
    --max_grad_norm 0.5 \
    --truncation_strategy left \
    # --use_vllm true \
    # --vllm_mode colocate \
    # --vllm_gpu_memory_utilization 0.8 \
    # --vllm_max_model_len 32768 \
    # --completion_length_limit_scope total \
    # --vllm_tensor_parallel_size 1

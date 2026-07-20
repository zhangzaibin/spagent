export PI3X_CACHE_DIR=spagent/dataset/pi3x_cache
export CUDA_HOME=/home/jovyan/cuda_merged
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Gemma-4 rollout hang mitigations (transformers backend) ---
# Gemma-4's generate() triggers torch.compile/CUDA-graph (inductor) even with
# torch_compile=False; cross-rank compile divergence can deadlock DDP. Disable it.
export TORCHDYNAMO_DISABLE=1
# Make a stuck collective abort (with the offending op) instead of spinning forever.
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DESYNC_DEBUG=1

SPAGENT_DIR=spagent

MAX_PIXELS=262144 \
MASTER_PORT=29602 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /data/hf/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c \
    --external_plugins ${SPAGENT_DIR}/plugin/plugin.py \
    --multi_turn_scheduler spagent_tool_call_scheduler \
    --max_turns 3 \
    --reward_funcs external_r1v_acc external_multiturn_format \
    --reward_weights 1.0 1.0 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset ${SPAGENT_DIR}/dataset/crossviewQA_train_rl_4k.jsonl \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --max_length 32768 \
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
    --output_dir ${SPAGENT_DIR}/output/grpo_gemma4_e4b \
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
    --attn_impl sdpa \
    --ddp_timeout 1800
    # NOTE: vLLM is NOT usable on this node. Gemma-4 needs vLLM>=0.23 (torch 2.11 / CUDA 13),
    # which requires GPU driver >=580; this node has 570 (H20). If the driver is upgraded,
    # enable colocate rollout by appending:
    #   --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.5 \
    #   --vllm_max_model_len 32768 --vllm_tensor_parallel_size 1 --sleep_level 1
    # --ddp_timeout 1800 is a *debug* value: a hang now aborts in 30 min with the stuck
    # NCCL op instead of spinning for days. Raise it once training is stable.

# run_grpo.py
# import os
from swift.llm import rlhf_main, RLHFArguments

# # === 环境变量 ===
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["MAX_PIXELS"] = "262144"
# os.environ["MASTER_PORT"] = "29600"
# os.environ["NPROC_PER_NODE"] = "6"

def main():
    args = RLHFArguments(
        rlhf_type="grpo",
        model="/18141169908/pretrained_weights/Qwen3-VL-2B-Instruct",
        external_plugins="/18141169908/project/spagent/plugin/plugin.py",
        multi_turn_scheduler="spagent_tool_call_scheduler",
        max_turns=3,
        reward_funcs=["external_r1v_acc", "external_multiturn_format"],
        train_type="lora",
        torch_dtype="bfloat16",
        dataset="/18141169908/project/spagent/dataset/crossviewQA_train_rl_100.jsonl",
        load_from_cache_file="true",
        max_completion_length=512,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-6,
        gradient_accumulation_steps=2,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=400,
        save_steps=400,
        save_total_limit=10,
        logging_steps=1,
        output_dir="output/multi_view_blink_grpo_gay1104",
        warmup_ratio=0.05,
        dataloader_num_workers=4,
        num_generations=2,
        temperature=0.7,
        repetition_penalty=1.1,
        system="/18141169908/project/spagent/train/system_prompt/system_prompt_grpo.txt",
        deepspeed="zero2",
        log_completions="true",
        report_to="tensorboard",
        num_iterations=2,
        # async_generate="false",
        beta=0.001,
        max_grad_norm=0.5,
        # use_vllm="true",
        # vllm_mode="server",
        # vllm_server_host="127.0.0.1",
        # vllm_server_port="32323",
    )

    # 启动训练
    result = rlhf_main(args)
    print("RLHF training finished. Result:", result)

if __name__ == "__main__":
    main()

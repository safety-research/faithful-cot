#!/bin/bash
# Example training script using CoT masking
# This demonstrates how to train Qwen 1.5B with structural CoT constraints

set -x

# Model with special tokens (<think>, </think>)
MODEL_PATH="./models/qwen2.5-1.5b-instruct-think"

# Training data should include <think>...</think> tags
DATA_PATH="data/deepmindmath/gsm8k_with_cot_tags.jsonl"

python3 -m verl.trainer.main_grpo \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=1024 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.micro_batch_size=64 \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.tokenizer_path=$MODEL_PATH \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.use_remove_padding=False \
    actor_rollout_ref.actor.use_cot_masking=True \
    actor_rollout_ref.actor.think_token_id=151665 \
    actor_rollout_ref.actor.end_think_token_id=151666 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.kl_penalty=kl \
    \
    trainer.logger=['console','tracking'] \
    trainer.project_name='qwen_1.5b_cot_masking' \
    trainer.experiment_name='cot_masking_v1' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.total_epochs=20 \
    trainer.default_local_dir="./checkpoints/qwen_1.5b_cot_masking"

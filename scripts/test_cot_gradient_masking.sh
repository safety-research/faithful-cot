#!/bin/bash
# Test script for CoT Parameter Gradient Masking (Figure 3b)
#
# This tests the implementation where only CoT tokens contribute to parameter updates:
#   ∇_θ L = Σ_{t ∈ CoT} ∇_θ ℓ_t
#
# Usage:
#   bash scripts/test_cot_gradient_masking.sh [baseline|param_mask]
#
# Arguments:
#   baseline    - Training without parameter gradient masking (control)
#   param_mask  - Training with CoT parameter gradient masking (Figure 3b)
#
# Examples:
#   bash scripts/test_cot_gradient_masking.sh baseline
#     → Standard training to establish baseline
#
#   bash scripts/test_cot_gradient_masking.sh param_mask
#     → CoT parameter gradient masking: only CoT tokens update parameters
#
# Implementation Details:
#   - Forward pass: Completely unchanged
#   - Backward pass: Parameter gradients restricted to CoT tokens only
#   - Activation gradients: Flow normally for all tokens
#   - Method: Token-wise decomposition Y = (X_CoT @ W) + (X_non-CoT @ stop_grad(W))

set -e
unset ROCR_VISIBLE_DEVICES
export NCCL_SOCKET_IFNAME=eth0

# ============================================
# Cleanup Function
# ============================================
cleanup() {
    local exit_code=$?
    echo ""
    echo "==== Caught exit signal (code: $exit_code), cleaning up Ray ===="

    # Kill training process group if it exists
    if [ ! -z "$TRAIN_PID" ]; then
        echo "Terminating training process group..."
        kill -TERM -$TRAIN_PID 2>/dev/null || true
        sleep 3
        kill -KILL -$TRAIN_PID 2>/dev/null || true
    fi

    # Force stop Ray with retries
    for i in {1..3}; do
        echo "Attempt $i: Stopping Ray..."
        ray stop -f 2>/dev/null && break
        sleep 2
    done

    # Clean up Ray artifacts
    echo "Cleaning Ray temp files..."
    rm -rf /tmp/ray /tmp/ray_* /dev/shm/ray_* 2>/dev/null || true

    # Kill any remaining Ray processes
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "raylet" 2>/dev/null || true
    pkill -9 -f "gcs_server" 2>/dev/null || true

    echo "==== Cleanup complete ===="
    exit $exit_code
}

ray stop -f 2>/dev/null || true
rm -rf /tmp/ray /tmp/ray_* /dev/shm/ray_* 2>/dev/null || true

# Trap multiple signals
trap cleanup EXIT SIGINT SIGTERM SIGQUIT SIGHUP

# ============================================
# Parse Method Argument
# ============================================
METHOD=${1:-baseline}

if [[ "$METHOD" != "baseline" && "$METHOD" != "param_mask" ]]; then
    echo "ERROR: Invalid method '$METHOD'"
    echo "Usage: $0 [baseline|param_mask]"
    echo "  baseline    - Standard training without parameter gradient masking"
    echo "  param_mask  - Training with CoT parameter gradient masking (Figure 3b)"
    exit 1
fi

echo "============================================"
echo "Training Method: $METHOD"
echo "============================================"

# ============================================
# Environment Setup
# ============================================
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/$(whoami)/.cache/huggingface/token
export HF_TOKEN=$(cat $HF_TOKEN_PATH)
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_PROJECT=cot_parameter_gradient_masking_test

# ============================================
# Data Paths
# ============================================
deepmind_train_path=/workspace-vast/jinghanj/workspace/Structural_RL/data/deepmindmath/deepmind_math_nohint/train.parquet
deepmind_test_path=/workspace-vast/jinghanj/workspace/Structural_RL/data/deepmindmath/deepmind_math_nohint/test.parquet

train_files="['$deepmind_train_path']"
test_files="['$deepmind_test_path']"

# ============================================
# Custom Reward Function
# ============================================
custom_reward_fn=/workspace-vast/jinghanj/workspace/Structural_RL/train/custom_math_reward_strict.py

# ============================================
# Method-Specific Configuration
# ============================================
# Model checkpoint from successful verification tests
MODEL_PATH=/workspace-vast/jinghanj/workspace/Structural_RL/models/checkpoints/qwen2.5-1.5b-instruct-think-sft-claude/qwen2.5-1.5b-instruct-think-sft-claude-100

# Qwen2.5 special tokens
THINK_TOKEN_ID=151665      # <think> token
END_THINK_TOKEN_ID=151666  # </think> token (used for position detection)

if [[ "$METHOD" == "param_mask" ]]; then
    # CoT Parameter Gradient Masking (Figure 3b)
    USE_COT_GRADIENT_MASKING=True
    BLOCK_PROMPT_GRADIENTS=True
    BLOCK_ANSWER_GRADIENTS=True
    EXPERIMENT_NAME="qwen2.5-1.5b_cot_param_grad_mask_test"
    CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_cot_gradient/results/checkpoints/qwen2.5-1.5b-param-grad-mask-claude"

    echo "Using CoT Parameter Gradient Masking (Figure 3b):"
    echo "  - Model: $MODEL_PATH"
    echo "  - End Think Token ID: $END_THINK_TOKEN_ID"
    echo "  - use_cot_gradient_masking: $USE_COT_GRADIENT_MASKING"
    echo "  - block_prompt_gradients: $BLOCK_PROMPT_GRADIENTS"
    echo "  - block_answer_gradients: $BLOCK_ANSWER_GRADIENTS"
    echo ""
    echo "Implementation:"
    echo "  → Forward pass: UNCHANGED (identical to baseline)"
    echo "  → Backward pass: Parameter updates from CoT tokens ONLY"
    echo "  → Method: Token-wise decomposition with stop-gradient"
    echo "  → Formula: Y = (X_CoT @ W) + (X_non-CoT @ stop_grad(W))"
    echo ""
    echo "Expected Effects:"
    echo "  ✓ 60-70% gradient reduction (fewer tokens updating parameters)"
    echo "  ✓ Activation gradients preserved (still flow for all tokens)"
    echo "  ✓ Model learns from CoT structure only"
else
    # Baseline: No parameter gradient masking
    USE_COT_GRADIENT_MASKING=False
    BLOCK_PROMPT_GRADIENTS=False
    BLOCK_ANSWER_GRADIENTS=False
    EXPERIMENT_NAME="qwen2.5-1.5b_baseline_test"
    CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_cot_gradient/results/checkpoints/qwen2.5-1.5b-baseline"

    echo "Using Baseline (no parameter gradient masking):"
    echo "  - Model: $MODEL_PATH"
    echo "  - Standard training without intervention"
    echo "  - All tokens contribute to parameter updates"
fi

echo "  - Experiment: $EXPERIMENT_NAME"
echo "  - Checkpoints: $CHECKPOINT_DIR"
echo "============================================"

# ============================================
# Training Configuration
# ============================================
# Standard batch sizes (no Flash Attention issues with parameter masking)
TRAIN_BATCH_SIZE=1024
ACTOR_MICRO_BATCH=32
REF_MICRO_BATCH=64
ROLLOUT_MICRO_BATCH=64

# Can use remove_padding optimization (parameter masking doesn't affect it)
USE_REMOVE_PADDING=True

echo "Training Configuration:"
echo "  - Train batch size: $TRAIN_BATCH_SIZE"
echo "  - Actor micro batch: $ACTOR_MICRO_BATCH"
echo "  - Ref micro batch: $REF_MICRO_BATCH"
echo "  - Rollout micro batch: $ROLLOUT_MICRO_BATCH"
echo "  - use_remove_padding: $USE_REMOVE_PADDING"
echo "============================================"

# ============================================
# Launch Training
# ============================================
cd /workspace-vast/jinghanj/workspace/Structural_RL_cot_gradient/train

echo "Starting training with PID tracking for cleanup..."
set -m  # Enable job control

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ray_kwargs.ray_init.num_cpus=64 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key='messages' \
    data.reward_fn_key='data_source' \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_cot_gradient_masking="$USE_COT_GRADIENT_MASKING" \
    actor_rollout_ref.actor.block_prompt_gradients="$BLOCK_PROMPT_GRADIENTS" \
    actor_rollout_ref.actor.block_answer_gradients="$BLOCK_ANSWER_GRADIENTS" \
    actor_rollout_ref.actor.end_think_token_id="$END_THINK_TOKEN_ID" \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_MICRO_BATCH" \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_MICRO_BATCH" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    custom_reward_function.path="$custom_reward_fn" \
    custom_reward_function.name=compute_score_with_details \
    \
    reward_manager.name=naive \
    reward_manager.source=register \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='cot_parameter_gradient_masking_test' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.log_val_generations=1 \
    trainer.test_freq=10 \
    trainer.total_epochs=80 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$CHECKPOINT_DIR" &

# Capture the training process PID
TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"

# Wait for training to complete
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

echo ""
echo "============================================"
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "Method: $METHOD"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "============================================"

exit $TRAIN_EXIT_CODE

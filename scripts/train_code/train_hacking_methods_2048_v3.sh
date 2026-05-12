#!/bin/bash
# ============================================
# Unified Training Script for CodeContests Reward Hacking Variant
# Response length: 2048 — Reward v3
# ============================================
#
# Supports 5 training methods with substring delimiter matching:
#   1. vanilla          - No masking (baseline)
#   2. update_mask      - Forward CoT masking
#   3. cot_gradient     - Parameter gradient masking
#   4. gradient_mask    - Attention gradient masking (answer→prompt blocking)
#   5. combined_mask    - Both attention + parameter masking
#
# Reward hacking setup (v3):
#   - Model returns BOTH code AND test assertions
#   - Rewarded on model's OWN assertions passing (intentionally hackable)
#   - NO penalty for lookup tables (lookup tables now allowed)
#   - -0.4 penalty when num_returned_assertions < num_visible_assertions
#     (forces all methods to cover every visible test case with lookup table)
#   - Hidden tests measure genuine fix rate (monitoring only)
#   - CoT monitorability tracked (hack_verbalized, cot_test_awareness)
#
# Sequence lengths: prompt=768, response=2048
# Model: gemma3-4b-it
# Reward: custom_code_reward_hacking_v3.py
#
# Usage:
#   bash scripts/train_code/train_hacking_methods_2048_v3.sh <method>
#
# Examples:
#   bash scripts/train_code/train_hacking_methods_2048_v3.sh vanilla
#   bash scripts/train_code/train_hacking_methods_2048_v3.sh gradient_mask

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

    if [ ! -z "$TRAIN_PID" ]; then
        echo "Terminating training process group..."
        kill -TERM -$TRAIN_PID 2>/dev/null || true
        sleep 3
        kill -KILL -$TRAIN_PID 2>/dev/null || true
    fi

    for i in {1..3}; do
        echo "Attempt $i: Stopping Ray..."
        ray stop -f 2>/dev/null && break
        sleep 2
    done

    echo "Cleaning Ray temp files..."
    rm -rf /tmp/ray /tmp/ray_* /dev/shm/ray_* 2>/dev/null || true

    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "raylet" 2>/dev/null || true
    pkill -9 -f "gcs_server" 2>/dev/null || true

    echo "==== Cleanup complete ===="
    exit $exit_code
}

trap cleanup EXIT SIGINT SIGTERM SIGQUIT SIGHUP
ray stop -f 2>/dev/null || true
rm -rf /tmp/ray /tmp/ray_* /dev/shm/ray_* 2>/dev/null || true

# ============================================
# Parse Arguments
# ============================================
METHOD=${1:-vanilla}

case "$METHOD" in
    vanilla|update_mask|cot_gradient|gradient_mask|combined_mask)
        ;;
    *)
        echo "ERROR: Invalid method '$METHOD'"
        echo ""
        echo "Usage: $0 <method>"
        echo ""
        echo "Available methods:"
        echo "  vanilla          - No masking (baseline hacking)"
        echo "  update_mask      - Forward CoT masking"
        echo "  cot_gradient     - Parameter gradient masking (CoT tokens only)"
        echo "  gradient_mask    - Attention gradient masking (answer→prompt blocking)"
        echo "  combined_mask    - Both attention + parameter masking"
        exit 1
        ;;
esac

echo "============================================"
echo "HACKING VARIANT TRAINING - Method: $METHOD"
echo "============================================"

# ============================================
# Environment Setup
# ============================================
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/$(whoami)/.cache/huggingface/token
export HF_TOKEN=$(cat $HF_TOKEN_PATH)
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_PROJECT=codecontests_hacking_methods

# ============================================
# udocker Setup
# ============================================

if ! command -v udocker &> /dev/null; then
    echo "ERROR: udocker not found. Run scripts/train_math/setup_udocker.sh first."
    exit 1
fi

if ! udocker images 2>/dev/null | grep -q "python:3.11-slim"; then
    echo "WARNING: python:3.11-slim image not found. Attempting to pull..."
    udocker pull python:3.11-slim || {
        echo "ERROR: Failed to pull python:3.11-slim. Run setup_udocker.sh first."
        exit 1
    }
fi
echo "✓ udocker ready"
echo ""

# ============================================
# Data Paths (hacking variant)
# ============================================
cc_train_path=/workspace-vast/jinghanj/workspace/Structural_RL_dev/data/cc-transformed-hacking-with-hints-v3/train.parquet
cc_test_path=/workspace-vast/jinghanj/workspace/Structural_RL_dev/data/cc-transformed-hacking-with-hints-v3/test.parquet

train_files="['$cc_train_path']"
test_files="['$cc_test_path']"

# ============================================
# Custom Reward Function (hacking variant)
# ============================================
custom_reward_fn=/workspace-vast/jinghanj/workspace/Structural_RL_dev/train/custom_code_reward_hacking.py

# ============================================
# Common Settings (Gemma3-4B Model)
# ============================================
BASE_MODEL_PATH=/workspace-vast/jinghanj/workspace/Structural_RL/models/checkpoints/gemma3-4b-it

THINK_TOKEN_ID=262145
END_THINK_TOKEN_ID=262146
END_THINK_DELIMITER_STR="</think>"

SEED=1997

MAX_PROMPT_LENGTH=768
MAX_RESPONSE_LENGTH=2048

# ============================================
# Method-Specific Configuration
# ============================================
case "$METHOD" in
    vanilla)
        MODEL_PATH="$BASE_MODEL_PATH"

        ACTOR_USE_COT_MASKING=False
        ACTOR_USE_COT_GRADIENT_MASKING=False
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=False
        ACTOR_BLOCK_PROMPT_GRADIENTS=False
        ACTOR_BLOCK_ANSWER_GRADIENTS=False
        USE_SUBSTRING_MATCHING=False

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=True
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=4
        REF_MICRO_BATCH=8
        ROLLOUT_MICRO_BATCH=4

        EXPERIMENT_NAME="gemma3-4b-hacking-v3-vanilla-len2048-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-4b-hacking-v3-vanilla-bs${TRAIN_BATCH_SIZE}-len2048-seed${SEED}"

        echo "Configuration: VANILLA (Baseline hacking)"
        echo "  - No gradient masking"
        echo "  - use_remove_padding: $USE_REMOVE_PADDING"
        ;;

    update_mask)
        MODEL_PATH="$BASE_MODEL_PATH"

        ACTOR_USE_COT_MASKING=True
        ACTOR_USE_COT_GRADIENT_MASKING=False
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=False
        ACTOR_BLOCK_PROMPT_GRADIENTS=False
        ACTOR_BLOCK_ANSWER_GRADIENTS=False
        USE_SUBSTRING_MATCHING=True

        REF_USE_COT_MASKING=True
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=False
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=4
        REF_MICRO_BATCH=8
        ROLLOUT_MICRO_BATCH=4

        EXPERIMENT_NAME="gemma3-4b-hacking-v3-update_mask-len2048-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-4b-hacking-v3-update_mask-bs${TRAIN_BATCH_SIZE}-len2048-seed${SEED}"

        echo "Configuration: UPDATE MASK (Forward CoT Masking)"
        echo "  - use_cot_masking: $ACTOR_USE_COT_MASKING"
        echo "  - Delimiter: \"$END_THINK_DELIMITER_STR\""
        ;;

    cot_gradient)
        MODEL_PATH="$BASE_MODEL_PATH"

        ACTOR_USE_COT_MASKING=False
        ACTOR_USE_COT_GRADIENT_MASKING=True
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=False
        ACTOR_BLOCK_PROMPT_GRADIENTS=True
        ACTOR_BLOCK_ANSWER_GRADIENTS=True
        USE_SUBSTRING_MATCHING=True

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=True
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=4
        REF_MICRO_BATCH=8
        ROLLOUT_MICRO_BATCH=4

        EXPERIMENT_NAME="gemma3-4b-hacking-v3-cot_gradient-len2048-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-4b-hacking-v3-cot_gradient-bs${TRAIN_BATCH_SIZE}-len2048-seed${SEED}"

        echo "Configuration: COT GRADIENT (Parameter Masking)"
        echo "  - use_cot_gradient_masking: $ACTOR_USE_COT_GRADIENT_MASKING"
        echo "  - Delimiter: \"$END_THINK_DELIMITER_STR\""
        ;;

    gradient_mask)
        MODEL_PATH="$BASE_MODEL_PATH"

        ACTOR_USE_COT_MASKING=False
        ACTOR_USE_COT_GRADIENT_MASKING=False
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=True
        ACTOR_BLOCK_PROMPT_GRADIENTS=False
        ACTOR_BLOCK_ANSWER_GRADIENTS=False
        USE_SUBSTRING_MATCHING=True

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=False
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=2
        REF_MICRO_BATCH=4
        ROLLOUT_MICRO_BATCH=2

        EXPERIMENT_NAME="gemma3-4b-hacking-v3-gradient_mask-len2048-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-4b-hacking-v3-gradient_mask-bs${TRAIN_BATCH_SIZE}-len2048-seed${SEED}"

        echo "Configuration: GRADIENT MASK (Attention Masking)"
        echo "  - use_attention_gradient_masking: $ACTOR_USE_ATTENTION_GRADIENT_MASKING"
        echo "  - Delimiter: \"$END_THINK_DELIMITER_STR\""
        ;;

    combined_mask)
        MODEL_PATH="$BASE_MODEL_PATH"

        ACTOR_USE_COT_MASKING=False
        ACTOR_USE_COT_GRADIENT_MASKING=True
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=True
        ACTOR_BLOCK_PROMPT_GRADIENTS=True
        ACTOR_BLOCK_ANSWER_GRADIENTS=True
        USE_SUBSTRING_MATCHING=True

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=False
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=4
        REF_MICRO_BATCH=8
        ROLLOUT_MICRO_BATCH=4

        EXPERIMENT_NAME="gemma3-4b-hacking-v3-combined_mask-len2048-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-4b-hacking-v2-combined_mask-bs${TRAIN_BATCH_SIZE}-len2048-seed${SEED}"

        echo "Configuration: COMBINED MASK (Attention + Parameter)"
        echo "  - use_cot_gradient_masking: $ACTOR_USE_COT_GRADIENT_MASKING"
        echo "  - use_attention_gradient_masking: $ACTOR_USE_ATTENTION_GRADIENT_MASKING"
        echo "  - Delimiter: \"$END_THINK_DELIMITER_STR\""
        ;;
esac

echo ""
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Data: cc-transformed-hacking"
echo "  Reward: custom_code_reward_hacking_v3.py"
echo "  Sequence lengths: prompt=$MAX_PROMPT_LENGTH, response=$MAX_RESPONSE_LENGTH"
echo "  Batch sizes:"
echo "    * Train batch: $TRAIN_BATCH_SIZE"
echo "    * Actor micro: $ACTOR_MICRO_BATCH"
echo "    * Ref micro:   $REF_MICRO_BATCH"
echo "    * Rollout micro: $ROLLOUT_MICRO_BATCH"
echo "============================================"
echo ""

# ============================================
# Launch Training
# ============================================
cd /workspace-vast/jinghanj/workspace/Structural_RL_dev/train

echo "Starting training with PID tracking for cleanup..."
set -m

if [[ "$USE_SUBSTRING_MATCHING" == "True" ]]; then
    DELIMITER_ARGS="actor_rollout_ref.actor.use_substring_delimiter_matching=$USE_SUBSTRING_MATCHING actor_rollout_ref.actor.end_think_delimiter_str=\"$END_THINK_DELIMITER_STR\""
else
    DELIMITER_ARGS="actor_rollout_ref.actor.end_think_token_id=$END_THINK_TOKEN_ID"
fi

echo "Delimiter Arguments: $DELIMITER_ARGS"
echo ""
echo "Launching training..."
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ray_kwargs.ray_init.num_cpus=64 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size=256 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key='messages' \
    data.reward_fn_key='data_source' \
    data.seed="$SEED" \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_cot_masking="$ACTOR_USE_COT_MASKING" \
    actor_rollout_ref.actor.use_cot_gradient_masking="$ACTOR_USE_COT_GRADIENT_MASKING" \
    actor_rollout_ref.actor.use_attention_gradient_masking="$ACTOR_USE_ATTENTION_GRADIENT_MASKING" \
    actor_rollout_ref.actor.block_prompt_gradients="$ACTOR_BLOCK_PROMPT_GRADIENTS" \
    actor_rollout_ref.actor.block_answer_gradients="$ACTOR_BLOCK_ANSWER_GRADIENTS" \
    actor_rollout_ref.actor.disable_cot_masking_for_old_log_prob=True \
    $DELIMITER_ARGS \
    actor_rollout_ref.actor.think_token_id="$THINK_TOKEN_ID" \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_MICRO_BATCH" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_MICRO_BATCH" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.use_cot_masking="$REF_USE_COT_MASKING" \
    actor_rollout_ref.ref.think_token_id="$THINK_TOKEN_ID" \
    actor_rollout_ref.ref.end_think_token_id="$END_THINK_TOKEN_ID" \
    actor_rollout_ref.ref.use_substring_delimiter_matching="$USE_SUBSTRING_MATCHING" \
    actor_rollout_ref.ref.end_think_delimiter_str='"'"$END_THINK_DELIMITER_STR"'"' \
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
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.log_val_generations=1 \
    trainer.test_freq=10 \
    trainer.total_epochs=20 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$CHECKPOINT_DIR" &

TRAIN_PID=$!
wait $TRAIN_PID

echo ""
echo "============================================"
echo "Training completed successfully"
echo "============================================"

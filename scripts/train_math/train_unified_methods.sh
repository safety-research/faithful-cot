#!/bin/bash
# ============================================
# Unified Training Script for DeepMind Math
# ============================================
#
# Supports 5 training methods with substring delimiter matching:
#   1. vanilla          - No masking (baseline)
#   2. update_mask      - Forward CoT masking (answer can't attend to prompt in forward pass)
#   3. cot_gradient     - Parameter gradient masking (CoT tokens only update params)
#   4. gradient_mask    - Attention gradient masking (answer→prompt blocking in backward pass)
#   5. combined_mask    - Both attention + parameter masking
#
# All intervention methods use substring matching for delimiter detection,
# making them robust and tokenizer-agnostic.
#
# Usage:
#   bash scripts/train_math/train_unified_methods.sh <method>
#
# Arguments:
#   method: vanilla | cot_gradient | gradient_mask | combined_mask
#
# Examples:
#   bash scripts/train_math/train_unified_methods.sh vanilla
#     → Standard training without any masking
#
#   bash scripts/train_math/train_unified_methods.sh update_mask
#     → Forward CoT masking: answer cannot attend to prompt during forward pass
#     → Uses substring matching to find </think> delimiter
#
#   bash scripts/train_math/train_unified_methods.sh cot_gradient
#     → Parameter gradient masking: only CoT tokens update parameters
#     → Uses substring matching to find </think> delimiter
#
#   bash scripts/train_math/train_unified_methods.sh gradient_mask
#     → Attention gradient masking: blocks answer→prompt gradients in backward pass
#     → Uses substring matching to find </think> delimiter
#
#   bash scripts/train_math/train_unified_methods.sh combined_mask
#     → Both attention + parameter gradient masking
#     → Uses substring matching to find </think> delimiter

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

# Trap multiple signals
trap cleanup EXIT SIGINT SIGTERM SIGQUIT SIGHUP
ray stop -f 2>/dev/null || true
rm -rf /tmp/ray /tmp/ray_* /dev/shm/ray_* 2>/dev/null || true

# ============================================
# Parse Arguments
# ============================================
METHOD=${1:-vanilla}

# Validate method
case "$METHOD" in
    vanilla|update_mask|cot_gradient|gradient_mask|combined_mask)
        ;;
    *)
        echo "ERROR: Invalid method '$METHOD'"
        echo ""
        echo "Usage: $0 <method>"
        echo ""
        echo "Available methods:"
        echo "  vanilla          - No masking (baseline)"
        echo "  update_mask      - Forward CoT masking (answer can't attend to prompt)"
        echo "  cot_gradient     - Parameter gradient masking (CoT tokens only)"
        echo "  gradient_mask    - Attention gradient masking (answer→prompt blocking)"
        echo "  combined_mask    - Both attention + parameter masking"
        echo ""
        echo "All intervention methods use substring matching for </think> delimiter."
        exit 1
        ;;
esac

echo "============================================"
echo "UNIFIED TRAINING - Method: $METHOD"
echo "============================================"

# ============================================
# Environment Setup
# ============================================
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/$(whoami)/.cache/huggingface/token
export HF_TOKEN=$(cat $HF_TOKEN_PATH)
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_PROJECT=deepmind_math_unified_methods

# ============================================
# Data Paths
# ============================================
deepmind_train_path=/workspace-vast/jinghanj/workspace/Structural_RL/data/deepmindmath_all/train.parquet
deepmind_test_path=/workspace-vast/jinghanj/workspace/Structural_RL/data/deepmindmath_all/test.parquet

train_files="['$deepmind_train_path']"
test_files="['$deepmind_test_path']"

# ============================================
# Custom Reward Function
# ============================================
custom_reward_fn=/workspace-vast/jinghanj/workspace/Structural_RL_dev/train/custom_math_reward_strict.py

# ============================================
# Common Settings (Gemma3-1B Model)
# ============================================
# Using Gemma3-1B as the base model for all methods
BASE_MODEL_PATH=/workspace-vast/jinghanj/workspace/Structural_RL/results/checkpoints/gemma3-1b-deepmindmath_all-format/gemma3-1b-formt

# Common token IDs for Gemma3
THINK_TOKEN_ID=262145        # <think> token
END_THINK_TOKEN_ID=262146    # </think> token (used only in special token mode)

# Delimiter string for substring matching (used in intervention methods)
END_THINK_DELIMITER_STR="</think>"

# Training hyperparameters (shared across methods)
SEED=1995

# ============================================
# Method-Specific Configuration
# ============================================
case "$METHOD" in
    vanilla)
        # ======== VANILLA: No Masking ========
        MODEL_PATH="$BASE_MODEL_PATH"

        # ACTOR settings: No masking
        ACTOR_USE_COT_MASKING=False
        ACTOR_USE_COT_GRADIENT_MASKING=False
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=False
        ACTOR_BLOCK_PROMPT_GRADIENTS=False
        ACTOR_BLOCK_ANSWER_GRADIENTS=False
        USE_SUBSTRING_MATCHING=False  # N/A for vanilla

        # REF settings: No masking
        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        # Optimization settings
        USE_REMOVE_PADDING=True       # Can use Flash Attention optimization
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=32
        REF_MICRO_BATCH=64
        ROLLOUT_MICRO_BATCH=64

        # Experiment naming
        EXPERIMENT_NAME="gemma3-1b-deepmindmath-vanilla-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-1b-format-vanilla-bs${TRAIN_BATCH_SIZE}-seed${SEED}"

        echo "Configuration: VANILLA (Baseline)"
        echo "  - Model: $MODEL_PATH"
        echo "  - No gradient masking"
        echo "  - use_remove_padding: $USE_REMOVE_PADDING (Flash Attention enabled)"
        echo ""
        echo "  Forward Pass:"
        echo "    → Normal attention (answer CAN attend to prompt)"
        echo ""
        echo "  Backward Pass:"
        echo "    → Standard gradient flow (no blocking)"
        echo "    → All tokens update parameters"
        ;;

    update_mask)
        # ======== UPDATE MASK: Forward CoT Masking ========
        MODEL_PATH="$BASE_MODEL_PATH"

        # ACTOR settings: Forward CoT masking with substring matching
        ACTOR_USE_COT_MASKING=True                      # YES forward masking
        ACTOR_USE_COT_GRADIENT_MASKING=False           # NO parameter masking
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=False     # NO attention gradient masking
        ACTOR_BLOCK_PROMPT_GRADIENTS=False             # N/A (no parameter masking)
        ACTOR_BLOCK_ANSWER_GRADIENTS=False             # N/A (no parameter masking)
        USE_SUBSTRING_MATCHING=True                    # YES substring matching

        # REF settings: Same forward masking as actor (important!)
        REF_USE_COT_MASKING=True                        # YES forward masking
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        # Optimization settings
        USE_REMOVE_PADDING=False      # Required for forward masking
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=32
        REF_MICRO_BATCH=64
        ROLLOUT_MICRO_BATCH=64

        # Experiment naming
        EXPERIMENT_NAME="gemma3-1b-deepmindmath-update_mask-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-1b-format-update_mask-bs${TRAIN_BATCH_SIZE}-seed${SEED}"

        echo "Configuration: UPDATE MASK (Forward CoT Masking)"
        echo "  - Model: $MODEL_PATH"
        echo "  - Delimiter Mode: Substring Matching"
        echo "  - Delimiter String: \"$END_THINK_DELIMITER_STR\""
        echo ""
        echo "  ACTOR (training model):"
        echo "    - use_cot_masking: $ACTOR_USE_COT_MASKING (YES - forward masking)"
        echo "    - use_cot_gradient_masking: $ACTOR_USE_COT_GRADIENT_MASKING"
        echo "    - use_attention_gradient_masking: $ACTOR_USE_ATTENTION_GRADIENT_MASKING"
        echo "    - use_substring_delimiter_matching: $USE_SUBSTRING_MATCHING"
        echo ""
        echo "  REFERENCE (frozen model):"
        echo "    - use_cot_masking: $REF_USE_COT_MASKING (YES - same forward masking)"
        echo ""
        echo "  Forward Pass:"
        echo "    → Modified attention: Answer CANNOT attend to prompt"
        echo "    → 4D attention mask with -inf for blocked connections"
        echo "    → Delimiter detection: O(n×m) substring matching"
        echo ""
        echo "  Backward Pass:"
        echo "    → Normal gradient flow (no additional blocking)"
        echo "    → All tokens can update parameters"
        echo ""
        echo "  Expected Effects:"
        echo "    ✓ Forces model to rely on CoT reasoning during generation"
        echo "    ✓ Answer must use information from CoT, not direct prompt access"
        ;;

    cot_gradient)
        # ======== COT GRADIENT: Parameter Masking Only ========
        MODEL_PATH="$BASE_MODEL_PATH"

        # ACTOR settings: Parameter gradient masking with substring matching
        ACTOR_USE_COT_MASKING=False                     # NO forward masking
        ACTOR_USE_COT_GRADIENT_MASKING=True            # YES parameter masking
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=False     # NO attention masking
        ACTOR_BLOCK_PROMPT_GRADIENTS=True             # Block prompt→parameters
        ACTOR_BLOCK_ANSWER_GRADIENTS=True             # Block answer→parameters
        USE_SUBSTRING_MATCHING=True                    # YES substring matching

        # REF settings: No masking
        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        # Optimization settings
        USE_REMOVE_PADDING=True       # Can use optimization (parameter masking doesn't affect it)
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=32
        REF_MICRO_BATCH=64
        ROLLOUT_MICRO_BATCH=64

        # Experiment naming
        EXPERIMENT_NAME="gemma3-1b-deepmindmath-cot_gradient-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-1b-format-cot_gradient-bs${TRAIN_BATCH_SIZE}-seed${SEED}"

        echo "Configuration: COT GRADIENT (Parameter Masking)"
        echo "  - Model: $MODEL_PATH"
        echo "  - Delimiter Mode: Substring Matching"
        echo "  - Delimiter String: \"$END_THINK_DELIMITER_STR\""
        echo ""
        echo "  ACTOR (training model):"
        echo "    - use_cot_gradient_masking: $ACTOR_USE_COT_GRADIENT_MASKING"
        echo "    - use_attention_gradient_masking: $ACTOR_USE_ATTENTION_GRADIENT_MASKING"
        echo "    - use_substring_delimiter_matching: $USE_SUBSTRING_MATCHING"
        echo "    - block_prompt_gradients: $ACTOR_BLOCK_PROMPT_GRADIENTS"
        echo "    - block_answer_gradients: $ACTOR_BLOCK_ANSWER_GRADIENTS"
        echo ""
        echo "  Forward Pass:"
        echo "    → Normal attention (answer CAN attend to prompt)"
        echo ""
        echo "  Backward Pass:"
        echo "    → Attention gradients: Normal flow"
        echo "    → Parameter gradients: ONLY CoT tokens update parameters"
        echo "    → Delimiter detection: O(n×m) substring matching"
        echo ""
        echo "  Expected Effects:"
        echo "    ✓ Model learns from CoT reasoning structure only"
        echo "    ✓ 60-70% gradient reduction (fewer tokens updating)"
        ;;

    gradient_mask)
        # ======== GRADIENT MASK: Attention Masking Only ========
        MODEL_PATH="$BASE_MODEL_PATH"

        # ACTOR settings: Attention gradient masking with substring matching
        ACTOR_USE_COT_MASKING=False                     # NO forward masking
        ACTOR_USE_COT_GRADIENT_MASKING=False           # NO parameter masking
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=True      # YES attention masking
        ACTOR_BLOCK_PROMPT_GRADIENTS=False             # N/A (no parameter masking)
        ACTOR_BLOCK_ANSWER_GRADIENTS=False             # N/A (no parameter masking)
        USE_SUBSTRING_MATCHING=True                    # YES substring matching

        # REF settings: No masking
        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        # Optimization settings
        USE_REMOVE_PADDING=False      # Required for masking
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=16
        REF_MICRO_BATCH=64
        ROLLOUT_MICRO_BATCH=64

        # Experiment naming
        EXPERIMENT_NAME="gemma3-1b-deepmindmath-gradient_mask-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-1b-format-gradient_mask-bs${TRAIN_BATCH_SIZE}-seed${SEED}"

        echo "Configuration: GRADIENT MASK (Attention Masking)"
        echo "  - Model: $MODEL_PATH"
        echo "  - Delimiter Mode: Substring Matching"
        echo "  - Delimiter String: \"$END_THINK_DELIMITER_STR\""
        echo ""
        echo "  ACTOR (training model):"
        echo "    - use_cot_gradient_masking: $ACTOR_USE_COT_GRADIENT_MASKING"
        echo "    - use_attention_gradient_masking: $ACTOR_USE_ATTENTION_GRADIENT_MASKING"
        echo "    - use_substring_delimiter_matching: $USE_SUBSTRING_MATCHING"
        echo ""
        echo "  Forward Pass:"
        echo "    → Normal attention (answer CAN attend to prompt)"
        echo ""
        echo "  Backward Pass:"
        echo "    → Attention gradients: Answer→Prompt BLOCKED"
        echo "    → Parameter gradients: All tokens can update"
        echo "    → Delimiter detection: O(n×m) substring matching"
        echo ""
        echo "  Expected Effects:"
        echo "    ✓ Prevents answer from learning prompt shortcuts"
        echo "    ✓ Forces model to use CoT reasoning path"
        ;;

    combined_mask)
        # ======== COMBINED MASK: Attention + Parameter Masking ========
        MODEL_PATH="$BASE_MODEL_PATH"

        # ACTOR settings: Both attention + parameter masking with substring matching
        ACTOR_USE_COT_MASKING=False                     # NO forward masking
        ACTOR_USE_COT_GRADIENT_MASKING=True            # YES parameter masking
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=True      # YES attention masking
        ACTOR_BLOCK_PROMPT_GRADIENTS=True             # Block prompt→parameters
        ACTOR_BLOCK_ANSWER_GRADIENTS=True             # Block answer→parameters
        USE_SUBSTRING_MATCHING=True                    # YES substring matching

        # REF settings: No masking
        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        # Optimization settings
        USE_REMOVE_PADDING=False      # Required for masking
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=16
        REF_MICRO_BATCH=64
        ROLLOUT_MICRO_BATCH=64

        # Experiment naming
        EXPERIMENT_NAME="gemma3-1b-deepmindmath-combined_mask-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-1b-format-combined_mask-bs${TRAIN_BATCH_SIZE}-seed${SEED}"

        echo "Configuration: COMBINED MASK (Attention + Parameter)"
        echo "  - Model: $MODEL_PATH"
        echo "  - Delimiter Mode: Substring Matching"
        echo "  - Delimiter String: \"$END_THINK_DELIMITER_STR\""
        echo ""
        echo "  ACTOR (training model):"
        echo "    - use_cot_gradient_masking: $ACTOR_USE_COT_GRADIENT_MASKING"
        echo "    - use_attention_gradient_masking: $ACTOR_USE_ATTENTION_GRADIENT_MASKING"
        echo "    - use_substring_delimiter_matching: $USE_SUBSTRING_MATCHING"
        echo "    - block_prompt_gradients: $ACTOR_BLOCK_PROMPT_GRADIENTS"
        echo "    - block_answer_gradients: $ACTOR_BLOCK_ANSWER_GRADIENTS"
        echo ""
        echo "  Forward Pass:"
        echo "    → Normal attention (answer CAN attend to prompt)"
        echo ""
        echo "  Backward Pass:"
        echo "    → Attention gradients: Answer→Prompt BLOCKED"
        echo "    → Parameter gradients: ONLY CoT tokens update parameters"
        echo "    → Delimiter detection: O(n×m) substring matching"
        echo ""
        echo "  Expected Effects:"
        echo "    ✓ Strongest structural bias towards CoT reasoning"
        echo "    ✓ Combines benefits of both methods"
        ;;
esac

echo ""
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Batch sizes:"
echo "    * Train batch: $TRAIN_BATCH_SIZE"
echo "    * Actor micro: $ACTOR_MICRO_BATCH"
echo "    * Ref micro: $REF_MICRO_BATCH"
echo "    * Rollout micro: $ROLLOUT_MICRO_BATCH"
echo "============================================"
echo ""

# ============================================
# Launch Training
# ============================================
cd /workspace-vast/jinghanj/workspace/Structural_RL_dev/train

echo "Starting training with PID tracking for cleanup..."
set -m  # Enable job control

# Build delimiter-specific arguments
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
    data.val_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
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
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
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

# Store the PID for cleanup
TRAIN_PID=$!

# Wait for training to complete
wait $TRAIN_PID

echo ""
echo "============================================"
echo "Training completed successfully"
echo "============================================"

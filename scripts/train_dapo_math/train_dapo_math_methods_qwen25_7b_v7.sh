#!/bin/bash
# ============================================
# Training Script for DAPO Math (Qwen2.5-7B-Instruct) — v5
# ============================================
#
# Supports 5 training methods:
#   1. vanilla          - No masking (baseline)
#   2. update_mask      - Forward CoT masking
#   3. cot_gradient     - Parameter gradient masking (CoT tokens only)
#   4. gradient_mask    - Attention gradient masking (answer→prompt blocking)
#   5. combined_mask    - Both attention + parameter masking
#
# Changes vs v3:
#   - cot_gradient: MAX_RESPONSE_LENGTH=10000 (longer CoT budget)
#                   OVERLONG_BUFFER_ENABLE=False (no overlong penalty)
#                   PPO_MULTIPLIER=1 (PPO_MAX=11024, one full sequence per micro-batch)
#
# Data: DAPO-Math-17k (no hints — pure math reward)
# Model: Qwen2.5-7B-Instruct (max_position_embeddings=32768 natively)
# Reward: DAPO reward manager (overlong penalty disabled for cot_gradient)
#
# NOTE: <think> and </think> are NOT special tokens in Qwen2.5 — substring matching is always used.
#
# Usage:
#   bash scripts/train_dapo_math/train_dapo_math_methods_qwen25_7b_v5.sh <method> [lr] [data_variant]

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
LR=${2:-1e-6}
DATA_VARIANT=${3:-v1}   # v1 = original data/dapo_math, anything else = data/dapo_math_v2/<DATA_VARIANT>

case "$METHOD" in
    vanilla|update_mask|cot_gradient|gradient_mask|combined_mask)
        ;;
    *)
        echo "ERROR: Invalid method '$METHOD'"
        echo "Available methods: vanilla | update_mask | cot_gradient | gradient_mask | combined_mask"
        exit 1
        ;;
esac

echo "============================================"
echo "DAPO MATH TRAINING (Qwen2.5-7B-Instruct) - Method: $METHOD"
echo "============================================"

# ============================================
# Environment Setup
# ============================================
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/$(whoami)/.cache/huggingface/token
export HF_TOKEN=$(cat $HF_TOKEN_PATH)
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_PROJECT=dapo_math_qwen25_7b

# ============================================
# Data Paths
# ============================================
if [ "$DATA_VARIANT" = "v1" ]; then
    DATA_DIR=/workspace-vast/jinghanj/workspace/Structural_RL_dev/data/dapo_math
else
    DATA_DIR=/workspace-vast/jinghanj/workspace/Structural_RL_dev/data/dapo_math_v2/${DATA_VARIANT}
fi
train_path=${DATA_DIR}/train.parquet
test_path=${DATA_DIR}/test.parquet

if [ ! -f "$train_path" ]; then
    echo "ERROR: train data not found: $train_path"; exit 1
fi

train_files="['$train_path']"
test_files="['$test_path']"

# ============================================
# Reward Function
# ============================================
custom_reward_fn=/workspace-vast/jinghanj/workspace/Structural_RL_dev/train/custom_math_reward_dapo_v2.py

# ============================================
# Common Settings (Qwen2.5-7B-Instruct)
# ============================================
BASE_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct

# <think> tokenizes as [13708, 766, 29] — use first token as marker
# Substring matching is ALWAYS used for Qwen (multi-token think delimiters)
THINK_TOKEN_ID=13708
END_THINK_TOKEN_ID=29
END_THINK_DELIMITER_STR="</think>"

SEED=1997
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=4096
OVERLONG_BUFFER_LEN=1024   # penalty kicks in after MAX_RESPONSE_LENGTH - OVERLONG_BUFFER_LEN tokens
OVERLONG_BUFFER_ENABLE=True  # default; set to False for cot_gradient (no penalty)
TRAIN_BATCH_SIZE=512
GPU_MEMORY_UTIL=0.80  # default; overridden per method
PPO_MULTIPLIER=2  # default; overridden to 1 for methods with high activation memory
ENABLE_GRADIENT_CHECKPOINTING=False  # default; overridden per method
REF_PARAM_OFFLOAD=False              # default; set to True for high-memory methods

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

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        MAX_RESPONSE_LENGTH=10000  # match cot_gradient for fair comparison
        OVERLONG_BUFFER_ENABLE=False  # no overlong penalty
        PPO_MULTIPLIER=1           # PPO_MAX=(1024+10000)*1=11024
        ENABLE_GRADIENT_CHECKPOINTING=True  # 10000 token sequences → large activations
        USE_REMOVE_PADDING=True
        ACTOR_MICRO_BATCH=1
        REF_MICRO_BATCH=4
        ROLLOUT_MICRO_BATCH=16
        ACTOR_PARAM_OFFLOAD=False

        EXPERIMENT_NAME="qwen25-7b-dapo-math-vanilla-${DATA_VARIANT}-lr${LR}-len10k-nopenalty-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/qwen25-7b-dapo-math-vanilla-${DATA_VARIANT}-lr${LR}-len10k-nopenalty-bs${TRAIN_BATCH_SIZE}-seed${SEED}-v7-wo-decay-kl"
        ;;

    update_mask)
        MODEL_PATH="$BASE_MODEL_PATH"

        ACTOR_USE_COT_MASKING=True
        ACTOR_USE_COT_GRADIENT_MASKING=False
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=False
        ACTOR_BLOCK_PROMPT_GRADIENTS=False
        ACTOR_BLOCK_ANSWER_GRADIENTS=False

        REF_USE_COT_MASKING=True
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=False
        ENABLE_GRADIENT_CHECKPOINTING=True  # custom 4D mask → SDPA → O(n²) attn per layer; GC recomputes 1 layer at a time
        GPU_MEMORY_UTIL=0.50  # vLLM KV cache not fully released before actor update on 140GiB GPU
        PPO_MULTIPLIER=1
        ACTOR_MICRO_BATCH=1
        REF_MICRO_BATCH=4
        ROLLOUT_MICRO_BATCH=16
        ACTOR_PARAM_OFFLOAD=True

        EXPERIMENT_NAME="qwen25-7b-dapo-math-update_mask-${DATA_VARIANT}-lr${LR}-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/qwen25-7b-dapo-math-update_mask-${DATA_VARIANT}-lr${LR}-bs${TRAIN_BATCH_SIZE}-seed${SEED}-v7"
        ;;

    cot_gradient)
        MODEL_PATH="$BASE_MODEL_PATH"

        ACTOR_USE_COT_MASKING=False
        ACTOR_USE_COT_GRADIENT_MASKING=True
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=False
        ACTOR_BLOCK_PROMPT_GRADIENTS=True
        ACTOR_BLOCK_ANSWER_GRADIENTS=True

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        MAX_RESPONSE_LENGTH=10000  # longer CoT budget; Qwen2.5-7B max_position_embeddings=32768 → 11024 fits
        OVERLONG_BUFFER_ENABLE=False  # no overlong penalty — allow model to use full CoT length freely
        PPO_MULTIPLIER=1           # PPO_MAX=(1024+10000)*1=11024 — one full sequence per micro-batch
        ENABLE_GRADIENT_CHECKPOINTING=True  # compatible: masking is forward-pass decomposition via thread-local, not activation hooks
        USE_REMOVE_PADDING=True
        ACTOR_MICRO_BATCH=1
        REF_MICRO_BATCH=4
        ROLLOUT_MICRO_BATCH=16
        ACTOR_PARAM_OFFLOAD=False  # gradient mask tensors grow with seq length — offload params to free GPU headroom
        REF_PARAM_OFFLOAD=False   # ref params (~14 GiB) sit idle during actor backward — offload to free GPU headroom

        EXPERIMENT_NAME="qwen25-7b-dapo-math-cot_gradient-${DATA_VARIANT}-lr${LR}-len10k-nopenalty-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/qwen25-7b-dapo-math-cot_gradient-${DATA_VARIANT}-lr${LR}-len10k-nopenalty-bs${TRAIN_BATCH_SIZE}-seed${SEED}-v7-kl"
        ;;

    gradient_mask)
        MODEL_PATH="$BASE_MODEL_PATH"

        ACTOR_USE_COT_MASKING=False
        ACTOR_USE_COT_GRADIENT_MASKING=False
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=True
        ACTOR_BLOCK_PROMPT_GRADIENTS=False
        ACTOR_BLOCK_ANSWER_GRADIENTS=False

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        GPU_MEMORY_UTIL=0.50  # lower vLLM: 28 layers × seq² attn_weights stored for backward — needs headroom
        PPO_MULTIPLIER=1
        USE_REMOVE_PADDING=False
        ACTOR_MICRO_BATCH=1
        REF_MICRO_BATCH=4
        ROLLOUT_MICRO_BATCH=16
        ACTOR_PARAM_OFFLOAD=True  # offload params to CPU to free additional GPU headroom

        EXPERIMENT_NAME="qwen25-7b-dapo-math-gradient_mask-${DATA_VARIANT}-lr${LR}-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/qwen25-7b-dapo-math-gradient_mask-${DATA_VARIANT}-lr${LR}-bs${TRAIN_BATCH_SIZE}-seed${SEED}-v7"
        ;;

    combined_mask)
        MODEL_PATH="$BASE_MODEL_PATH"

        ACTOR_USE_COT_MASKING=False
        ACTOR_USE_COT_GRADIENT_MASKING=True
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=True
        ACTOR_BLOCK_PROMPT_GRADIENTS=True
        ACTOR_BLOCK_ANSWER_GRADIENTS=True

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=False
        ACTOR_MICRO_BATCH=1
        REF_MICRO_BATCH=4
        ROLLOUT_MICRO_BATCH=16
        ACTOR_PARAM_OFFLOAD=True  # gradient mask tensors + attn_weights: offload params to free headroom

        EXPERIMENT_NAME="qwen25-7b-dapo-math-combined_mask-${DATA_VARIANT}-lr${LR}-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/qwen25-7b-dapo-math-combined_mask-${DATA_VARIANT}-lr${LR}-bs${TRAIN_BATCH_SIZE}-seed${SEED}-v7"
        ;;
esac

PPO_MAX_TOKEN_LEN_PER_GPU=$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * PPO_MULTIPLIER ))

echo "  Model     : $MODEL_PATH"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Batch sizes: train=$TRAIN_BATCH_SIZE  actor_micro=$ACTOR_MICRO_BATCH  ref_micro=$REF_MICRO_BATCH  rollout_micro=$ROLLOUT_MICRO_BATCH"
echo "  Seq lengths: prompt=$MAX_PROMPT_LENGTH  response=$MAX_RESPONSE_LENGTH"
echo "  Overlong buffer: enable=$OVERLONG_BUFFER_ENABLE  len=$OVERLONG_BUFFER_LEN tokens"
echo "  remove_padding=$USE_REMOVE_PADDING"
echo "============================================"
echo ""

# ============================================
# Launch Training
# ============================================
cd /workspace-vast/jinghanj/workspace/Structural_RL_dev/train

set -m  # Enable job control

# Always use substring matching for Qwen (<think> is multi-token)
DELIMITER_ARGS="actor_rollout_ref.actor.use_substring_delimiter_matching=True actor_rollout_ref.actor.end_think_delimiter_str=\"$END_THINK_DELIMITER_STR\""

echo "Launching training..."
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ray_kwargs.ray_init.num_cpus=64 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size=512 \
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
    actor_rollout_ref.model.enable_gradient_checkpointing="$ENABLE_GRADIENT_CHECKPOINTING" \
    \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
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
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTIL" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_MICRO_BATCH" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
    actor_rollout_ref.ref.use_cot_masking="$REF_USE_COT_MASKING" \
    actor_rollout_ref.ref.think_token_id="$THINK_TOKEN_ID" \
    actor_rollout_ref.ref.end_think_token_id="$END_THINK_TOKEN_ID" \
    actor_rollout_ref.ref.use_substring_delimiter_matching=True \
    actor_rollout_ref.ref.end_think_delimiter_str='"'"$END_THINK_DELIMITER_STR"'"' \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    custom_reward_function.path="$custom_reward_fn" \
    custom_reward_function.name=compute_score_with_details \
    \
    reward_manager.name=dapo \
    reward_manager.source=register \
    +reward_model.reward_kwargs.max_resp_len=$MAX_RESPONSE_LENGTH \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=$OVERLONG_BUFFER_ENABLE \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=$OVERLONG_BUFFER_LEN \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
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
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$CHECKPOINT_DIR" &

TRAIN_PID=$!
wait $TRAIN_PID

echo ""
echo "============================================"
echo "Training completed successfully"
echo "============================================"

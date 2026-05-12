#!/bin/bash
# ============================================
# FACT Training Script for CodeContests Reward Hacking
# FACT = Faithful Adversarial CoT Training
# Response length: 2048 — Reward v3
# ============================================
#
# Supports same 5 methods as v3 + FACT (pep):
#   1. vanilla          - No masking (baseline)
#   2. update_mask      - Forward CoT masking
#   3. cot_gradient     - Parameter gradient masking
#   4. gradient_mask    - Attention gradient masking
#   5. combined_mask    - Both attention + parameter masking
#   6. pep              - FACT: Prompt Embedding Adversarial Training
#
# FACT hyperparameters (optional env vars):
#   PEP_EPSILON   noise magnitude as fraction of embedding norm (default: 0.05)
#   PEP_LAYER     layer to perturb: -1=embed_tokens, N=model.layers[N] (default: -1)
#   PEP_MODE      random | worst_case (default: worst_case)
#   PEP_PGD_STEPS 1=FGSM, >1=PGD (default: 1)
#   PEP_ATTACK_ANSWER_ONLY True=attack loss on answer only (default: False = CoT+answer)
#
# Sequence lengths: prompt=768, response=2048
# Model: gemma3-4b-it
# Reward: custom_code_reward_hacking_v3.py
#
# Usage:
#   bash scripts/train_code/train_hacking_methods_2048_v3_fact.sh pep
#   PEP_LAYER=8 bash scripts/train_code/train_hacking_methods_2048_v3_fact.sh pep
#   bash scripts/train_code/train_hacking_methods_2048_v3_fact.sh vanilla

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
METHOD=${1:-pep}

case "$METHOD" in
    vanilla|update_mask|cot_gradient|gradient_mask|combined_mask|pep)
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
        echo "  pep              - FACT: Prompt Embedding Adversarial Training"
        exit 1
        ;;
esac

echo "============================================"
echo "HACKING VARIANT TRAINING (FACT) - Method: $METHOD"
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

# PEP-specific defaults (can be overridden by env vars)
PEP_EPSILON=${PEP_EPSILON:-0.05}
PEP_NORMALIZE=${PEP_NORMALIZE:-True}
PEP_LAYER=${PEP_LAYER:--1}          # -1=embed_tokens (FACT default); N=transformer layer N
PEP_MODE=${PEP_MODE:-worst_case}    # worst_case=FGSM/PGD (FACT default); random=Gaussian
PEP_PGD_STEPS=${PEP_PGD_STEPS:-1}  # 1=FGSM (FACT default); >1=PGD
PEP_DUAL_LOSS=${PEP_DUAL_LOSS:-False}
PEP_DUAL_LOSS_COEF=${PEP_DUAL_LOSS_COEF:-1.0}
PEP_ATTACK_ANSWER_ONLY=${PEP_ATTACK_ANSWER_ONLY:-False}  # False = full CoT+answer loss (FACT)

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
        ACTOR_USE_PROMPT_EMB_PERTURB=False
        USE_SUBSTRING_MATCHING=False

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=True
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=4
        REF_MICRO_BATCH=8
        ROLLOUT_MICRO_BATCH=4

        EXPERIMENT_NAME="gemma3-4b-hacking-v3-vanilla-fact-len2048-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-4b-hacking-v3-vanilla-fact-bs${TRAIN_BATCH_SIZE}-len2048-seed${SEED}"

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
        ACTOR_USE_PROMPT_EMB_PERTURB=False
        USE_SUBSTRING_MATCHING=True

        REF_USE_COT_MASKING=True
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=False
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=4
        REF_MICRO_BATCH=8
        ROLLOUT_MICRO_BATCH=4

        EXPERIMENT_NAME="gemma3-4b-hacking-v3-update_mask-fact-len2048-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-4b-hacking-v3-update_mask-fact-bs${TRAIN_BATCH_SIZE}-len2048-seed${SEED}"

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
        ACTOR_USE_PROMPT_EMB_PERTURB=False
        USE_SUBSTRING_MATCHING=True

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=True
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=4
        REF_MICRO_BATCH=8
        ROLLOUT_MICRO_BATCH=4

        EXPERIMENT_NAME="gemma3-4b-hacking-v3-cot_gradient-fact-len2048-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-4b-hacking-v3-cot_gradient-fact-bs${TRAIN_BATCH_SIZE}-len2048-seed${SEED}"

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
        ACTOR_USE_PROMPT_EMB_PERTURB=False
        USE_SUBSTRING_MATCHING=True

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=False
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=2
        REF_MICRO_BATCH=4
        ROLLOUT_MICRO_BATCH=2

        EXPERIMENT_NAME="gemma3-4b-hacking-v3-gradient_mask-fact-len2048-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-4b-hacking-v3-gradient_mask-fact-bs${TRAIN_BATCH_SIZE}-len2048-seed${SEED}"

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
        ACTOR_USE_PROMPT_EMB_PERTURB=False
        USE_SUBSTRING_MATCHING=True

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=False
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=4
        REF_MICRO_BATCH=8
        ROLLOUT_MICRO_BATCH=4

        EXPERIMENT_NAME="gemma3-4b-hacking-v3-combined_mask-fact-len2048-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-4b-hacking-v3-combined_mask-fact-bs${TRAIN_BATCH_SIZE}-len2048-seed${SEED}"

        echo "Configuration: COMBINED MASK (Attention + Parameter)"
        echo "  - use_cot_gradient_masking: $ACTOR_USE_COT_GRADIENT_MASKING"
        echo "  - use_attention_gradient_masking: $ACTOR_USE_ATTENTION_GRADIENT_MASKING"
        echo "  - Delimiter: \"$END_THINK_DELIMITER_STR\""
        ;;

    pep)
        MODEL_PATH="$BASE_MODEL_PATH"

        ACTOR_USE_COT_MASKING=False
        ACTOR_USE_COT_GRADIENT_MASKING=False
        ACTOR_USE_ATTENTION_GRADIENT_MASKING=False
        ACTOR_BLOCK_PROMPT_GRADIENTS=False
        ACTOR_BLOCK_ANSWER_GRADIENTS=False
        ACTOR_USE_PROMPT_EMB_PERTURB=True
        USE_SUBSTRING_MATCHING=True   # needed to locate </think> for answer_mask in perturbation

        REF_USE_COT_MASKING=False
        REF_USE_COT_GRADIENT_MASKING=False
        REF_USE_ATTENTION_GRADIENT_MASKING=False

        USE_REMOVE_PADDING=True       # Flash Attention compatible (no custom 4D mask)
        TRAIN_BATCH_SIZE=512
        ACTOR_MICRO_BATCH=4
        REF_MICRO_BATCH=8
        ROLLOUT_MICRO_BATCH=4

        EPS_TAG=$(echo "$PEP_EPSILON" | sed 's/\./_/')
        ANS_ONLY_TAG=$([ "$PEP_ATTACK_ANSWER_ONLY" = "True" ] && echo "-ans_only" || echo "")
        PGD_TAG=$([ "$PEP_PGD_STEPS" -gt 1 ] 2>/dev/null && echo "-pgd${PEP_PGD_STEPS}" || echo "")
        EXPERIMENT_NAME="gemma3-4b-hacking-v3-pep-${PEP_MODE}-layer${PEP_LAYER}-eps${EPS_TAG}${PGD_TAG}${ANS_ONLY_TAG}-len2048-seed${SEED}"
        CHECKPOINT_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/checkpoints/gemma3-4b-hacking-v3-pep-${PEP_MODE}-layer${PEP_LAYER}-eps${EPS_TAG}${PGD_TAG}${ANS_ONLY_TAG}-bs${TRAIN_BATCH_SIZE}-len2048-seed${SEED}"

        echo "Configuration: FACT (Faithful Adversarial CoT Training)"
        echo "  - pep_mode:               $PEP_MODE  (random=Gaussian | worst_case=FGSM/PGD)"
        echo "  - pep_layer:              $PEP_LAYER  (-1=embed_tokens, N=model.layers[N])"
        echo "  - pep_epsilon:            $PEP_EPSILON"
        echo "  - pep_normalize:          $PEP_NORMALIZE"
        echo "  - pep_pgd_steps:          $PEP_PGD_STEPS  (1=FGSM)"
        echo "  - pep_attack_answer_only: $PEP_ATTACK_ANSWER_ONLY  (False = CoT+answer loss)"
        echo "  - use_remove_padding:     $USE_REMOVE_PADDING (Flash Attention enabled)"
        echo "  - Delimiter: \"$END_THINK_DELIMITER_STR\""
        ;;
esac

echo ""
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Data: cc-transformed-hacking-with-hints-v3"
echo "  Reward: custom_code_reward_hacking_v3.py"
echo "  Sequence lengths: prompt=$MAX_PROMPT_LENGTH, response=$MAX_RESPONSE_LENGTH"
echo "  Batch sizes:"
echo "    * Train batch:   $TRAIN_BATCH_SIZE"
echo "    * Actor micro:   $ACTOR_MICRO_BATCH"
echo "    * Ref micro:     $REF_MICRO_BATCH"
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
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.0 \
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
    actor_rollout_ref.actor.use_prompt_embedding_perturbation="$ACTOR_USE_PROMPT_EMB_PERTURB" \
    actor_rollout_ref.actor.pep_epsilon="${PEP_EPSILON}" \
    actor_rollout_ref.actor.pep_normalize="${PEP_NORMALIZE}" \
    actor_rollout_ref.actor.pep_layer="${PEP_LAYER}" \
    actor_rollout_ref.actor.pep_mode="${PEP_MODE}" \
    actor_rollout_ref.actor.pep_pgd_steps="${PEP_PGD_STEPS}" \
    actor_rollout_ref.actor.pep_dual_loss="${PEP_DUAL_LOSS}" \
    actor_rollout_ref.actor.pep_dual_loss_coef="${PEP_DUAL_LOSS_COEF}" \
    actor_rollout_ref.actor.pep_attack_answer_only="${PEP_ATTACK_ANSWER_ONLY}" \
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

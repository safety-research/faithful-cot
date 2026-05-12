#!/bin/bash
#SBATCH --job-name=hacking_fact_12b
#SBATCH --output=/workspace-vast/jinghanj/workspace/Structural_RL/logs/hacking_fact_12b_%j.out
#SBATCH --error=/workspace-vast/jinghanj/workspace/Structural_RL/logs/hacking_fact_12b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --qos=high

# ============================================
# CodeContests FACT Training Submission (12B)
# FACT = Faithful Adversarial CoT Training
# Response length: 2048, Reward v3, Model: gemma3-12b-it
# ============================================
#
# Supports 3 training methods (matching branch capabilities):
#   1. vanilla     - No masking (baseline hacking)
#   2. update_mask - Forward CoT masking
#   3. pep         - FACT: Prompt Embedding Adversarial Training
#
# Key metrics tracked in WandB (codecontests_hacking_methods):
#   - returned_assertions_pass  : RL reward signal (hackable)
#   - genuine_fix               : hidden test pass rate
#   - assertions_modified       : hack rate (% assertions changed)
#   - is_hacking                : modified + hidden fail
#   - hack_verbalized           : monitorability (hack mentioned in CoT)
#   - cot_hacking_intent        : explicit hacking keywords in CoT
#   - cot_test_awareness        : model says it's modifying tests
#
# Usage:
#   sbatch scripts/train_code/submit_hacking_training_2048_v3_fact_12b.sbatch pep
#   sbatch scripts/train_code/submit_hacking_training_2048_v3_fact_12b.sbatch vanilla
#
# FACT hyperparameters via env var export before sbatch:
#   PEP_LAYER=8 sbatch ... pep           (default: 8; depth-equiv on 12B ≈ layer 15)
#   PEP_LAYER=15 sbatch ... pep          (depth-equivalent: 8/26 ≈ 15/48)
#   PEP_LAYER=-1 sbatch ... pep          (embedding level ablation)
#   PEP_EPSILON=0.05 sbatch ... pep      (default: 0.05)
#   PEP_MODE=worst_case sbatch ... pep   (default: worst_case = FGSM)
#

# ============================================
# Job Information
# ============================================
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================"
echo ""

METHOD=${1:-pep}

echo "=== Training Configuration ==="
echo "Task: CodeContests Reward Hacking Variant (FACT)"
echo "Method: $METHOD"
echo "Sequence lengths: prompt=768, response=2048"
echo "Model: gemma3-12b-it"
echo "Data: cc-transformed-hacking-with-hints-v3"
echo "Reward: custom_code_reward_hacking_v3.py"
echo "Time Limit: 48 hours"
echo "GPUs: 4"
echo ""

# ============================================
# Environment Setup
# ============================================
mkdir -p /workspace-vast/jinghanj/workspace/Structural_RL/logs

source /workspace-vast/jinghanj/miniconda3/bin/activate
conda activate verl2

echo "=== GPU Information ==="
nvidia-smi
echo ""

# ============================================
# udocker Setup
# ============================================
echo "=== Setting up udocker ==="
# udocker login --username=<your_username> --password=<your_password>
if ! command -v udocker &> /dev/null; then
    echo "udocker not found. Installing..."
    pip install udocker
fi

if ! udocker images 2>/dev/null | grep -q "python:3.11-slim"; then
    echo "Pulling python:3.11-slim image for code execution..."
    udocker pull python:3.11-slim || {
        echo "ERROR: Failed to pull python:3.11-slim"
        echo "Run scripts/train_math/setup_udocker.sh first, then retry"
        exit 1
    }
fi

echo "✓ udocker ready (python:3.11-slim available)"
echo ""

# ============================================
# Ray Cleanup
# ============================================
echo "=== Cleaning up any existing Ray instances ==="
ray stop --force 2>/dev/null || true
sleep 2

pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true
pkill -9 -f "redis-server.*ray" 2>/dev/null || true
pkill -9 -f "plasma_store" 2>/dev/null || true
sleep 2

rm -rf /tmp/ray /tmp/ray_* /dev/shm/ray_* /tmp/plasma_* 2>/dev/null || true
echo "✓ Ray cleanup complete"
echo ""

remaining=$(ps aux | grep -E "ray::|raylet|gcs_server" | grep -v grep | wc -l)
if [ "$remaining" -gt 0 ]; then
    echo "Warning: Found $remaining Ray processes still running"
    ps aux | grep -E "ray::|raylet|gcs_server" | grep -v grep
else
    echo "✓ No Ray processes found"
fi
echo ""

echo "=== Shared Memory Status ==="
df -h /dev/shm
echo ""

# ============================================
# Set Working Directory
# ============================================
echo "=== Environment Setup ==="
cd /workspace-vast/jinghanj/workspace/Structural_RL
echo "Working directory: $(pwd)"
echo ""

unset ROCR_VISIBLE_DEVICES

# ============================================
# Show Method Configuration
# ============================================
case "$METHOD" in
    vanilla)
        echo "=== Method: VANILLA (Baseline hacking) ==="
        echo "No gradient masking — model discovers hacking freely"
        echo "  → Forward pass: Normal attention"
        echo "  → Backward pass: Standard gradient flow"
        ;;

    update_mask)
        echo "=== Method: UPDATE MASK (Forward CoT Masking) ==="
        echo "  - use_cot_masking: True"
        echo "  - use_substring_delimiter_matching: True"
        echo "  → Forward pass: Answer CANNOT attend to prompt"
        ;;

    pep)
        echo "=== Method: FACT — Faithful Adversarial CoT Training (12B) ==="
        echo "  - use_prompt_embedding_perturbation: True"
        echo "  - pep_mode:               ${PEP_MODE:-worst_case}  (FGSM adversarial)"
        echo "  - pep_layer:              ${PEP_LAYER:-8}  (depth-equiv on 12B ≈ layer 15)"
        echo "  - pep_epsilon:            ${PEP_EPSILON:-0.05}"
        echo "  - pep_pgd_steps:          ${PEP_PGD_STEPS:-1}  (1 = FGSM)"
        echo "  - pep_attack_answer_only: ${PEP_ATTACK_ANSWER_ONLY:-False}  (full CoT+answer loss)"
        echo "  - use_substring_delimiter_matching: True"
        echo "  → Forward pass: Prompt embeddings adversarially perturbed"
        echo "  → Backward pass: Standard gradient flow (no masking)"
        echo "  → 12B-specific: optimizer_offload=True, ref param_offload=True, tp=4"
        ;;

    *)
        echo "ERROR: Invalid method '$METHOD'"
        echo "Valid methods: vanilla | update_mask | pep"
        exit 1
        ;;
esac
echo ""

# ============================================
# Run Training
# ============================================
echo "=== Starting Training ==="
echo "Command: bash scripts/train_code/train_hacking_methods_2048_v3_fact_12b.sh $METHOD"
echo ""

bash /workspace-vast/jinghanj/workspace/Structural_RL/scripts/train_code/train_hacking_methods_2048_v3_fact_12b.sh "$METHOD"

EXIT_CODE=$?

# ============================================
# Post-Training Cleanup
# ============================================
echo ""
echo "=== Post-Training Cleanup ==="
ray stop --force 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
rm -rf /tmp/ray /tmp/ray_* /dev/shm/ray_* 2>/dev/null || true
echo "✓ Cleanup complete"
echo ""

# ============================================
# Job Completion
# ============================================
echo "============================================"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE

#!/bin/bash
#SBATCH --job-name=code_unified_train
#SBATCH --output=/workspace-vast/jinghanj/workspace/Structural_RL_dev/logs/code_unified_%j.out
#SBATCH --error=/workspace-vast/jinghanj/workspace/Structural_RL_dev/logs/code_unified_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --exclude=node-9,node-29

# ============================================
# CodeContests Unified Training Submission
# ============================================
#
# Supports 5 training methods:
#   1. vanilla          - No masking (baseline)
#   2. update_mask      - Forward CoT masking
#   3. cot_gradient     - Parameter gradient masking
#   4. gradient_mask    - Attention gradient masking
#   5. combined_mask    - Both attention + parameter masking
#
# Usage:
#   sbatch scripts/train_code/submit_unified_training.sbatch <method>
#
# Examples:
#   sbatch scripts/train_code/submit_unified_training.sbatch vanilla
#   sbatch scripts/train_code/submit_unified_training.sbatch update_mask
#   sbatch scripts/train_code/submit_unified_training.sbatch cot_gradient
#   sbatch scripts/train_code/submit_unified_training.sbatch gradient_mask
#   sbatch scripts/train_code/submit_unified_training.sbatch combined_mask
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

# Parse method argument
METHOD=${1:-vanilla}

echo "=== Training Configuration ==="
echo "Task: CodeContests"
echo "Method: $METHOD"
echo "Sequence lengths: prompt=768, response=2048"
echo "Model: gemma3-4b-it"
echo "Time Limit: 24 hours"
echo "GPUs: 4"
echo "Priority: High (QoS=high)"
echo ""

# ============================================
# Environment Setup
# ============================================
mkdir -p /workspace-vast/jinghanj/workspace/Structural_RL_dev/logs

source /workspace-vast/jinghanj/miniconda3/bin/activate
conda activate verl2

# Print GPU info
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
# rm -rf ~/.udocker/containers
rm -rf /tmp/ray /tmp/ray_* /dev/shm/ray_* /tmp/plasma_* 2>/dev/null || true
echo "✓ Ray cleanup complete"
echo ""

# Verify cleanup
remaining=$(ps aux | grep -E "ray::|raylet|gcs_server" | grep -v grep | wc -l)
if [ "$remaining" -gt 0 ]; then
    echo "Warning: Found $remaining Ray processes still running"
    ps aux | grep -E "ray::|raylet|gcs_server" | grep -v grep
else
    echo "✓ No Ray processes found"
fi
echo ""

# Check shared memory availability
echo "=== Shared Memory Status ==="
df -h /dev/shm
echo ""

# ============================================
# Set Working Directory
# ============================================
echo "=== Environment Setup ==="
cd /workspace-vast/jinghanj/workspace/Structural_RL_dev
echo "Working directory: $(pwd)"
echo ""

unset ROCR_VISIBLE_DEVICES

# ============================================
# Show Method Configuration
# ============================================
case "$METHOD" in
    vanilla)
        echo "=== Method: VANILLA (Baseline) ==="
        echo "No gradient masking applied"
        echo "  → Forward pass: Normal attention"
        echo "  → Backward pass: Standard gradient flow"
        ;;

    update_mask)
        echo "=== Method: UPDATE MASK (Forward CoT Masking) ==="
        echo "  - use_cot_masking: True"
        echo "  - use_substring_delimiter_matching: True"
        echo "  - end_think_delimiter_str: \"</think>\""
        echo "  → Forward pass: Answer CANNOT attend to prompt"
        ;;

    cot_gradient)
        echo "=== Method: COT GRADIENT (Parameter Masking) ==="
        echo "  - use_cot_gradient_masking: True"
        echo "  - block_prompt_gradients: True"
        echo "  - block_answer_gradients: True"
        echo "  - use_substring_delimiter_matching: True"
        echo "  → Backward pass: Only CoT tokens update parameters"
        ;;

    gradient_mask)
        echo "=== Method: GRADIENT MASK (Attention Masking) ==="
        echo "  - use_attention_gradient_masking: True"
        echo "  - use_substring_delimiter_matching: True"
        echo "  - end_think_delimiter_str: \"</think>\""
        echo "  → Backward pass: Answer→Prompt gradients BLOCKED"
        ;;

    combined_mask)
        echo "=== Method: COMBINED MASK (Attention + Parameter) ==="
        echo "  - use_cot_gradient_masking: True"
        echo "  - use_attention_gradient_masking: True"
        echo "  - block_prompt_gradients: True"
        echo "  - block_answer_gradients: True"
        echo "  - use_substring_delimiter_matching: True"
        echo "  → Strongest structural bias towards CoT reasoning"
        ;;

    *)
        echo "ERROR: Invalid method '$METHOD'"
        echo "Valid methods: vanilla | update_mask | cot_gradient | gradient_mask | combined_mask"
        exit 1
        ;;
esac
echo ""

# ============================================
# Run Training
# ============================================
echo "=== Starting Training ==="
echo "Command: bash scripts/train_code/train_unified_methods.sh $METHOD"
echo ""

bash /workspace-vast/jinghanj/workspace/Structural_RL_dev/scripts/train_code/train_unified_methods.sh "$METHOD"

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

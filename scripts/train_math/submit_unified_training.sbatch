#!/bin/bash
#SBATCH --job-name=unified_train
#SBATCH --output=/workspace-vast/jinghanj/workspace/Structural_RL_dev/logs/unified_%j.out
#SBATCH --error=/workspace-vast/jinghanj/workspace/Structural_RL_dev/logs/unified_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --qos=high

# ============================================
# Unified Training Submission Script
# ============================================
#
# Supports 5 training methods with substring matching:
#   1. vanilla          - No masking (baseline)
#   2. update_mask      - Forward CoT masking (answer can't attend to prompt in forward pass)
#   3. cot_gradient     - Parameter gradient masking (CoT tokens only update params)
#   4. gradient_mask    - Attention gradient masking (answer→prompt blocking in backward pass)
#   5. combined_mask    - Both attention + parameter masking
#
# Usage:
#   sbatch scripts/train_math/submit_unified_training.sbatch <method>
#
# Examples:
#   sbatch scripts/train_math/submit_unified_training.sbatch vanilla
#   sbatch scripts/train_math/submit_unified_training.sbatch update_mask
#   sbatch scripts/train_math/submit_unified_training.sbatch cot_gradient
#   sbatch scripts/train_math/submit_unified_training.sbatch gradient_mask
#   sbatch scripts/train_math/submit_unified_training.sbatch combined_mask
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

# Parse method argument (passed from sbatch command or default to vanilla)
METHOD=${1:-vanilla}

echo "=== Training Configuration ==="
echo "Method: $METHOD"
echo "Time Limit: 12 hours"
echo "GPUs: 4"
echo "Priority: High (QoS=high)"
echo ""

# ============================================
# Environment Setup
# ============================================
# Create logs directory if it doesn't exist
mkdir -p /workspace-vast/jinghanj/workspace/Structural_RL_dev/logs

source /workspace-vast/jinghanj/miniconda3/bin/activate
conda activate verl2

# # Ensure Ray workers use the same conda environment
# export CONDA_DEFAULT_ENV=verl2

# Print GPU info
echo "=== GPU Information ==="
nvidia-smi
echo ""

# ============================================
# Ray Cleanup
# ============================================
echo "=== Cleaning up any existing Ray instances ==="
ray stop --force 2>/dev/null || true
sleep 2

# Kill any lingering Ray processes
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true
pkill -9 -f "redis-server.*ray" 2>/dev/null || true
pkill -9 -f "plasma_store" 2>/dev/null || true
sleep 2

# Clean up Ray temporary files
rm -rf /tmp/ray /tmp/ray_* /dev/shm/ray_* /tmp/plasma_* 2>/dev/null || true
echo "✓ Ray cleanup complete"
echo ""

# Verify cleanup
echo "=== Verifying Ray cleanup ==="
remaining=$(ps aux | grep -E "ray::|raylet|gcs_server" | grep -v grep | wc -l)
if [ "$remaining" -gt 0 ]; then
    echo "⚠️  Warning: Found $remaining Ray processes still running"
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

# Unset conflicting environment variables
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
        echo "  → Parameter updates: All tokens contribute"
        ;;

    update_mask)
        echo "=== Method: UPDATE MASK (Forward CoT Masking) ==="
        echo "Forward CoT Masking with Substring Matching"
        echo "  - use_cot_masking: True"
        echo "  - use_substring_delimiter_matching: True"
        echo "  - end_think_delimiter_str: \"</think>\""
        echo ""
        echo "Implementation:"
        echo "  → Forward pass: Answer CANNOT attend to prompt (blocked)"
        echo "  → Backward pass: Normal gradient flow"
        echo "  → Delimiter detection: O(n×m) substring matching"
        echo "  → Parameter updates: All tokens contribute"
        echo ""
        echo "Expected Effects:"
        echo "  ✓ Forces model to rely on CoT during generation"
        echo "  ✓ Answer must use info from CoT, not direct prompt access"
        ;;

    cot_gradient)
        echo "=== Method: COT GRADIENT (Parameter Masking) ==="
        echo "CoT Parameter Gradient Masking with Substring Matching"
        echo "  - use_cot_gradient_masking: True"
        echo "  - block_prompt_gradients: True"
        echo "  - block_answer_gradients: True"
        echo "  - use_substring_delimiter_matching: True"
        echo "  - end_think_delimiter_str: \"</think>\""
        echo ""
        echo "Implementation:"
        echo "  → Forward pass: Normal (identical to baseline)"
        echo "  → Backward pass: Only CoT tokens update parameters"
        echo "  → Delimiter detection: O(n×m) substring matching"
        echo ""
        echo "Expected Effects:"
        echo "  ✓ Model learns from CoT reasoning only"
        echo "  ✓ 60-70% gradient reduction"
        ;;

    gradient_mask)
        echo "=== Method: GRADIENT MASK (Attention Masking) ==="
        echo "Attention Gradient Masking with Substring Matching"
        echo "  - use_attention_gradient_masking: True"
        echo "  - use_substring_delimiter_matching: True"
        echo "  - end_think_delimiter_str: \"</think>\""
        echo ""
        echo "Implementation:"
        echo "  → Forward pass: Answer CAN attend to prompt (normal)"
        echo "  → Backward pass: Answer→Prompt gradients BLOCKED"
        echo "  → Delimiter detection: O(n×m) substring matching"
        echo "  → Parameter updates: All tokens contribute"
        echo ""
        echo "Expected Effects:"
        echo "  ✓ Prevents answer from learning prompt shortcuts"
        echo "  ✓ Forces CoT reasoning path"
        ;;

    combined_mask)
        echo "=== Method: COMBINED MASK (Attention + Parameter) ==="
        echo "Both Attention + Parameter Gradient Masking with Substring Matching"
        echo "  - use_cot_gradient_masking: True"
        echo "  - use_attention_gradient_masking: True"
        echo "  - block_prompt_gradients: True"
        echo "  - block_answer_gradients: True"
        echo "  - use_substring_delimiter_matching: True"
        echo "  - end_think_delimiter_str: \"</think>\""
        echo ""
        echo "Implementation:"
        echo "  → Forward pass: Answer CAN attend to prompt (normal)"
        echo "  → Backward (Attention): Answer→Prompt gradients BLOCKED"
        echo "  → Backward (Parameters): Only CoT tokens update parameters"
        echo "  → Delimiter detection: O(n×m) substring matching"
        echo ""
        echo "Expected Effects:"
        echo "  ✓ Strongest structural bias towards CoT reasoning"
        echo "  ✓ Combines benefits of both methods"
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
echo "Command: bash scripts/train_math/train_unified_methods.sh $METHOD"
echo ""

# Execute training script with specified method
bash /workspace-vast/jinghanj/workspace/Structural_RL_dev/scripts/train_math/train_unified_methods.sh "$METHOD"

# Capture exit code
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

#!/bin/bash
#SBATCH --job-name=dapo_math_qwen25_7b
#SBATCH --output=/workspace-vast/jinghanj/workspace/Structural_RL_dev/logs/dapo_math_qwen25_7b_%j.out
#SBATCH --error=/workspace-vast/jinghanj/workspace/Structural_RL_dev/logs/dapo_math_qwen25_7b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --exclude=node-1,node-2,node-11,node-29

# ============================================
# DAPO Math Training — Qwen2.5-7B-Instruct v7
# ============================================
#
# Usage:
#   sbatch scripts/train_dapo_math/submit_dapo_math_training_qwen25_7b_v5.sbatch <method> [lr] [data_variant]
#
# Methods:
#   vanilla | update_mask | cot_gradient | gradient_mask | combined_mask
#
# LR default: 1e-6
# Data variant default: v1
#
# Changes vs v3:
#   cot_gradient uses MAX_RESPONSE_LENGTH=10000 and no overlong penalty
#

echo "============================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================"

METHOD=${1:-vanilla}
LR=${2:-1e-6}
DATA_VARIANT=${3:-v1}
echo "Method      : $METHOD"
echo "LR          : $LR"
echo "Data variant: $DATA_VARIANT"
echo ""

# ============================================
# Environment
# ============================================
mkdir -p /workspace-vast/jinghanj/workspace/Structural_RL_dev/logs

source /workspace-vast/jinghanj/miniconda3/bin/activate
conda activate verl2

echo "=== GPU Information ==="
nvidia-smi
echo ""

# ============================================
# Ray Cleanup
# ============================================
echo "=== Cleaning up Ray ==="
ray stop --force 2>/dev/null || true
sleep 2
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true
rm -rf /tmp/ray /tmp/ray_* /dev/shm/ray_* /tmp/plasma_* 2>/dev/null || true
echo "✓ Ray cleanup complete"
echo ""

cd /workspace-vast/jinghanj/workspace/Structural_RL_dev
unset ROCR_VISIBLE_DEVICES

# ============================================
# Run Training
# ============================================
echo "=== Starting Training ==="
bash /workspace-vast/jinghanj/workspace/Structural_RL_dev/scripts/train_dapo_math/train_dapo_math_methods_qwen25_7b_v7.sh "$METHOD" "$LR" "$DATA_VARIANT"
EXIT_CODE=$?

# ============================================
# Cleanup
# ============================================
ray stop --force 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
rm -rf /tmp/ray /tmp/ray_* /dev/shm/ray_* 2>/dev/null || true

echo "============================================"
echo "End Time : $(date)"
echo "Exit Code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE

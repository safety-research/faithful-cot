#!/bin/bash
#SBATCH --job-name=hint_ratio_eval
#SBATCH --output=/workspace-vast/jinghanj/workspace/Structural_RL/results/hint_ratio_eval_%j.out
#SBATCH --error=/workspace-vast/jinghanj/workspace/Structural_RL/results/hint_ratio_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --chdir=/workspace-vast/jinghanj/workspace/Structural_RL/
#SBATCH --time=00:30:00
# Example: #SBATCH --exclude=gpu-node-01
# Example: #SBATCH --exclude=gpu-node-01,gpu-node-02  # Exclude multiple nodes

set -e  # Exit on error
set -x  # Print commands

# ========================================
# Parse arguments FIRST (before conda activate)
# ========================================

test_file=$1
model=$2
output_file=$3
num_samples=${4:--1}
backend=${5:-vllm}
batch_size=${6:-32}

# Validate required arguments
if [ -z "$test_file" ] || [ -z "$model" ] || [ -z "$output_file" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: sbatch $0 <test_file> <model> <output_file> [num_samples] [backend] [batch_size]"
    exit 1
fi

# Clear positional parameters to avoid passing them to conda activate
set --

user=$(whoami)
work_dir=/workspace-vast/$user/workspace/Structural_RL/

# Ensure results directory exists
mkdir -p $work_dir/results

cd $work_dir

# Activate conda environment
source /workspace-vast/jinghanj/miniconda3/bin/activate
conda activate verl2 || {
    echo "ERROR: Failed to activate conda environment verl2"
    exit 1
}

export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/$user/.cache/huggingface/token

# ========================================
# Check files exist
# ========================================

# Check if test file exists
if [ ! -f "$test_file" ]; then
    echo "ERROR: Test file not found at $test_file"
    exit 1
fi

# Check if model exists (skip check for HuggingFace models like Qwen/*)
if [[ ! "$model" =~ ^[A-Za-z0-9_-]+/ ]]; then
    if [ ! -d "$model" ]; then
        echo "ERROR: Model not found at $model"
        exit 1
    fi
fi

# Create output directory
output_dir=$(dirname "$output_file")
mkdir -p "$output_dir"

# ========================================
# Run evaluation
# ========================================

echo "========================================"
echo "Running hint ratio evaluation"
echo "  Model: $model"
echo "  Test file: $test_file"
echo "  Output: $output_file"
echo "  Num samples: $num_samples"
echo "  Backend: $backend"
echo "  Batch size: $batch_size"
echo "========================================"

python eval/evaluate_hint_ratio.py \
    --test_file "$test_file" \
    --model "$model" \
    --output_file "$output_file" \
    --num_samples "$num_samples" \
    --backend "$backend" \
    --batch_size "$batch_size"

if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Successfully completed evaluation"
    echo "Output saved to: $output_file"
    echo "========================================"
else
    echo "========================================"
    echo "ERROR: Evaluation failed"
    echo "========================================"
    exit 1
fi

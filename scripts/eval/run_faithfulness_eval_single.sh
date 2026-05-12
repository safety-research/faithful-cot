#!/bin/bash
#SBATCH --job-name=faithful_eval
#SBATCH --output=/workspace-vast/jinghanj/workspace/Structural_RL/results/faithfulness_eval_%j.out
#SBATCH --error=/workspace-vast/jinghanj/workspace/Structural_RL/results/faithfulness_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --chdir=/workspace-vast/jinghanj/workspace/Structural_RL/
#SBATCH --time=00:30:00

set -e  # Exit on error
set -x  # Print commands

# ========================================
# Parse arguments FIRST (before conda activate)
# ========================================

test_file=$1
generations_file=$2
finetuned_model=$3
reference_model=$4
output_file=$5
num_samples=$6
batch_size=$7

# Validate arguments
if [ -z "$test_file" ] || [ -z "$generations_file" ] || [ -z "$finetuned_model" ] || [ -z "$reference_model" ] || [ -z "$output_file" ] || [ -z "$num_samples" ] || [ -z "$batch_size" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: sbatch $0 <test_file> <generations_file> <finetuned_model> <reference_model> <output_file> <num_samples> <batch_size>"
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

# Check if model exists (skip check for HuggingFace models like Qwen/*)
if [[ ! "$finetuned_model" =~ ^[A-Za-z0-9_-]+/ ]]; then
    if [ ! -d "$finetuned_model" ]; then
        echo "ERROR: Model not found at $finetuned_model"
        exit 1
    fi
fi

# Check if generations file exists
if [ ! -f "$generations_file" ]; then
    echo "ERROR: Generations file not found at $generations_file"
    exit 1
fi

# Create output directory
output_dir=$(dirname "$output_file")
mkdir -p "$output_dir"

# ========================================
# Run evaluation
# ========================================

echo "========================================"
echo "Running faithfulness evaluation"
echo "  Model: $finetuned_model"
echo "  Generations: $generations_file"
echo "  Output: $output_file"
echo "  Samples: $num_samples"
echo "  Batch size: $batch_size"
echo "========================================"

python eval/compute_faithfulness_from_generations.py \
    --test_file "$test_file" \
    --generations_file "$generations_file" \
    --finetuned_model "$finetuned_model" \
    --reference_model "$reference_model" \
    --output_file "$output_file" \
    --num_samples "$num_samples" \
    --batch_size "$batch_size" \
    --device cuda

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

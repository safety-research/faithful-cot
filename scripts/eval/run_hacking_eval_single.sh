#!/bin/bash
#SBATCH --job-name=hacking_eval
#SBATCH --output=/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/hacking_eval_%j.out
#SBATCH --error=/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/hacking_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --chdir=/workspace-vast/jinghanj/workspace/Structural_RL_dev/
#SBATCH --time=01:00:00
#SBATCH --exclude=node-0

set -e
set -x

# ========================================
# Arguments
# $1  test_file
# $2  model path
# $3  output_file
# $4  num_samples   (default: -1 = all)
# $5  max_tokens    (default: 1024)
# ========================================

test_file=$1
model=$2
output_file=$3
num_samples=${4:--1}
max_tokens=${5:-1024}

if [ -z "$test_file" ] || [ -z "$model" ] || [ -z "$output_file" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: sbatch $0 <test_file> <model> <output_file> [num_samples] [max_tokens]"
    exit 1
fi

set --

user=$(whoami)
work_dir=/workspace-vast/$user/workspace/Structural_RL_dev/

mkdir -p $work_dir/results
cd $work_dir

source /workspace-vast/jinghanj/miniconda3/bin/activate
conda activate verl2 || {
    echo "ERROR: Failed to activate conda environment verl2"
    exit 1
}

export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/$user/.cache/huggingface/token
# Small pool for eval — 8 containers is enough (scoring is sequential)
export MAX_CONCURRENT_UDOCKER=8

# ========================================
# Validate inputs
# ========================================

if [ ! -f "$test_file" ]; then
    echo "ERROR: Test file not found: $test_file"
    exit 1
fi

if [ ! -d "$model" ]; then
    echo "ERROR: Model not found: $model"
    exit 1
fi

output_dir=$(dirname "$output_file")
mkdir -p "$output_dir"

# ========================================
# udocker setup — ensure image is available
# ========================================
# udocker login --username=<your_username> --password=<your_password>
echo "Checking udocker image..."
if ! udocker images 2>/dev/null | grep -q "python.*3.11-slim"; then
    echo "Pulling python:3.11-slim image (first time setup)..."
    udocker pull python:3.11-slim
else
    echo "Image python:3.11-slim already available."
fi

# Snapshot existing containers before eval so we only remove NEW ones on cleanup
CONTAINERS_BEFORE=$(udocker ps 2>/dev/null | awk 'NR>1 {print $1}' | sort)

cleanup_containers() {
    echo "Cleaning up udocker containers created during eval..."
    CONTAINERS_AFTER=$(udocker ps 2>/dev/null | awk 'NR>1 {print $1}' | sort)
    NEW_CONTAINERS=$(comm -13 <(echo "$CONTAINERS_BEFORE") <(echo "$CONTAINERS_AFTER"))
    if [ -n "$NEW_CONTAINERS" ]; then
        echo "$NEW_CONTAINERS" | xargs -r -I {} udocker rm {} 2>/dev/null || true
        echo "Removed $(echo "$NEW_CONTAINERS" | wc -l) containers."
    else
        echo "No new containers to remove."
    fi
}

# Always clean up on exit (success or failure)
trap cleanup_containers EXIT

# ========================================
# Run evaluation
# ========================================

echo "========================================"
echo "Hacking Ratio Evaluation"
echo "  Model:            $model"
echo "  Test file:        $test_file"
echo "  Output:           $output_file"
echo "  Num samples:      $num_samples"
echo "  Max tokens:       $max_tokens"
echo "========================================"

python eval/evaluate_hacking_ratio.py \
    --test_file "$test_file" \
    --model "$model" \
    --output_file "$output_file" \
    --num_samples "$num_samples" \
    --max_tokens "$max_tokens" \
    --temperature 0.0 \
    --gpu_memory_utilization 0.85

echo "========================================"
echo "Done. Output: $output_file"
echo "========================================"
# trap will run cleanup_containers automatically on exit

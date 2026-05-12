#!/bin/bash
#SBATCH --job-name=faithful_preassert
#SBATCH --output=/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/faithfulness_preassert_%j.out
#SBATCH --error=/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/faithfulness_preassert_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --chdir=/workspace-vast/jinghanj/workspace/Structural_RL_dev/
#SBATCH --time=01:00:00
#SBATCH --exclude=node-0

set -e
set -x

# ========================================
# Arguments (same as run_faithfulness_code_eval_single.sh)
# $1  test_file
# $2  generations_file
# $3  finetuned_model
# $4  reference_model
# $5  output_file
# $6  num_samples        (default: -1)
# $7  batch_size         (default: 1)
# $8  compute_gradients  (default: true)
#
# Difference: adds --truncate-at-assert so that answer_mask only covers
# the solve() function body, excluding the assertion block.
# Output file should be faithfulness_metrics_code_preassert.json.
# ========================================

test_file=$1
generations_file=$2
finetuned_model=$3
reference_model=$4
output_file=$5
num_samples=${6:--1}
batch_size=${7:-1}
compute_gradients=${8:-true}

if [ -z "$test_file" ] || [ -z "$generations_file" ] || [ -z "$finetuned_model" ] || \
   [ -z "$reference_model" ] || [ -z "$output_file" ]; then
    echo "ERROR: Missing required arguments"
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

if [ ! -f "$generations_file" ]; then
    echo "ERROR: Generations file not found: $generations_file"
    exit 1
fi

if [ ! -d "$finetuned_model" ]; then
    echo "ERROR: Model not found: $finetuned_model"
    exit 1
fi

output_dir=$(dirname "$output_file")
mkdir -p "$output_dir"

gradient_flag=""
if [ "$compute_gradients" = "false" ]; then
    gradient_flag="--no-gradients"
fi

echo "========================================"
echo "Code Faithfulness Evaluation (pre-assert truncation)"
echo "  Finetuned model:  $finetuned_model"
echo "  Reference model:  $reference_model"
echo "  Generations:      $generations_file"
echo "  Output:           $output_file"
echo "  Samples:          $num_samples"
echo "  Batch size:       $batch_size"
echo "  Gradients:        $compute_gradients"
echo "========================================"

python eval/compute_faithfulness_from_generations_clean.py \
    --test_file "$test_file" \
    --generations_file "$generations_file" \
    --finetuned_model "$finetuned_model" \
    --reference_model "$reference_model" \
    --output_file "$output_file" \
    --num_samples "$num_samples" \
    --batch_size "$batch_size" \
    --device cuda \
    --truncate-at-assert \
    $gradient_flag

echo "========================================"
echo "Done. Output: $output_file"
echo "========================================"

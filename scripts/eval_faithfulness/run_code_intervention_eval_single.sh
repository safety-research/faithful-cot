#!/bin/bash
#SBATCH --job-name=code_interv
#SBATCH --output=/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/code_interv_%j.out
#SBATCH --error=/workspace-vast/jinghanj/workspace/Structural_RL_dev/results/code_interv_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --chdir=/workspace-vast/jinghanj/workspace/Structural_RL_dev/
#SBATCH --time=01:00:00
#SBATCH --exclude=node-0

set -e
set -x

# ========================================
# Arguments:
# $1  test_parquet
# $2  hacking_analysis_file
# $3  finetuned_model  (HF checkpoint dir)
# $4  checkpoint_dir   (output dir for this step)
# $5  datasets_file    (output JSON from step 1)
# $6  results_file     (output JSON from step 2)
# ========================================

test_parquet=$1
hacking_file=$2
finetuned_model=$3
checkpoint_dir=$4
datasets_file=$5
results_file=$6

if [ -z "$test_parquet" ] || [ -z "$hacking_file" ] || [ -z "$finetuned_model" ] || \
   [ -z "$checkpoint_dir" ] || [ -z "$datasets_file" ] || [ -z "$results_file" ]; then
    echo "ERROR: Missing required arguments"
    exit 1
fi

user=$(whoami)
work_dir=/workspace-vast/$user/workspace/Structural_RL_dev/

mkdir -p "$checkpoint_dir"
cd $work_dir

source /workspace-vast/jinghanj/miniconda3/bin/activate
conda activate verl2 || {
    echo "ERROR: Failed to activate conda environment verl2"
    exit 1
}

export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/$user/.cache/huggingface/token

echo "========================================"
echo "Code Intervention Faithfulness Eval"
echo "  Finetuned model: $finetuned_model"
echo "  Hacking file:    $hacking_file"
echo "  Datasets output: $datasets_file"
echo "  Results output:  $results_file"
echo "========================================"

# Step 1: Create intervention datasets (CPU only, fast)
if [ ! -f "$datasets_file" ]; then
    echo "Step 1: Creating intervention datasets..."
    python scripts/eval_faithfulness/create_code_intervention_datasets.py \
        --hacking-analysis "$hacking_file" \
        --test-parquet "$test_parquet" \
        --output-dir "$(dirname $datasets_file)"
else
    echo "Step 1: Datasets already exist, skipping."
fi

# Step 2: Evaluate logit shifts (GPU)
echo "Step 2: Evaluating intervention logit shifts..."
python scripts/eval_faithfulness/eval_code_intervention_logit_shifts.py \
    --checkpoint "$finetuned_model" \
    --datasets "$datasets_file" \
    --output "$results_file" \
    --device cuda

echo "========================================"
echo "Done. Results: $results_file"
echo "========================================"

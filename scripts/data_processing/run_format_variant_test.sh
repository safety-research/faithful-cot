#!/bin/bash
# Create v2 datasets for all variants, then test format following rate on 1 GPU.
# Usage: bash scripts/data_processing/run_format_variant_test.sh [GPU_ID]
#   GPU_ID defaults to 0

set -e
set -x

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

source /workspace-vast/jinghanj/miniconda3/bin/activate
conda activate verl2

export HF_HOME=/workspace-vast/pretrained_ckpts

cd /workspace-vast/jinghanj/workspace/Structural_RL_dev

# Step 1 — create converted datasets for all variants (CPU only, fast)
for variant in v1_first_tok v2_strong v3_numbered v4_minimal; do
    echo "=== Creating dataset: $variant ==="
    python scripts/data_processing/create_dapo_math_v2.py --variant $variant
done

# Step 2 — test format following rate across all variants (needs GPU)
echo "=== Testing format following rate ==="
python scripts/data_processing/test_format_variants.py \
    --n_samples 1000 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max_tokens 4096 \
    --temperature 0.0 \
    --output results/format_variant_test.json

echo "Done. Results: results/format_variant_test.json"

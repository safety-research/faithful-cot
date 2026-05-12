#!/bin/bash
#
# Submit code hacking intervention faithfulness evaluation jobs.
# Two-step pipeline per checkpoint:
#   Step 1: create_code_intervention_datasets.py  → JSON datasets
#   Step 2: eval_code_intervention_logit_shifts.py → faithfulness results
#
# Usage:
#   bash scripts/eval_faithfulness/submit_code_intervention_evals.sh [--dry-run]
#

set -e

WORK_DIR=/workspace-vast/jinghanj/workspace/Structural_RL_dev

test_parquet=${WORK_DIR}/data/cc-transformed-hacking-with-hints-v2/test.parquet

declare -A CHECKPOINTS
declare -A OUTPUT_DIRS

CHECKPOINTS[vanilla]=${WORK_DIR}/results/checkpoints/gemma3-4b-hacking-v3-vanilla-bs512-len2048-seed1997
OUTPUT_DIRS[vanilla]=${WORK_DIR}/results/gemma3-4b-hacking-v3-vanilla-bs512-len2048-seed1997

CHECKPOINTS[cot_gradient]=${WORK_DIR}/results/checkpoints/gemma3-4b-hacking-v3-cot_gradient-bs512-len2048-seed1997
OUTPUT_DIRS[cot_gradient]=${WORK_DIR}/results/gemma3-4b-hacking-v3-cot_gradient-bs512-len2048-seed1997

CHECKPOINTS[gradient_mask]=${WORK_DIR}/results/checkpoints/gemma3-4b-hacking-v3-gradient_mask-bs512-len2048-seed1997
OUTPUT_DIRS[gradient_mask]=${WORK_DIR}/results/gemma3-4b-hacking-v3-gradient_mask-bs512-len2048-seed1997

CHECKPOINTS[update_mask]=${WORK_DIR}/results/checkpoints/gemma3-4b-hacking-v3-update_mask-bs512-len2048-seed1997
OUTPUT_DIRS[update_mask]=${WORK_DIR}/results/gemma3-4b-hacking-v3-update_mask-bs512-len2048-seed1997

DRY_RUN=0
if [ "${1}" = "--dry-run" ]; then
    DRY_RUN=1
    echo "[DRY RUN] No jobs will be submitted."
fi

echo ""
echo "========================================"
echo "Submitting code intervention eval jobs"
echo "  Methods: vanilla cot_gradient gradient_mask update_mask"
echo "========================================"
echo ""

for method in vanilla cot_gradient gradient_mask update_mask; do
    finetuned_model_dir="${CHECKPOINTS[$method]}"
    out_dir="${OUTPUT_DIRS[$method]}"

    if [ ! -d "$finetuned_model_dir" ]; then
        echo "⚠️  Method '$method': checkpoint dir not found, skipping..."
        continue
    fi

    if [ ! -d "$out_dir" ]; then
        echo "⚠️  Method '$method': output dir not found, skipping..."
        continue
    fi

    echo "Processing method: $method"

    for step in {10..500..10}; do
        finetuned_model=${finetuned_model_dir}/global_step_${step}_hf
        hacking_file=${out_dir}/checkpoint${step}/hacking_analysis.json
        datasets_file=${out_dir}/checkpoint${step}/code_intervention_datasets.json
        results_file=${out_dir}/checkpoint${step}/intervention_logit_shifts.json

        if [ ! -d "$finetuned_model" ]; then
            echo "  step $step — no checkpoint, stopping."
            break
        fi

        if [ ! -f "$hacking_file" ]; then
            echo "  step $step — hacking_analysis.json not found, skipping."
            continue
        fi

        if [ -f "$results_file" ]; then
            echo "  step $step — ✓ already done, skipping."
            continue
        fi

        echo "  step $step — submitting..."

        if [ $DRY_RUN -eq 0 ]; then
            sbatch ${WORK_DIR}/scripts/eval_faithfulness/run_code_intervention_eval_single.sh \
                "$test_parquet" \
                "$hacking_file" \
                "$finetuned_model" \
                "${out_dir}/checkpoint${step}" \
                "$datasets_file" \
                "$results_file"
        else
            echo "    [DRY RUN] -> $results_file"
        fi
    done
done

echo ""
echo "========================================"
echo "Done."
echo "Monitor: squeue -u \$(whoami)"
echo "========================================"

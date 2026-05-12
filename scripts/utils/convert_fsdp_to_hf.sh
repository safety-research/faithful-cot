#!/bin/bash

# Convert all FSDP checkpoints to HuggingFace format.
#
# Behavior after conversion:
#   - All checkpoints except the LAST: original .pt shard files are deleted
#   - LAST checkpoint (highest step): original .pt shard files are kept
#
# Usage: ./convert_fsdp_to_hf.sh [checkpoint_dir]
#
# Example:
#   ./convert_fsdp_to_hf.sh /path/to/checkpoints

CHECKPOINT_DIR="${1:-/workspace-vast/jinghanj/workspace/Structural_RL/results/checkpoints/qwen2.5-1.5b-instruct-new-hints}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Directory $CHECKPOINT_DIR does not exist"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_DIR="$PROJECT_ROOT/train"

if [ ! -d "$TRAIN_DIR" ]; then
    echo "Error: train directory not found at $TRAIN_DIR"
    exit 1
fi

BASE_DIR=$(dirname "$CHECKPOINT_DIR")
CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")

echo "Converting FSDP checkpoints in: $CHECKPOINT_DIR"
echo "================================================"

# Collect all valid step dirs and their step numbers
declare -a STEP_DIRS
declare -a STEP_NUMBERS

for step_dir in "$CHECKPOINT_DIR"/global_step_*; do
    [ ! -d "$step_dir" ] && continue
    actor_dir="$step_dir/actor"
    [ ! -d "$actor_dir" ] && continue
    # Must have FSDP .pt shard files (skip if already cleaned up)
    ls "$actor_dir"/model_world_size_*_rank_*.pt 1>/dev/null 2>&1 || continue

    step_name=$(basename "$step_dir")
    if [[ $step_name =~ global_step_([0-9]+) ]]; then
        STEP_DIRS+=("$step_dir")
        STEP_NUMBERS+=("${BASH_REMATCH[1]}")
    fi
done

if [ ${#STEP_DIRS[@]} -eq 0 ]; then
    echo "No FSDP checkpoints found (all may already be converted)."
    exit 0
fi

# Find the index of the last (highest step number) checkpoint
LAST_IDX=0
LAST_STEP=${STEP_NUMBERS[0]}
for i in "${!STEP_NUMBERS[@]}"; do
    if [ "${STEP_NUMBERS[$i]}" -gt "$LAST_STEP" ]; then
        LAST_STEP="${STEP_NUMBERS[$i]}"
        LAST_IDX=$i
    fi
done

echo "Found ${#STEP_DIRS[@]} checkpoints to process."
echo "Last checkpoint: global_step_${LAST_STEP} (original files will be kept)"
echo ""

N_CONVERTED=0
N_DELETED=0
N_FAILED=0

for i in "${!STEP_DIRS[@]}"; do
    step_dir="${STEP_DIRS[$i]}"
    step_number="${STEP_NUMBERS[$i]}"
    step_name=$(basename "$step_dir")
    actor_dir="$step_dir/actor"
    output_dir="$BASE_DIR/${CHECKPOINT_NAME}/${step_name}_hf"

    if [ "$i" -eq "$LAST_IDX" ]; then
        label="LAST — keeping originals"
    else
        label="will delete originals after conversion"
    fi

    echo "[$((i+1))/${#STEP_DIRS[@]}] $step_name  ($label)"
    echo "  Source: $actor_dir"
    echo "  Target: $output_dir"

    # Convert if not already done
    if [ -d "$output_dir" ] && [ -f "$output_dir/config.json" ]; then
        echo "  Already converted — skipping conversion"
    else
        cd "$TRAIN_DIR" && python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$actor_dir" \
            --target_dir "$output_dir"
        CONVERSION_STATUS=$?
        cd - > /dev/null

        if [ $CONVERSION_STATUS -ne 0 ]; then
            echo "  ✗ Conversion FAILED — skipping deletion"
            ((N_FAILED++))
            echo ""
            continue
        fi
        echo "  ✓ Conversion successful"
        ((N_CONVERTED++))
    fi

    # Delete original .pt shard files for all non-last checkpoints
    if [ "$i" -ne "$LAST_IDX" ]; then
        echo "  Deleting FSDP shard files..."
        rm -f "$actor_dir"/model_world_size_*_rank_*.pt
        rm -f "$actor_dir"/optim_world_size_*_rank_*.pt
        rm -f "$actor_dir"/extra_state_world_size_*_rank_*.pt
        echo "  ✓ Deleted shard files"
        ((N_DELETED++))
    fi

    echo ""
done

echo "================================================"
echo "Summary:"
echo "  Converted:      $N_CONVERTED"
echo "  Shards deleted: $N_DELETED"
echo "  Failed:         $N_FAILED"
echo "  Last step kept: global_step_${LAST_STEP}"
echo "================================================"

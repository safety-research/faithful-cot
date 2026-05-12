#!/bin/bash

# Script to convert all FSDP checkpoints in a directory to HuggingFace format
# Usage: ./convert_fsdp_to_hf.sh [checkpoint_dir]
# If no directory is provided, uses the default path

CHECKPOINT_DIR="${1:-/workspace-vast/jinghanj/workspace/Structural_RL/results/checkpoints/qwen2.5-1.5b-instruct-new-hints}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Directory $CHECKPOINT_DIR does not exist"
    exit 1
fi

echo "Converting FSDP checkpoints in: $CHECKPOINT_DIR"
echo "================================================"

# Get the base directory for output
BASE_DIR=$(dirname "$CHECKPOINT_DIR")
CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")

# Find all global_step_* directories
for step_dir in "$CHECKPOINT_DIR"/global_step_*; do
    if [ ! -d "$step_dir" ]; then
        continue
    fi
    
    # Check if actor subdirectory exists
    actor_dir="$step_dir"
    if [ ! -d "$actor_dir" ]; then
        echo "Skipping $step_dir: no actor/ subdirectory found"
        continue
    fi
    
    # Check if FSDP model files exist
    if ! ls "$actor_dir"/model_world_size_*_rank_*.pt 1> /dev/null 2>&1; then
        echo "Skipping $step_dir: no FSDP model files found in actor/"
        continue
    fi
    
    # Extract step number from directory name
    step_name=$(basename "$step_dir")
    
    # Create output directory name
    output_dir="$BASE_DIR/${CHECKPOINT_NAME}/${step_name}_hf"
    
    # Check if already converted
    if [ -d "$output_dir" ] && [ -f "$output_dir/config.json" ]; then
        echo "Skipping $step_name: already converted (output exists at $output_dir)"
        continue
    fi
    
    echo "Converting $step_name..."
    echo "  Source: $actor_dir"
    echo "  Target: $output_dir"
    
    # Run the conversion
    # Get the absolute path to the train directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    TRAIN_DIR="$PROJECT_ROOT/train"
    
    if [ ! -d "$TRAIN_DIR" ]; then
        echo "  ✗ Failed: train directory not found at $TRAIN_DIR"
        continue
    fi
    
    cd "$TRAIN_DIR" && python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$output_dir"
    CONVERSION_STATUS=$?
    cd - > /dev/null
    
    if [ $CONVERSION_STATUS -eq 0 ]; then
        echo "  ✓ Successfully converted $step_name"
    else
        echo "  ✗ Failed to convert $step_name"
    fi
    echo ""
done

echo "================================================"
echo "Conversion complete!"

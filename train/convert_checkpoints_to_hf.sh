#!/bin/bash

# Script to convert VERL FSDP checkpoints to HuggingFace format
# Usage: bash convert_checkpoints_to_hf.sh

# Base checkpoint directory
CHECKPOINT_BASE_DIR="/workspace-vast/jinghanj/workspace/Structural_RL/results/checkpoints/qwen2.5-1.5b-instruct-new-hints-new-dataset"

# Path to the converter script
CONVERTER_SCRIPT="/workspace-vast/jinghanj/workspace/Structural_RL/train/scripts/legacy_model_merger.py"

# Change to train directory to ensure imports work correctly
cd /workspace-vast/jinghanj/workspace/Structural_RL/train

echo "Starting checkpoint conversion to HuggingFace format..."
echo "Base directory: $CHECKPOINT_BASE_DIR"
echo ""

# Loop through all global_step_* directories
for CHECKPOINT_DIR in "$CHECKPOINT_BASE_DIR"/global_step_*; do
    # Skip if no directories match
    [ -d "$CHECKPOINT_DIR" ] || continue

    # Extract the step number from directory name
    STEP_NAME=$(basename "$CHECKPOINT_DIR")

    # Skip if _hf directory already exists
    HF_OUTPUT_DIR="${CHECKPOINT_DIR}_hf"
    if [ -d "$HF_OUTPUT_DIR" ]; then
        echo "⏭️  Skipping $STEP_NAME - HF model already exists at ${STEP_NAME}_hf"
        continue
    fi

    echo "================================================"
    echo "Converting: $STEP_NAME"
    echo "Input:  $CHECKPOINT_DIR/actor"
    echo "Output: $HF_OUTPUT_DIR"
    echo "================================================"

    # Run the converter script
    python "$CONVERTER_SCRIPT" merge \
        --backend fsdp \
        --local_dir "$CHECKPOINT_DIR/actor" \
        --target_dir "$HF_OUTPUT_DIR"

    # Check if conversion was successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully converted $STEP_NAME to HuggingFace format"
        echo ""
    else
        echo "❌ Failed to convert $STEP_NAME"
        echo ""
    fi
done

echo "================================================"
echo "Conversion complete!"
echo "================================================"

#!/bin/bash
# Setup udocker with custom storage location

set -e

echo "=================================================="
echo "Setting up udocker"
echo "=================================================="

# Storage location (can be overridden by UDOCKER_DIR env var)
STORAGE_DIR="${UDOCKER_DIR:-$HOME/.udocker}"
IMAGE="python:3.11-slim"

echo "Storage directory: $STORAGE_DIR"
echo "Image: $IMAGE"
echo ""

# Check if udocker is installed
if ! command -v udocker &> /dev/null; then
    echo "❌ udocker not found! Installing..."
    pip install udocker
fi

echo "✓ udocker is installed"
echo ""

# Initialize udocker in the custom directory
if [ ! -d "$STORAGE_DIR" ]; then
    echo "Initializing udocker in $STORAGE_DIR..."
    udocker install
    echo "✓ udocker initialized"
else
    echo "✓ udocker already initialized"
fi

echo ""

# Check if image is already pulled
if udocker images | grep -q "$IMAGE"; then
    echo "✓ Image '$IMAGE' is already available"
else
    echo "Pulling image '$IMAGE' (this may take a few minutes)..."
    udocker pull "$IMAGE"
    echo "✓ Image pulled successfully"
fi

echo ""
echo "=================================================="
echo "✓ udocker setup complete!"
echo "=================================================="
echo ""
echo "Storage location: $STORAGE_DIR"
echo "Available images:"
udocker images
echo ""

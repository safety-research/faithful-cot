#!/bin/bash
# Manual udocker setup with multiple fallback options

set -e

UDOCKER_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/.udocker"
export UDOCKER_DIR

echo "=========================================="
echo "Manual udocker Setup"
echo "=========================================="
echo "Storage: $UDOCKER_DIR"
echo ""

# Option 1: Try pulling from Docker Hub directly
echo "[Option 1] Trying to pull python:3.11-slim from Docker Hub..."
if udocker pull python:3.11-slim; then
    echo "✓ Success!"
    exit 0
fi

echo "❌ Failed. Trying alternatives..."
echo ""

# Option 2: Try smaller Alpine image
echo "[Option 2] Trying smaller python:3.11-alpine image..."
if udocker pull python:3.11-alpine; then
    echo "✓ Success! Using alpine variant"
    echo "Note: Update UDOCKER_IMAGE in your reward function to 'python:3.11-alpine'"
    exit 0
fi

echo "❌ Failed. Trying more alternatives..."
echo ""

# Option 3: Try using home directory temporarily
echo "[Option 3] Trying to pull to home directory first..."
TEMP_DIR=~/.udocker_temp
export UDOCKER_DIR=$TEMP_DIR
mkdir -p $TEMP_DIR

if udocker pull python:3.11-slim; then
    echo "✓ Downloaded to temp location"
    echo "Moving to target location..."

    # Export and import to new location
    udocker save python:3.11-slim > /tmp/python-3.11-slim.tar

    export UDOCKER_DIR="/workspace-vast/jinghanj/workspace/Structural_RL_dev/.udocker"
    udocker load < /tmp/python-3.11-slim.tar

    # Cleanup
    rm /tmp/python-3.11-slim.tar
    rm -rf $TEMP_DIR

    echo "✓ Success!"
    exit 0
fi

echo ""
echo "=========================================="
echo "❌ All automatic methods failed"
echo "=========================================="
echo ""
echo "Manual steps:"
echo ""
echo "1. Check network connectivity:"
echo "   ping -c 3 registry-1.docker.io"
echo ""
echo "2. Check disk space:"
echo "   df -h $UDOCKER_DIR"
echo ""
echo "3. Try downloading on another machine and transfer:"
echo "   # On machine with good network:"
echo "   udocker pull python:3.11-slim"
echo "   udocker save python:3.11-slim > python-3.11-slim.tar"
echo "   scp python-3.11-slim.tar user@target:/tmp/"
echo ""
echo "   # On this machine:"
echo "   export UDOCKER_DIR=/workspace-vast/jinghanj/workspace/Structural_RL_dev/.udocker"
echo "   udocker load < /tmp/python-3.11-slim.tar"
echo ""
echo "4. Use Docker instead (if available):"
echo "   docker pull python:3.11-slim"
echo "   docker save python:3.11-slim | udocker load"
echo ""

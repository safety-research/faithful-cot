#!/bin/bash
#
# Comprehensive Ray cleanup script
# Use this when Ray refuses to start due to stale processes/files
#
# Usage: bash scripts/cleanup_ray.sh
#

echo "========================================"
echo "Ray Cleanup Script"
echo "========================================"
echo ""

# 1. Force stop Ray (multiple attempts)
echo "1. Stopping Ray..."
for i in {1..3}; do
    echo "   Attempt $i/3"
    ray stop -f 2>/dev/null && echo "   ✓ Ray stopped" && break
    sleep 2
done
echo ""

# 2. Kill all Ray processes (no quotes on wildcards!)
echo "2. Killing Ray processes..."
pkill -9 -f "ray::" 2>/dev/null && echo "   ✓ Killed ray:: processes"
pkill -9 -f "raylet" 2>/dev/null && echo "   ✓ Killed raylet"
pkill -9 -f "gcs_server" 2>/dev/null && echo "   ✓ Killed gcs_server"
pkill -9 -f "dashboard" 2>/dev/null && echo "   ✓ Killed dashboard"
pkill -9 -f "ray_" 2>/dev/null && echo "   ✓ Killed other ray processes"
echo ""

# 3. Clean up temporary files (NO QUOTES around wildcards!)
echo "3. Cleaning temporary files..."

# /tmp/ray
if [ -d /tmp/ray ]; then
    rm -rf /tmp/ray 2>/dev/null && echo "   ✓ Removed /tmp/ray"
fi

# /tmp/ray_* (no quotes!)
rm -rf /tmp/ray_* 2>/dev/null && echo "   ✓ Removed /tmp/ray_*"

# /dev/shm/ray_* (no quotes!)
rm -rf /dev/shm/ray_* 2>/dev/null && echo "   ✓ Removed /dev/shm/ray_*"

# /tmp/session_* (sometimes Ray uses these)
rm -rf /tmp/session_* 2>/dev/null && echo "   ✓ Removed /tmp/session_*"

echo ""

# 4. Check for remaining Ray processes
echo "4. Checking for remaining Ray processes..."
ray_procs=$(ps aux | grep -i ray | grep -v grep | grep -v cleanup_ray.sh | wc -l)
if [ "$ray_procs" -eq 0 ]; then
    echo "   ✓ No Ray processes found"
else
    echo "   ⚠ WARNING: $ray_procs Ray processes still running:"
    ps aux | grep -i ray | grep -v grep | grep -v cleanup_ray.sh
    echo ""
    echo "   Try running: killall -9 python"
fi
echo ""

# 5. Check /dev/shm usage
echo "5. Checking /dev/shm usage..."
shm_usage=$(df -h /dev/shm | tail -1 | awk '{print $5}')
shm_available=$(df -h /dev/shm | tail -1 | awk '{print $4}')
echo "   /dev/shm usage: $shm_usage (available: $shm_available)"
if [[ "${shm_usage%?}" -gt 90 ]]; then
    echo "   ⚠ WARNING: /dev/shm is ${shm_usage} full!"
    echo "   Large files in /dev/shm:"
    du -h /dev/shm/* 2>/dev/null | sort -rh | head -10
fi
echo ""

# 6. Check for port conflicts
echo "6. Checking for Ray port conflicts..."
ray_ports="6379 8265 10001"
for port in $ray_ports; do
    if lsof -i :$port >/dev/null 2>&1; then
        echo "   ⚠ WARNING: Port $port is in use:"
        lsof -i :$port
    else
        echo "   ✓ Port $port is free"
    fi
done
echo ""

echo "========================================"
echo "Cleanup Complete!"
echo "========================================"
echo ""
echo "You can now try starting Ray again."
echo "If problems persist, try:"
echo "  1. Reboot the node: sudo reboot"
echo "  2. Or contact admin to clean /dev/shm"
echo ""

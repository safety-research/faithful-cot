#!/bin/bash
# Compact script to show GPU assignments per node

echo "GPU Assignments by Node"
echo "======================================================================"

# Get all running GPU jobs and process them
squeue -h -t RUNNING -o "%i %u %P %N %b" | while read jobid user partition nodelist gpus; do
    # Skip non-GPU jobs
    gpu_count=$(echo "$gpus" | grep -oP '\d+' | head -1)
    [ -z "$gpu_count" ] || [ "$gpu_count" == "0" ] && continue

    # Get GPU indices from scontrol
    gpu_indices=$(scontrol show job $jobid -d 2>/dev/null | grep -oP 'GRES=gpu[^)]*\(IDX:\K[^\)]+' || echo "?")

    # Expand nodelist if needed
    nodes=$(scontrol show hostnames "$nodelist" 2>/dev/null || echo "$nodelist")

    for node in $nodes; do
        printf "%-10s | Job: %-8s User: %-12s GPUs: %-2s IDs: %s\n" \
            "$node" "$jobid" "$user" "$gpu_count" "$gpu_indices"
    done
done | sort -k1,1 -k4,4n

echo "======================================================================"

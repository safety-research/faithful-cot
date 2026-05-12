#!/bin/bash
# Script to show GPU assignments per node using SLURM controller database
# This script queries scontrol with -d flag to get actual GPU device indices

echo "======================================================================"
echo "GPU Assignments by Node (from SLURM Controller)"
echo "======================================================================"
echo ""

# Associative arrays to store GPU info per node
declare -A node_gpus
declare -A node_jobs

# Get all running jobs with GPUs
while read jobid user partition nodelist gpus state; do
    # Skip if no GPUs
    gpu_count=$(echo "$gpus" | grep -oP '\d+' | head -1)
    if [ -z "$gpu_count" ] || [ "$gpu_count" == "0" ]; then
        continue
    fi

    # Get detailed job info with GPU indices
    job_detail=$(scontrol show job $jobid -d 2>/dev/null)

    # Extract GPU indices from the detailed output
    # Look for pattern like: GRES=gpu:X(IDX:Y,Z,...)
    gpu_indices=$(echo "$job_detail" | grep -oP 'GRES=gpu[^)]*\(IDX:\K[^\)]+' || echo "")

    # If no indices found, mark as "unknown"
    if [ -z "$gpu_indices" ]; then
        gpu_indices="?"
    fi

    # Handle multiple nodes (though typically GPU jobs run on single nodes)
    # Expand nodelist if needed
    nodes=$(scontrol show hostnames "$nodelist" 2>/dev/null || echo "$nodelist")

    for node in $nodes; do
        # Store job info for this node
        job_info=$(printf "%-10s %-12s %-10s %-8s %s" "$jobid" "$user" "$partition" "$gpu_count" "$gpu_indices")

        if [ -z "${node_jobs[$node]}" ]; then
            node_jobs[$node]="$job_info"
        else
            node_jobs[$node]="${node_jobs[$node]}"$'\n'"$job_info"
        fi
    done

done < <(squeue -h -t RUNNING -o "%i %u %P %N %b %T")

# Sort nodes and display
for node in $(printf '%s\n' "${!node_jobs[@]}" | sort); do
    # Get node GPU info
    node_info=$(scontrol show node $node 2>/dev/null)
    total_gpus=$(echo "$node_info" | grep -oP 'Gres=gpu:\K\d+' | head -1)
    alloc_gpus=$(echo "$node_info" | grep -oP 'AllocTRES=.*gres/gpu=\K\d+' | head -1)

    [ -z "$total_gpus" ] && total_gpus="?"
    [ -z "$alloc_gpus" ] && alloc_gpus="0"

    echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
    printf "║ %-40s GPUs: %s/%s in use      ║\n" "$node" "$alloc_gpus" "$total_gpus"
    echo "╠═══════════════════════════════════════════════════════════════════════════════╣"
    printf "║ %-10s %-12s %-10s %-8s %-20s ║\n" "JobID" "User" "Partition" "#GPUs" "GPU_Device_IDs"
    echo "╟───────────────────────────────────────────────────────────────────────────────╢"

    # Print jobs for this node
    echo "${node_jobs[$node]}" | while IFS= read -r job_line; do
        printf "║ %s ║\n" "$job_line"
    done

    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
done

# Summary
echo "======================================================================"
echo "Summary:"
echo "======================================================================"

# Count nodes with GPU usage
total_nodes_with_gpus=$(printf '%s\n' "${!node_jobs[@]}" | wc -l)
echo "Nodes with active GPU jobs: $total_nodes_with_gpus"

# Count total jobs
total_jobs=0
for node in "${!node_jobs[@]}"; do
    job_count=$(echo "${node_jobs[$node]}" | wc -l)
    total_jobs=$((total_jobs + job_count))
done
echo "Total GPU jobs running: $total_jobs"

echo ""
echo "Note: GPU Device IDs shown are the physical GPU indices (0-7 typically)"
echo "      These correspond to the GPU numbers shown in nvidia-smi"
echo "======================================================================"

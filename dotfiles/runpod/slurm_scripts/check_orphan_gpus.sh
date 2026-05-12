#!/bin/bash
# Script to find GPUs with memory usage > 100MiB that are not allocated in SLURM queue
# Uses SSH to query each node directly

MEMORY_THRESHOLD=100  # MiB

echo "======================================================================"
echo "Checking for GPUs with memory usage > ${MEMORY_THRESHOLD}MiB not in SLURM queue"
echo "======================================================================"
echo ""

# Get list of all nodes with GPUs
nodes=$(sinfo -N -h -o "%N" | sort -u)

for node in $nodes; do
    # Check if node has GPUs
    node_info=$(scontrol show node $node 2>/dev/null)
    total_gpus=$(echo "$node_info" | grep -oP 'Gres=gpu:\K\d+' | head -1)

    # Skip nodes without GPUs
    [ -z "$total_gpus" ] && continue

    # Get allocated GPU IDs from SLURM for this node
    allocated_gpus=()
    while read jobid; do
        [ -z "$jobid" ] && continue
        gpu_indices=$(scontrol show job $jobid -d 2>/dev/null | grep "Nodes=$node" | grep -oP 'GRES=gpu[^)]*\(IDX:\K[^\)]+')
        if [ -n "$gpu_indices" ]; then
            # Parse comma-separated or range format (e.g., "0-3" or "1,3,5")
            # Convert ranges to individual numbers
            for range in $(echo $gpu_indices | tr ',' ' '); do
                if [[ $range =~ ^([0-9]+)-([0-9]+)$ ]]; then
                    # It's a range like "0-3"
                    start=${BASH_REMATCH[1]}
                    end=${BASH_REMATCH[2]}
                    for ((i=start; i<=end; i++)); do
                        allocated_gpus+=($i)
                    done
                else
                    # It's a single number
                    allocated_gpus+=($range)
                fi
            done
        fi
    done < <(squeue -h -w $node -o "%i")

    # Query actual GPU memory usage on the node via SSH
    gpu_mem_info=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 $node \
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits" 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "[$node] WARNING: Could not connect via SSH or nvidia-smi failed"
        continue
    fi

    # Check each GPU
    orphan_found=false
    while IFS=',' read -r gpu_idx mem_used; do
        gpu_idx=$(echo $gpu_idx | xargs)
        mem_used=$(echo $mem_used | xargs)

        # Check if this GPU is allocated in SLURM
        is_allocated=false
        for alloc_gpu in "${allocated_gpus[@]}"; do
            if [ "$alloc_gpu" == "$gpu_idx" ]; then
                is_allocated=true
                break
            fi
        done

        # If not allocated but using significant memory, report it
        if [ "$is_allocated" = false ] && [ "$mem_used" -gt "$MEMORY_THRESHOLD" ]; then
            if [ "$orphan_found" = false ]; then
                echo "[$node] Found orphan GPU(s):"
                orphan_found=true
            fi

            # Try to get process info
            proc_info=$(ssh -o StrictHostKeyChecking=no $node \
                "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader --id=$gpu_idx" 2>/dev/null)

            printf "  GPU %s: %6s MiB used (NOT in SLURM queue)\n" "$gpu_idx" "$mem_used"

            if [ -n "$proc_info" ]; then
                echo "$proc_info" | while IFS=',' read -r pid proc_name proc_mem; do
                    pid=$(echo $pid | xargs)
                    proc_name=$(echo $proc_name | xargs)
                    proc_mem=$(echo $proc_mem | xargs)

                    # Try to find owner of the process
                    owner=$(ssh -o StrictHostKeyChecking=no $node "ps -o user= -p $pid 2>/dev/null" | xargs)
                    [ -z "$owner" ] && owner="unknown"

                    printf "    PID: %-8s User: %-12s Mem: %6s MiB  Cmd: %s\n" \
                        "$pid" "$owner" "$proc_mem" "$proc_name"
                done
            fi
        fi
    done <<< "$gpu_mem_info"

    if [ "$orphan_found" = true ]; then
        echo ""
    fi
done

echo "======================================================================"
echo "Check complete."
echo ""
echo "Note: 'Orphan GPUs' are GPUs using memory but not allocated in SLURM."
echo "      This may indicate:"
echo "      - Jobs that didn't clean up properly"
echo "      - Processes started outside of SLURM"
echo "      - Background processes left running"
echo "======================================================================"

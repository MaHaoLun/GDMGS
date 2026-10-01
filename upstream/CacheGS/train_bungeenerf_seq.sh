#!/bin/bash

# CacheGS BungeeNeRF Sequential Training Script
# Based on Octree-GS train_bungeenerf_seq.sh but adapted for YAML configs

exp_name="baseline"
gpu=-1

run_time=$(date "+%Y-%m-%d_%H-%M-%S")
log_root="/ssddata/lun/cachegs/outputs/logs/bungeenerf/${exp_name}/${run_time}"
mkdir -p "${log_root}"

echo "Starting CacheGS BungeeNeRF sequential training..."
echo "Experiment name: ${exp_name}"
echo "GPU: ${gpu}"
echo "Log directory: ${log_root}"
echo "Start time: ${run_time}"

# List of all bungeenerf scenes
scenes=("amsterdam" "barcelona" "bilbao" "chicago" "hollywood" "quebec" "rome" "pompidou")

# Train each scene sequentially
for scene in "${scenes[@]}"; do
    echo "=========================================="
    echo "Training scene: ${scene}"
    echo "Start time: $(date)"
    echo "=========================================="
    
    # Run training with YAML config
    python train.py --config config/bungee/lod_${scene}.yaml --gpu ${gpu} > "${log_root}/${scene}.log" 2>&1
    
    # Check if training completed successfully
    if [ $? -eq 0 ]; then
        echo "✓ ${scene} training completed successfully"
        echo "End time: $(date)"
    else
        echo "✗ ${scene} training failed"
        echo "Check log: ${log_root}/${scene}.log"
        # Continue with next scene even if current one failed
    fi
    
    echo ""
done

echo "=========================================="
echo "All scenes training completed!"
echo "Total log directory: ${log_root}"
echo "End time: $(date)"
echo "=========================================="

# Summary of all logs
echo ""
echo "Training Summary:"
echo "----------------"
for scene in "${scenes[@]}"; do
    log_file="${log_root}/${scene}.log"
    if [ -f "$log_file" ]; then
        echo "Scene ${scene}: ${log_file}"
        # Show last few lines of each log for quick status check
        echo "  Last lines:"
        tail -3 "$log_file" | sed 's/^/    /'
        echo ""
    else
        echo "Scene ${scene}: Log file not found"
    fi
done

#!/usr/bin/env bash
set -euo pipefail

step7b=/ssddata/lun/gdmgs_artifacts/proxygs_step7b_cross_pose_20260916
runner="$step7b/runtime/Proxy-GS-eac937e8/tools/run_step7b_scene.sh"
run_id=formal_step7b_unconditional_v1_20260916
scenes=(amsterdam barcelona bilbao chicago hollywood pompidou quebec rome)
views=(161 160 129 160 125 161 160 158)
gpus=(4 5 4 5 4 5 4 5)
ledger="$step7b/runs/formal/command_ledger.csv"

mkdir -p "$step7b/runs/formal"
printf 'scene,full_view_count,window_frame_count,gpu,run_id,state,log\n' > "$ledger"
pids=()
for index in "${!scenes[@]}"; do
  scene="${scenes[$index]}"
  expected="${views[$index]}"
  gpu="${gpus[$index]}"
  output="$step7b/runs/formal/$scene/$run_id"
  log="$step7b/runs/formal/${scene}.log"
  if [[ -f "$output/summary.json" ]] && [[ "$(jq -r .state "$output/summary.json")" == complete ]]; then
    printf '%s,%s,32,%s,%s,reused_complete,%s\n' \
      "$scene" "$expected" "$gpu" "$run_id" "$log" >> "$ledger"
    continue
  fi
  printf '%s,%s,32,%s,%s,launched,%s\n' \
    "$scene" "$expected" "$gpu" "$run_id" "$log" >> "$ledger"
  "$runner" "$scene" "$expected" "$gpu" > "$log" 2>&1 &
  pids+=("$!")
  # Alternate GPUs while keeping at most one active scene per GPU.
  if (( ${#pids[@]} == 2 )); then
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    pids=()
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

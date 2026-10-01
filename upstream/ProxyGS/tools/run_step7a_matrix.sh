#!/usr/bin/env bash
set -euo pipefail

step7a=/ssddata/lun/gdmgs_artifacts/proxygs_step7a_full_bundle_cache_20260916
runner="$step7a/runtime/Proxy-GS-eac937e8/tools/run_step7a_scene.sh"
run_id=formal_step7a_cache_core_v4_20260916
gpu="${1:-4}"
scenes=(amsterdam barcelona bilbao chicago hollywood pompidou quebec rome)
views=(161 160 129 160 125 161 160 158)
ledger="$step7a/runs/formal/command_ledger.csv"

mkdir -p "$step7a/runs/formal"
printf 'scene,full_view_count,window_frame_count,gpu,run_id,state,log\n' > "$ledger"
for index in "${!scenes[@]}"; do
  scene="${scenes[$index]}"
  expected="${views[$index]}"
  output="$step7a/runs/formal/$scene/$run_id"
  log="$step7a/runs/formal/${scene}.log"
  if [[ -f "$output/summary.json" ]] && [[ "$(jq -r .state "$output/summary.json")" == complete ]]; then
    printf '%s,%s,32,%s,%s,reused_complete,%s\n' \
      "$scene" "$expected" "$gpu" "$run_id" "$log" >> "$ledger"
    continue
  fi
  printf '%s,%s,32,%s,%s,launched,%s\n' \
    "$scene" "$expected" "$gpu" "$run_id" "$log" >> "$ledger"
  "$runner" "$scene" "$expected" "$gpu" > "$log" 2>&1
done

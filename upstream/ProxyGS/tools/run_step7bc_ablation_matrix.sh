#!/usr/bin/env bash
set -euo pipefail
root=/ssddata/lun/gdmgs_artifacts/proxygs_step7bc_temporal_20260916
runtime="$root/runtime/Proxy-GS-eac937e8"
run_id="${1:-ablation_v1}"
scenes=(amsterdam barcelona bilbao chicago hollywood pompidou quebec rome)
views=(161 160 129 160 125 161 160 158)
for lane in 0 1; do
  (
    for ((i=lane; i<8; i+=2)); do
      scene="${scenes[$i]}"
      if [[ -f "$root/runs/$scene/$run_id/summary.json" ]]; then continue; fi
      bash "$runtime/tools/run_step7bc_ablation_scene.sh" "$scene" "${views[$i]}" "$((4+lane))" "$run_id" > "$root/logs/${scene}_${run_id}.log" 2>&1
    done
  ) &
  pids[$lane]=$!
done
for pid in "${pids[@]}"; do wait "$pid"; done

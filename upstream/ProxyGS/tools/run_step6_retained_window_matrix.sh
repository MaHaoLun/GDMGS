#!/usr/bin/env bash
set -euo pipefail

diag=/ssddata/lun/gdmgs_artifacts/proxygs_step6_bvh_diagnosis_20260915
runner="$diag/runtime/Proxy-GS-eac937e8/tools/run_step6_retained_window_scene.sh"
scenes=(amsterdam barcelona bilbao chicago hollywood pompidou quebec rome)
views=(161 160 129 160 125 161 160 158)
tokens=(
  proxy_mesh_20260911:amsterdam:490999092:1789212176500569557
  proxy_mesh_20260911:barcelona:701003468:1789207614899027427
  proxy_mesh_20260911:bilbao:386986940:1789208693675434861
  proxy_mesh_20260911:chicago:356287956:1789209689992315892
  proxy_mesh_20260911:hollywood:528543476:1789211808848641393
  proxy_mesh_20260911:pompidou:544336076:1789240251869306554
  proxy_mesh_20260911:quebec:529247380:1789226351686149504
  proxy_mesh_20260911:rome:471337100:1789230896674517787
)

printf 'scene,full_view_count,window_frame_count,gpu,mesh_token,log\n' > "$diag/formal/full_chain_command_ledger.csv"
for index in "${!scenes[@]}"; do
  scene="${scenes[$index]}"
  log="$diag/formal/full_chain_${scene}.log"
  printf '%s,%s,32,5,%s,%s\n' \
    "$scene" "${views[$index]}" "${tokens[$index]}" "$log" \
    >> "$diag/formal/full_chain_command_ledger.csv"
  "$runner" "$scene" "${views[$index]}" 5 "${tokens[$index]}" > "$log" 2>&1
done

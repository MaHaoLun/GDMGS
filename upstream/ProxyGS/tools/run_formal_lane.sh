#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: run_formal_lane.sh GPU_UUID SCENE [SCENE ...]" >&2
  exit 2
fi

gpu_uuid="$1"
shift
root="/ssddata/lun/gdmgs_artifacts/proxygs_step3_gdmgs_backend_20260914"
step2="/ssddata/lun/gdmgs_artifacts/proxygs_step2_20260913"
runtime="$root/runtime/Proxy-GS-eac937e8"
python_bin="$root/env/proxy-gs-cu124/bin/python"
run_id="formal_g0_full_20260914"

declare -A expected=(
  [amsterdam]=161
  [barcelona]=160
  [bilbao]=129
  [chicago]=160
  [hollywood]=125
  [pompidou]=161
  [quebec]=160
  [rome]=158
)

cd "$runtime"
for scene in "$@"; do
  if [ -z "${expected[$scene]+x}" ]; then
    echo "unknown scene: $scene" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="$gpu_uuid" "$python_bin" render_gdmgs_backend.py \
    --scene "$scene" \
    --source-path "/ssddata/lun/data/bungeenerf/$scene" \
    --model-path "$step2/runs/bungee/$scene/formal_proxygs_native_40k_20260913" \
    --output-root "$root/runs/gdmgs_backend_full" \
    --run-id "$run_id" \
    --iteration 40000 \
    --expected-views "${expected[$scene]}" \
    --selection-mode all \
    --render-mode RGB \
    --warmup 3 \
    --repeat 3 \
    --native-diagnostic first \
    --lpips \
    --formal
done

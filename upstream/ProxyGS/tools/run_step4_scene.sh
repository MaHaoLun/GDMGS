#!/usr/bin/env bash
set -euo pipefail

scene="$1"
expected="$2"
gpu="$3"
mode="$4"
token="$5"

root=/ssddata/lun/gdmgs_artifacts/proxygs_step4_cpu_mesh_index_g1_v2_20260914
step2=/ssddata/lun/gdmgs_artifacts/proxygs_step2_20260913
step3=/ssddata/lun/gdmgs_artifacts/proxygs_step3_gdmgs_backend_20260914
runtime="$root/runtime/Proxy-GS-eac937e8"
py="$root/env/proxy-gs-cu124/bin/python"
export GDMGS_NATIVE_DIR="$root/native"
export TORCH_EXTENSIONS_DIR="$root/torch_extensions"
export CUDA_HOME=/usr/local/cuda-12.3
export PATH="$CUDA_HOME/bin:$root/env/proxy-gs-cu124/bin:/usr/local/bin:/usr/bin:/bin"
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="$gpu"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case "$mode" in
  qualification)
    output="$root/runs/qualification"
    run_id=formal_complete_scene_qualification_20260914
    extra=(--qualification --save-depth)
    ;;
  formal)
    output="$root/runs/formal"
    run_id=formal_step4_g1_v2_20260914
    extra=(--formal --no-save-depth)
    ;;
  *)
    echo "mode must be qualification or formal" >&2
    exit 2
    ;;
esac

cd "$runtime"
"$py" -u render_g1.py \
  --scene "$scene" \
  --source-path "/ssddata/lun/data/bungeenerf/$scene" \
  --model-path "$step2/runs/bungee/$scene/formal_proxygs_native_40k_20260913" \
  --mesh-input "/ssddata/lun/gdmgs_artifacts/proxy_mesh_20260911/proxy_mesh/bungee/$scene/cpu_mesh_index_input.npz" \
  --mesh-index "$root/indices/$scene/mesh_bvh.npz" \
  --mesh-token "$token" \
  --depth-oracle-root "$step2/depth/bungee" \
  --renderer-settings "$step3/manifests/renderer_settings.json" \
  --explicit-selection-contract "$step3/manifests/explicit_selection_contract.json" \
  --output-root "$output" \
  --run-id "$run_id" \
  --expected-views "$expected" \
  --warmup 1 --repeat 1 --lpips --save-images "${extra[@]}"


#!/usr/bin/env bash
set -euo pipefail

scene="$1"
expected="$2"
gpu="$3"
token="$4"

diag=/ssddata/lun/gdmgs_artifacts/proxygs_step6_bvh_diagnosis_20260915
step2=/ssddata/lun/gdmgs_artifacts/proxygs_step2_20260913
step3=/ssddata/lun/gdmgs_artifacts/proxygs_step3_gdmgs_backend_20260914
step4=/ssddata/lun/gdmgs_artifacts/proxygs_step4_cpu_mesh_index_g1_v2_20260914
step5=/ssddata/lun/gdmgs_artifacts/proxygs_step5_cpu_anchor_index_g2_20260915
step6=/ssddata/lun/gdmgs_artifacts/proxygs_step6_high_overlap_cpu_index_20260915
runtime="$diag/runtime/Proxy-GS-eac937e8"
py="$step2/env/proxy-gs-cu124/bin/python"
export GDMGS_NATIVE_DIR="$diag/native_fast"
export TORCH_EXTENSIONS_DIR="$step4/torch_extensions"
export CUDA_HOME=/usr/local/cuda-12.3
export PATH="$CUDA_HOME/bin:$step2/env/proxy-gs-cu124/bin:/usr/local/bin:/usr/bin:/bin"
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="$gpu"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH=.

cd "$runtime"
"$py" -u render_step6_windows.py \
  --scene "$scene" \
  --source-path "/ssddata/lun/data/bungeenerf/$scene" \
  --model-path "$step2/runs/bungee/$scene/formal_proxygs_native_40k_20260913" \
  --mesh-index "$step4/indices/$scene/mesh_bvh.npz" \
  --mesh-token "$token" \
  --anchor-index "$step5/indices/$scene/anchor_point_bvh.npz" \
  --step4-run "$step4/runs/formal/$scene/formal_step4_g1_v2_20260914" \
  --step5-run "$step5/runs/formal/$scene/formal_step5_g2_v1_20260915" \
  --selection "$step6/review/speed_selected_high_overlap_windows.json" \
  --mesh-window-profile "$diag/qualification/retained_profile_conservative.json" \
  --renderer-settings "$step3/manifests/renderer_settings.json" \
  --explicit-selection-contract "$step3/manifests/explicit_selection_contract.json" \
  --output-root "$diag/runs/windows" \
  --run-id formal_step6_retained_window_v2_20260915 \
  --expected-views "$expected" \
  --leaf-capacity 4096 --max-depth 32 \
  --mesh-threads 16 --index-repeat 3 \
  --warmup 1 --repeat 1 --lpips --no-save-images --formal

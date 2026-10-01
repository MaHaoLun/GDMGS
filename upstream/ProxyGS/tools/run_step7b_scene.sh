#!/usr/bin/env bash
set -euo pipefail

scene="$1"
expected_views="$2"
gpu="$3"

step7b=/ssddata/lun/gdmgs_artifacts/proxygs_step7b_cross_pose_20260916
step2=/ssddata/lun/gdmgs_artifacts/proxygs_step2_20260913
runtime="$step7b/runtime/Proxy-GS-eac937e8"
python="$step2/env/proxy-gs-cu124/bin/python"

export CUDA_VISIBLE_DEVICES="$gpu"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH=.

cd "$runtime"
"$python" -u render_step7b_fidelity.py \
  --scene "$scene" \
  --source-path "/ssddata/lun/data/bungeenerf/$scene" \
  --model-path "$step2/runs/bungee/$scene/formal_proxygs_native_40k_20260913" \
  --output-root "$step7b/runs/formal" \
  --run-id formal_step7b_unconditional_v1_20260916 \
  --expected-views "$expected_views" \
  --warmup 1 --repeat 1 --formal

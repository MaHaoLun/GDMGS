#!/usr/bin/env bash
set -euo pipefail
scene="$1"
views="$2"
gpu="$3"
run_id="${4:-formal_v1}"
root=/ssddata/lun/gdmgs_artifacts/proxygs_step7bc_temporal_20260916
step2=/ssddata/lun/gdmgs_artifacts/proxygs_step2_20260913
export CUDA_VISIBLE_DEVICES="$gpu"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=.
cd "$root/runtime/Proxy-GS-eac937e8"
"$step2/env/proxy-gs-cu124/bin/python" -u render_step7bc_temporal.py \
  --scene "$scene" --source-path "/ssddata/lun/data/bungeenerf/$scene" \
  --model-path "$step2/runs/bungee/$scene/formal_proxygs_native_40k_20260913" \
  --output-root "$root/runs" --run-id "$run_id" --expected-views "$views" \
  --repeat 3 --formal

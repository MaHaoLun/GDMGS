#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
: "${CUDA_HOME:?Set CUDA_HOME to a working CUDA toolkit before building.}"
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$project_root/requirements-runtime.txt"
python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
python -m pip install --no-build-isolation "$project_root/upstream/ProxyGS/submodules/diff-gaussian-rasterization"
python -m pip install --no-build-isolation "$project_root/upstream/ProxyGS/submodules/simple-knn"
python "$project_root/scripts/build_native.py" --cuda

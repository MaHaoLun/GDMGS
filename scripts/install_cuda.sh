#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
backend="${1:-cachegs}"
case "$backend" in cachegs|proxygs) ;; *) echo 'backend must be cachegs or proxygs' >&2; exit 2;; esac
: "${CUDA_HOME:?Set CUDA_HOME to a working CUDA toolkit before building.}"
if [[ "$backend" == cachegs ]]; then
  if ! python -c 'import importlib.util; assert importlib.util.find_spec("fvdb") is not None'; then
    echo 'CacheGS needs the compatible fVDB build described in dependencies/fvdb-observed.json.' >&2
    echo 'Build that dependency in this environment first; no unrelated fvdb package will be substituted.' >&2
    exit 2
  fi
fi
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$project_root/requirements-runtime.txt"
python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
if [[ "$backend" == cachegs ]]; then
  python -m pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
  python -m pip install PyYAML laspy scikit-learn
else
  python -m pip install --no-build-isolation "$project_root/upstream/ProxyGS/submodules/diff-gaussian-rasterization"
  python -m pip install --no-build-isolation "$project_root/upstream/ProxyGS/submodules/simple-knn"
fi
python "$project_root/scripts/build_native.py" --cuda

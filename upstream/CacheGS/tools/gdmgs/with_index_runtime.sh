#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set the intended visible CUDA device explicitly}"
gdmgs_source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
gdmgs_environment="${GDMGS_RENDER_ENV:-/ssddata/lun/miniconda3/envs/render}"
gdmgs_artifact_root="${GDMGS_INDEX_ARTIFACT_ROOT:-/ssddata/lun/gdmgs_artifacts/index_20260908}"
export CUDA_HOME="${GDMGS_CUDA_HOME:-/usr/local/cuda-12.3}"
export PATH="${gdmgs_environment}/bin:${CUDA_HOME}/bin:${gdmgs_artifact_root}/dependencies/bin:${PATH}"
export PYTHONPATH="${gdmgs_source_root}:${gdmgs_artifact_root}/dependencies${PYTHONPATH:+:${PYTHONPATH}}"
export GDMGS_NATIVE_DIR="${gdmgs_artifact_root}/native"
export TORCH_EXTENSIONS_DIR="${gdmgs_artifact_root}/torch_extensions"
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
cd "${gdmgs_source_root}"
exec "${gdmgs_environment}/bin/python" "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR="${REPO_ROOT}/exp"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/exp_run_${TIMESTAMP}.log"
FPS_LOG="${LOG_DIR}/fps_${TIMESTAMP}.csv"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[info] Log file: ${LOG_FILE}"
echo "[info] Starting experiment run at $(date)"
echo "[info] Repository root: ${REPO_ROOT}"

cd "${REPO_ROOT}"

unset PRECOMP_INDICES_PATH CACHE_ENABLE CACHE_IMPL CACHE_LOG_STATS CACHE_LOG_INTERVAL
unset CACHE_REQUIRE_JAGGED VIS_SUMMARY_SAMPLES

echo "label,fps" > "${FPS_LOG}"

RENDER_ENV="${RENDER_ENV:-render}"
PYTHON_CMD=(conda run -n "${RENDER_ENV}" --no-capture-output python)

py_render() {
    "${PYTHON_CMD[@]}" "$@"
}

RUN_DIR="/ssddata/lun/outputs/bungeenerf/rome/2025-11-11_10:59:10"
RENDER_ITER="${RENDER_ITER:-40000}"
PRECOMP_PATH="${RUN_DIR}/precomputed_indices.pt"
OUTPUT_ROOT="${RUN_DIR}/test/ours_${RENDER_ITER}"

RENDER_PY="${REPO_ROOT}/render.py"
PRECOMPUTE_PY="${REPO_ROOT}/precompute_cache.py"
METRICS_PY="${REPO_ROOT}/metrics.py"

snapshot_cache_stats() {
    local label="$1"
    local cache_log="${OUTPUT_ROOT}/cache_stats.jsonl"
    if [[ -f "${cache_log}" ]]; then
        local dest="${LOG_DIR}/cache_stats_${label}_${TIMESTAMP}.jsonl"
        cp "${cache_log}" "${dest}"
        echo "[cache] captured ${label} stats -> ${dest}"
    else
        echo "[warn] cache stats missing for ${label} at ${cache_log}"
        return 0
    fi
}

assert_no_cache_stats() {
    local label="$1"
    local cache_log="${OUTPUT_ROOT}/cache_stats.jsonl"
    if [[ -f "${cache_log}" ]]; then
        echo "[error] unexpected cache_stats.jsonl for ${label} (cache should be disabled)"
        exit 1
    fi
    echo "[cache] confirmed no cache stats for ${label}"
}

verify_precompute_payload() {
    local payload_path="$1"
    if [[ ! -f "${payload_path}" ]]; then
        echo "[error] expected precompute payload missing: ${payload_path}"
        exit 1
    fi
    PRECOMP_VERIFY_PATH="${payload_path}" "${PYTHON_CMD[@]}" - <<'PY'
import os
import torch

path = os.environ["PRECOMP_VERIFY_PATH"]
payload = torch.load(path, map_location="cpu")
version = int(payload.get("version", 1))
if version != 2:
    raise SystemExit(f"Precompute payload at {path} expected version=2, got {version}")
frames = payload.get("frames") or []
if not frames:
    raise SystemExit(f"Precompute payload at {path} has no frames")
sample = frames[min(len(frames) - 1, 0)]
required = {"indices", "ijk_jidx", "ijk_jdata"}
missing = required.difference(sample.keys())
if missing:
    raise SystemExit(f"Precompute payload at {path} missing jagged keys: {missing}")
print(f"[validate] {path} -> {len(frames)} frames with jagged metadata (version={version})")
PY
}

verify_per_view_metadata() {
    local label="$1"
    local expect_cache="$2"  # 1 or 0
    local expect_levels="$3" # 1 or 0
    local per_view="${OUTPUT_ROOT}/per_view_count.json"
    if [[ ! -f "${per_view}" ]]; then
        echo "[error] per_view_count.json missing for ${label}: ${per_view}"
        exit 1
    fi
    EXPECT_CACHE="${expect_cache}" EXPECT_LEVELS="${expect_levels}" PER_VIEW_PATH="${per_view}" "${PYTHON_CMD[@]}" - <<'PY'
import json
import os

path = os.environ["PER_VIEW_PATH"]
expect_cache = os.environ.get("EXPECT_CACHE", "0") == "1"
expect_levels = os.environ.get("EXPECT_LEVELS", "0") == "1"
with open(path) as fp:
    data = json.load(fp)
if not data:
    raise SystemExit(f"{path} is empty")
# Legacy payloads store visible counts as integers; normalize to dicts for checks.
normalized_entries = []
legacy = False
for entry in data.values():
    if isinstance(entry, dict):
        normalized_entries.append(entry)
    else:
        legacy = True
        normalized_entries.append({"visible_gaussians": int(entry)})

# If legacy schema detected we cannot require jagged metadata.
if legacy:
    expect_levels = False
    expect_cache = False

has_levels = any("levels" in entry for entry in normalized_entries)
has_cache = any("cache" in entry for entry in normalized_entries)
if expect_levels and not has_levels:
    raise SystemExit(f"{path} missing level histograms (legacy={legacy})")
if expect_cache and not has_cache:
    raise SystemExit(f"{path} missing cache metadata")
if not expect_cache and has_cache:
    raise SystemExit(f"{path} unexpectedly contains cache metadata while cache was disabled")
legacy_msg = " legacy_schema" if legacy else ""
print(f"[validate] per_view_count ({len(data)} views){legacy_msg} levels={has_levels} cache={has_cache}")
PY
}

prepare_render_output() {
    local output_name="$1"
    OUTPUT_ROOT="${RUN_DIR}/${output_name}/ours_${RENDER_ITER}"
    mkdir -p "${OUTPUT_ROOT}"
    rm -f "${OUTPUT_ROOT}/cache_stats.jsonl" "${OUTPUT_ROOT}/per_view_count.json"
}

extract_fps_from_log() {
    local label="$1"
    local start_line="$2"
    local start=$((start_line + 1))
    local fps_line
    fps_line=$(tail -n +"${start}" "${LOG_FILE}" | grep -E "Test FPS" | tail -n 1 || true)
    if [[ -z "${fps_line}" ]]; then
        echo "[warn] Unable to capture FPS for ${label}"
        return
    fi
    local fps
    fps=$(FPS_LINE="${fps_line}" python - <<'PY'
import os
import re
line = os.environ["FPS_LINE"]
match = re.search(r"Test FPS:[^\d]*(\d+(?:\.\d+)?)", line)
if not match:
    raise SystemExit(1)
print(match.group(1))
PY
) || {
        echo "[warn] Failed to parse FPS for ${label}"
        return
    }
    printf "[fps] %s: %s FPS\n" "${label}" "${fps}"
    printf "%s,%s\n" "${label}" "${fps}" >> "${FPS_LOG}"
}

echo
echo "[setup] Repo diagnostics"
git rev-parse HEAD
py_render -c 'import torch; print(torch.__version__)'
"${PYTHON_CMD[@]}" - <<'PY'
from gaussian_renderer.render import _PRECOMP_CACHE
_PRECOMP_CACHE["loaded"] = False
PY

echo
echo "[setup] Cache env defaults"
export CACHE_IMPL=optimized
export CACHE_REQUIRE_JAGGED=1
export VIS_SUMMARY_SAMPLES="${VIS_SUMMARY_SAMPLES:-64}"
echo "[setup] RUN_DIR=${RUN_DIR}"
echo "[setup] RENDER_ITER=${RENDER_ITER}"

echo
BASELINE_NAME="baseline"
PRECOMP_NAME="precompute"
CACHE_ONLY_NAME="cache_only"
FINAL_NAME="test"

echo
echo "[scaffold] Baseline render.py (no cache)"
prepare_render_output "${BASELINE_NAME}"
unset PRECOMP_INDICES_PATH
export CACHE_ENABLE=0
unset CACHE_LOG_STATS CACHE_LOG_INTERVAL
baseline_marker=$(wc -l < "${LOG_FILE}")
py_render "${RENDER_PY}" -m "${RUN_DIR}" --iteration "${RENDER_ITER}" --output_name "${BASELINE_NAME}"
verify_per_view_metadata "baseline" 0 1
assert_no_cache_stats "baseline"
extract_fps_from_log "baseline" "${baseline_marker}"

echo
echo "[scaffold] Precompute indices (jagged)"
py_render "${PRECOMPUTE_PY}" -m "${RUN_DIR}" --iteration "${RENDER_ITER}"
export PRECOMP_INDICES_PATH="${PRECOMP_PATH}"
verify_precompute_payload "${PRECOMP_INDICES_PATH}"
export CACHE_ENABLE=0
unset CACHE_LOG_STATS CACHE_LOG_INTERVAL
echo "[scaffold] Render with precomputed indices (no runtime cache)"
prepare_render_output "${PRECOMP_NAME}"
precompute_marker=$(wc -l < "${LOG_FILE}")
py_render "${RENDER_PY}" -m "${RUN_DIR}" --iteration "${RENDER_ITER}" --output_name "${PRECOMP_NAME}"
verify_per_view_metadata "precompute_only" 0 1
assert_no_cache_stats "precompute_only"
extract_fps_from_log "precompute_only" "${precompute_marker}"

echo
echo "[scaffold] Cache-only render (jagged ingestion)"
prepare_render_output "${CACHE_ONLY_NAME}"
unset PRECOMP_INDICES_PATH
export CACHE_ENABLE=1
export CACHE_LOG_STATS=0
export CACHE_LOG_INTERVAL=1
cache_only_marker=$(wc -l < "${LOG_FILE}")
py_render "${RENDER_PY}" -m "${RUN_DIR}" --iteration "${RENDER_ITER}" --enable_cache --output_name "${CACHE_ONLY_NAME}"
verify_per_view_metadata "cache_only" 1 1
snapshot_cache_stats "cache_only"
extract_fps_from_log "cache_only" "${cache_only_marker}"

echo
echo "[scaffold] Precompute + runtime cache render"
prepare_render_output "${FINAL_NAME}"
export PRECOMP_INDICES_PATH="${PRECOMP_PATH}"
export CACHE_ENABLE=1
export CACHE_LOG_STATS=0
export CACHE_LOG_INTERVAL=1
final_marker=$(wc -l < "${LOG_FILE}")
py_render "${RENDER_PY}" -m "${RUN_DIR}" --iteration "${RENDER_ITER}" --enable_cache --output_name "${FINAL_NAME}"
verify_per_view_metadata "precompute_plus_cache" 1 1
snapshot_cache_stats "precompute_plus_cache"
extract_fps_from_log "precompute_plus_cache" "${final_marker}"

echo
echo "[scaffold] Metrics with visibility summaries"
py_render "${METRICS_PY}" -m "${RUN_DIR}" --max_visibility_samples 16

echo
echo "[info] Experiments completed at $(date)"
echo "[info] All artifacts live under ${OUTPUT_ROOT} (cache logs) and ${LOG_DIR} (cache stats + FPS CSV ${FPS_LOG})"

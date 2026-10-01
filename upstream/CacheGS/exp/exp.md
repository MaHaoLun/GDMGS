# Phase 2/3 jagged-cache validation – Rome (ScaffoldGS)

Run directory: `/ssddata/lun/outputs/bungeenerf/rome/2025-11-11_10:59:10`

This page preserves the legacy cache validation runbook. See [the current A/B usage guide](../docs/gdmgs_ab_usage.md) for isolated outputs and the new explicit-ID rendering path. Everything below assumes the trained ScaffoldGS checkpoint above (iteration `40000`) and the unified cache backend (`cache/rendering_cache_optimized.py`).

## Performance parity notes

- The 45 FPS regression observed immediately after the fVDB migration came from `JaggedVisibilityDescriptor.from_indices` rebuilding full-grid boolean masks for every frame, forcing `grid.ijk.r_masked_select` to walk ~1.3 M voxels even when only ~100 k anchors were visible. The helper now slices `grid.ijk` directly via the requested indices and rehydrates a jagged tensor in-place, restoring the ~90 FPS baseline without touching training.
- `sh/run_exp.sh` captures the reported `Test FPS` for each sweep (baseline, precompute-only, cache-only, cache+precompute) and writes them to `exp/fps_<timestamp>.csv`. Use this CSV to compare fVDB-enabled renders against the Phase 0 reference and flag any drift in PRs.

## Common setup

```bash
cd /home/zyl/lun/CacheGS
git rev-parse HEAD
conda run -n render --no-capture-output python -c "import torch; print(torch.__version__)"
conda run -n render --no-capture-output python - <<'PY'
from gaussian_renderer.render import _PRECOMP_CACHE
_PRECOMP_CACHE["loaded"] = False  # cold-start cache shim
PY
unset PRECOMP_INDICES_PATH CACHE_ENABLE CACHE_IMPL CACHE_LOG_STATS CACHE_LOG_INTERVAL
export CACHE_IMPL=optimized
export CACHE_REQUIRE_JAGGED=1
export VIS_SUMMARY_SAMPLES=64

# Or run the scripted sweep (logs FPS deltas to exp/fps_<timestamp>.csv):
#   RENDER_ENV=render sh/run_exp.sh
```

## Rendering sweeps

All commands render the same Rome camera sweep in deterministic order. `cache_stats.jsonl` and `per_view_count.json` are emitted under `test/ours_40000/` for cache-enabled runs.

### 1. Baseline render.py (no cache)

```bash
export CACHE_ENABLE=0
conda run -n render --no-capture-output python render.py \
  -m /ssddata/lun/outputs/bungeenerf/rome/2025-11-11_10:59:10
```

Notes:

- Expect `cache_stats.jsonl` to be absent (cache disabled).
- The `exp/fps_<timestamp>.csv` row labeled `baseline` should stay at ~90 FPS on the Rome scene with a V100/3090-class GPU. Investigate immediately if it drifts by >5%.

### 2. Precompute-only (Jagged visibility descriptor)

```bash
conda run -n render --no-capture-output python precompute_cache.py \
  -m /ssddata/lun/outputs/bungeenerf/rome/2025-11-11_10:59:10
export PRECOMP_INDICES_PATH=/ssddata/lun/outputs/bungeenerf/rome/2025-11-11_10:59:10/precomputed_indices.pt
export CACHE_ENABLE=0
conda run -n render --no-capture-output python render.py \
  -m /ssddata/lun/outputs/bungeenerf/rome/2025-11-11_10:59:10
```

Validation:

- `torch.load($PRECOMP_INDICES_PATH)["version"]` must be `2`, and each frame entry should expose `{"indices", "ijk_jidx", "ijk_jdata"}` from `JaggedVisibilityDescriptor`.
- Rendering must complete without calling `torch.isin` (visible in CUDA trace).
- The `precompute_only` row in the FPS CSV should match the baseline within measurement noise because the decoder now hydrates jagged descriptors by slicing `grid.ijk` instead of re-scanning masks.

### 3. Runtime cache only (Jagged ingestion path)

```bash
unset PRECOMP_INDICES_PATH
export CACHE_ENABLE=1
export CACHE_LOG_STATS=0
export CACHE_LOG_INTERVAL=1
conda run -n render --no-capture-output python render.py \
  -m /ssddata/lun/outputs/bungeenerf/rome/2025-11-11_10:59:10 \
  --enable_cache
```

Checks:

- `cache_stats.jsonl` contains per-level `reuse`, `hits`, and `misses` extracted from `RenderingCache.get_cache_statistics`.
- `per_view_count.json` now includes `levels`, `sample_voxels`, and `cache` sections mirroring the jagged descriptor summaries.
- `cache_only` in the FPS CSV should land within ~5% of the baseline; larger drops indicate a cache invalidation bug.

### 4. Precompute + runtime cache

```bash
export PRECOMP_INDICES_PATH=/ssddata/lun/outputs/bungeenerf/rome/2025-11-11_10:59:10/precomputed_indices.pt
export CACHE_ENABLE=1
export CACHE_LOG_STATS=0
export CACHE_LOG_INTERVAL=1
conda run -n render --no-capture-output python render.py \
  -m /ssddata/lun/outputs/bungeenerf/rome/2025-11-11_10:59:10 \
  --enable_cache
```

Expectations:

- Cache hit-rate approaches the precompute-only visible counts (see `cache_stats.jsonl`).
- `tensor.is_contiguous()` assertions inside the cache remain true; no fallback to flat indices occurs because `CACHE_REQUIRE_JAGGED=1`.
- The `precompute_plus_cache` row in the FPS CSV is the number we publish in PRs (expect parity with baseline, ± a few percent).

## Diagnostics & metrics

- `cache_stats.jsonl` is newline-delimited JSON placed under `test/ours_40000/`. Plot it with the Phase 3 notebooks to inspect level saturation (`per_level_summary`).
- `per_view_count.json` captures frame time, jagged level histograms, and sampled `(level, ijk)` tuples for debugging.
- `conda run -n render --no-capture-output python metrics.py -m /ssddata/lun/outputs/bungeenerf/rome/2025-11-11_10:59:10 --max_visibility_samples 16` replays the enriched metadata and reports PSNR/SSIM/LPIPS plus cache tallies. Match these against the values already stored in `results.json`.
- `exp/fps_<timestamp>.csv` gives a one-line summary of all FPS readings gathered during the sweep; attach it (or paste the table) when filing perf-related PRs.

## Legacy asset upgrade (optional)

To reuse older flat-index dumps with the new cache:

```bash
conda run -n render --no-capture-output python precompute_cache.py \
  -m /ssddata/lun/outputs/bungeenerf/rome/2025-11-11_10:59:10 \
  --upgrade_from /ssddata/lun/outputs/bungeenerf/rome/2025-10-16_15:14:15/precomputed_indices.pt \
  --upgrade_out /ssddata/lun/outputs/bungeenerf/rome/2025-10-16_15:14:15/precomputed_indices_v2.pt
```

The upgrader rebuilds the jagged metadata through `JaggedVisibilityDescriptor` so runtime cache ingestion stays zero-copy, satisfying the Phase 2 exit criteria.

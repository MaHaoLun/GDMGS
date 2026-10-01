# Repository Guidelines

## Project Structure & Module Organization
`train.py` remains the training entrypoint; evaluation uses `render.py` and `metrics.py`. Method configs live under `config/{2dgs,3dgs,scaffoldgs}/` and ship with `base_model.yaml` / `lod_model.yaml` templates—copy before editing. Rendering kernels, cache plumbing, and precompute helpers sit under `gaussian_renderer/` plus the unified `cache/rendering_cache_optimized.py` (now the only cache backend). Shared camera/loss/visualization utilities are in `utils/`, and dataset abstractions are grouped in `scene/`. Assets, logs, and ad-hoc scripts live in `assets/`, `outputs.log`, and `sh/`.

> **fVDB reference:** when working on the fVDB migration, you can inspect `/home/zyl/XCube` for canonical `GridBatch`/`JaggedTensor` usage patterns (multiple levels of sparse grids, jagged zero-padding, chunked updates).

## Build, Test, and Development Commands
```bash
conda env create --file environment.yml && conda activate octree_gs
python train.py --config config/3dgs/lod_model.yaml
python precompute_cache.py -m /ssddata/.../trained_run
PRECOMP_INDICES_PATH=/ssddata/.../trained_run/precomputed_indices.pt CACHE_ENABLE=1 \
  python render.py -m /ssddata/.../trained_run --enable_cache
python metrics.py -m /ssddata/.../trained_run
```
Create the Python 3.8 Torch environment first, then launch training with a method-specific YAML. Regenerate indices before benchmarking large scenes, run cached renders only when `CACHE_ENABLE=1`, and finish by collecting PSNR/SSIM metrics.

When running commands from tooling scripts or notebooks, prefer the dedicated render environment instead of manually activating shells:
```bash
conda run -n render --no-capture-output python render.py -m /ssddata/.../trained_run --enable_cache
```
Replace the python invocation above with whichever command you need; `conda run -n render <cmd>` guarantees the fvdb build inside that env is used even from non-interactive jobs.

## Cache & Precompute Flags
Caching is opt-in: `CACHE_ENABLE` defaults to `0`, so pass `--enable_cache` *and* export `CACHE_ENABLE=1` to exercise reuse. `PRECOMP_INDICES_PATH` points to the `.pt` emitted by `precompute_cache.py`; unset it for live culling. Optional knobs (`CACHE_DEPTH_*`, `CACHE_LOG_*`) feed straight into `RenderingCache` and follow integer semantics—prefer documenting overrides in PRs.

## Legacy Asset Conversion
Legacy octree checkpoints, precompute dumps, and logs are no longer supported. Re-export scenes with the current fvdb-native training pipeline (fresh `train.py` run followed by `precompute_cache.py`) instead of upgrading artefacts in-place. Any attempt to load Phase 1/2 assets will now raise a hard error so stale payloads cannot flow into rendering or cache jobs.

## Coding Style & Naming Conventions
Follow PEP 8: 4-space indentation, `snake_case` for functions/modules, `CamelCase` for classes. Keep configuration keys descriptive and lower-case with underscores, mirroring `lod_model.yaml`. Use explicit device placement (`tensor.to(device)`), prefer pathlib/f-strings, and keep cache-specific constants centralized in `rendering_cache_optimized.py`. Document non-obvious math with short comments; avoid duplicating logic already in `utils/`.

## Testing Guidelines
There is no unit-test harness; rely on deterministic evaluation. Always re-run `python metrics.py -m <run_dir>` and compare PSNR/SSIM against the baseline reported in the PR. When modifying render/cache code, verify three cases: baseline (`CACHE_ENABLE=0`), precompute-only, and combined cache+precompute, noting FPS deltas in your PR. For loader changes under `scene/`, validate at least one 2D-GS and one 3D-GS config and record datasets plus commands.

## Commit & Pull Request Guidelines
Commits use single-line, imperative summaries (e.g., “Optimize GPU usage in precompute_cache”). Keep commits scoped and mention affected modules. PRs should include: problem statement, config diffs, hardware specs, commit hash for FPS measurements, and attached metric tables or TensorBoard screenshots. Link tracking issues when available and flag any required data assets outside the repo.

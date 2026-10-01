# Additive GPU-anchor downstream runner

`render_step8_gpu_schedule.py` preserves the old runner and accepts its existing data/model/profile/cache arguments. `tools/run_gpu_downstream_matrix.py` assembles the frozen scene inputs. The seven mode names are its default matrix:

- `cpu_serial_fresh_mesh32`
- `cpu_serial_cache_mesh32`
- `cpu_scheduled_cache_q2_mesh32`
- `gpu_serial_fresh_mesh32`
- `gpu_serial_cache_mesh32`
- `gpu_scheduled_cache_q1_mesh32`
- `gpu_scheduled_cache_q2_mesh32`

Every mode performs the same online 32-frame Mesh discovery batch. CPU modes use the original CPU tree; GPU modes use the fused GPUAnchorIndex with `OnlineProxyDepthRasterizer.render(copy_to_cpu=False)`. Candidate and selected IDs remain CUDA tensors through direct fresh decode or `TemporalBundleCache.resolve`. `GPUPairDemandLatch` validates metadata without copying/cloning IDs. The producer query completes before publication.

A shared reentrant lock protects selection, rendering, model state and the raster context; all GPU work uses the default stream. CPU tree work executes outside that GPU lock. GPU q1/q2 are measured host lookahead policies with one/two-pair submission lead. The CPU q2 comparator preserves the historical actual three-pair lead and reports that distinction. They are not evidence of simultaneous GPU execution. Late pairs retain the old complete-pair fresh-decode fallback. Cache age, capacity and pair refresh rules are unchanged.

Candidate/selected correctness oracles and camera matrices are preloaded before timing. Each run retains generated ID tensors until post-wall equality validation; hashes derive from the matching CPU oracle after equality succeeds. The wall timer excludes this equality check and reports its duration separately. All retained audit ID bytes and resident oracle bytes are reported separately from cache occupancy, alongside CUDA allocated memory before the wall interval and peak allocation. These correctness buffers are not bounded by q1/q2. Mesh equality and the CPU index's native range-expansion self-check remain inside measured wall time. GPU index query synchronization, dynamic compaction, cache scalar checks and synchronization remain in timing.

Qualification collects RGB, expected depth and alpha for all frames; its wall time is explicitly diagnostic-only. Performance runs disable those diagnostics, alternate mode order over three repetitions, and report complete-window wall time. CPU fresh is the required quality reference. GPU fresh must additionally match all 32 CPU fresh RGB, expected-depth and alpha tensors exactly. Progressive model decode restores current-view transition state under the GPU lock and includes this restoration in render time. Renderer and explicit-selection settings are checked and bound by identity. The independent reviewer recomputes quality budgets and rejects missing modes, scenes, frames or repeats.

Local validation: Python compilation and four CPU-only scheduler control tests pass. Those tests exercise seven complete 32-frame mode paths, q2's two-frame development edge case, actual GPU/CPU submission lead bounds, and invalid-mode rejection. Actual CUDA execution and the full scene matrix require the remote experiment environment; local tests establish no numerical or performance result.

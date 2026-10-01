# Additive CPU Mesh pipeline experiment

The v1 seven-mode experiment remains intact. This v2 extension targets measured whole-window CPU Mesh startup cost by doing only CPU Mesh discovery ahead of the current GPU pair.

`render_gpu_cpu_mesh_pipeline.py` subclasses the retained v1 runtime. It reuses model and contract validation, CUDA candidate/depth/fused Anchor selection, pair cache resolution, fresh RGB/depth/alpha exactness, qualification budgets, three-repeat alternating performance order, and post-wall ID equality. It adds these modes:

- `gpu_cpu_mesh_pipeline_q1_pool2`: one future pair, two CPU workers.
- `gpu_cpu_mesh_pipeline_q2_pool4`: two future pairs, four CPU workers.
- `gpu_cpu_mesh_pipeline_window_retained`: all 32 CPU Mesh queries submitted online using the unchanged frozen profile (24/32 workers), then consume each pair as soon as its two queries finish. This mode requires the full frozen 32-frame window and bounds pending pairs to 16 (lead 15). It changes waiting order while preserving the original CPU batch concurrency.

Controls are `cpu_serial_fresh_mesh32`, `gpu_serial_fresh_mesh32`, and unchanged `gpu_serial_cache_mesh32`. The new six-mode matrix requires its own independent review; it cannot satisfy the v1 seven-mode gate.

Each Mesh query still uses the exact retained backend and native implementation, with one native thread. The q1/q2 fixed pools intentionally differ from the frozen full-window profile; window_retained preserves that profile exactly. Selection records report actual pool capacity and preserve the profile count separately. Submission maintains at most current plus q future pairs. Both current Mesh results must finish before the calling thread runs current-pair CUDA selection and cache rendering. There are no background GPU queries. Mesh waits do not trigger skipped frames, stale requests, or fallback decodes.

Timings include pool setup, initial submissions, first wait, GPU current-pair selection, cache rendering, every subsequent wait and final executor shutdown. Reports include startup-to-first-selection readiness, per-pair and summed Mesh waits, actual queue bounds, and Mesh/GPU-stage host-span overlap. Overlap uses unions so concurrent CPU workers cannot double-count elapsed time. Host-span overlap is not a CUDA kernel overlap claim. All 32 frames retain quality verification; GPU audit/oracle memory is reported as in v1.

Five CPU tests validate full order and pair bounds, CPU worker versus calling-thread GPU ownership, failure cleanup, overlap accounting, and the one-pair development edge case. Those tests and compilation pass locally. Numerical equality, quality and timing remain remote experiment obligations.

The retained-worker control was added after the five-mode Amsterdam qualification showed that reducing Mesh concurrency to 2/4 workers increased exposed waiting. Prior five-mode artifacts remain immutable. This is the final controlled worker-policy ablation; the q1/q2 results remain negative evidence in the six-mode matrix.

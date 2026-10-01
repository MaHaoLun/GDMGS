# Method and actual implementation

Sources were selected using the supplied method TeX and the
[Notion project record](https://www.notion.so/3d8efdb220d0818d8259ef0bfe6a1d0e),
[sharing history](https://www.notion.so/3e3efdb220d081f3a10ae330ddb679a1), and
[scheduling history](https://www.notion.so/3ddefdb220d081a1a86ccd20332ee528).

| Method stage | Complete implementation |
| --- | --- |
| Original training/model/checkpoint | Default CacheGS/GDMGS_Codebase source and YAML loader; explicit historical ProxyGS compatibility |
| Anchor bounds and Morton radix index | `gdmgs/anchor_frustum/gpu_construction.py` and all native construction/query sources |
| CPU Cull/Keep/Descend | `system/cpu_select.cpp`, including active-hole propagation and contiguous subtree reporting |
| Solid classification and merge | `system/occupancy.py`, exposed by `scripts/prepare_occluders.py` |
| Occluder octree / near-plane subdivision | `holed_index.py`; device-resident `gpu_occluders.py` |
| CPU/GPU hole predicates | `hole_planes_native.cpp`, `gpu_occluders.py`, `hole_filter.cu` |
| Stable IDs and union/membership | Retained `cache_build_optim.py`, `PlannedArena`, `PriorityRenderer` demand bits |
| Source pose and split decode | Retained `epoch_cache.py`, `dense_math.py`, `fullblock_timed.py` |
| Per-target projection/sort/render | Retained `priority_renderer.py`, `parallel_renderer.py`, staged intersection C++/CUDA and Triton sanitization |
| CPU/GPU allocation and barrier | `planning.py`, `full_system.py`, explicit completion events and bounded complete batches |
| Exact-row admission | Retained `Admission`, before attribute materialization; no capacity-driven truncation |

## Corrections introduced during integration

1. Historical hole tests used candidate-center boxes while frustum tests used
   the stored scale bound. Both native CPU and CUDA hole predicates now expand
   the center envelope by `3 * max_scale` on each axis. A compiled regression
   test covers an anchor with hidden centers but support outside the hole.
2. Historical CPU/GPU `log2` rounding produced isolated LoD differences. Both
   processors now compare squared distance against the same precomputed double
   threshold. Round-to-even equality is explicit; GPU uses round-to-nearest
   operations without FMA. This changes the implementation at native rounding
   boundaries and requires full-scene GPU/image requalification.
3. Full occluder cells crossing the near plane are subdivided down to the
   declared grid level, instead of discarding the whole parent. CPU and device
   retrieval use the same sparse region-octree structure.
4. The full source decoder and fused renderer replace the previous draft's
   tensor reference renderer. No old absolute experiment source paths are needed
   by the integrated entrypoint. Inputs and build caches are explicit.

## Boundaries that must remain explicit

The runtime retains the original renderer-aware projected support rejection in
its anchor kernels. This is more involved than the TeX's pure six-plane AABB
presentation. GPU execution first queries the anchor index and then evaluates
holes for survivors; CPU execution additionally prunes whole hidden subtrees.
They implement the same per-anchor predicate, but do not have identical work.

The `3 * max_scale` correction covers the decoder's finite Gaussian support.
A blanket proof of lossless occlusion for gsplat's additional screen-space
covariance dilation has not been established here. Mesh opacity/interior
assumptions also remain necessary. Therefore neither the paper's absolute
soundness claim nor historical image-quality numbers should be attributed to
this integration without full renderer-level validation.

The archived optimized joint anchor/triangle query is a different selection
variant. It is not silently substituted for solid-cell occlusion. The current
root path uses the supplied method's solid cells; legacy depth/native paths
remain available in the complete upstream source tree.

The retained union puts the first target's sorted IDs before remaining IDs.
That is a physical layout choice; original anchor IDs and per-target membership
are retained explicitly. It differs from a globally sorted physical union in
the TeX, without changing which anchors belong to each target.

This task was restricted to local work by the user. CUDA compilation/execution,
original-checkpoint tensor parity and full target quality were not run for this
integration. No historical PASS is reused as its acceptance result.

## Model source correction

The default training/model implementation is now the remotely verified
CacheGS/GDMGS_Codebase snapshot. Eight January training backups differ from
its training entrypoint only in the output-root literal. The later experimental
model lineage is retained explicitly and must not be labeled as the training
foundation of GDM-GS. See `source_audit.json` for both checkpoint inventories.

The CacheGS decoder stores scale/rotation as FP16 then materializes them as FP32.
The new `model_bridge.py` preserves that rounding and complete row ownership.
It calls pose-local decoding instead of mutating shared fVDB state. CacheGS LoD
positions use the anchor itself; only the historical backend uses its half-voxel
shift. No checkpoint is silently translated into the other model format.

The user requested deferring query/schedule-variant adjudication. Retained
original implementations are therefore also stored under `research/retained/`
without selecting a new algorithm or importing their experiment launchers.

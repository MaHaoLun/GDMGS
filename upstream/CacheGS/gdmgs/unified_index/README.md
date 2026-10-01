# Anchor index contract

This module builds an independent CPU octree over every finalized PLY row. PLY
IDs are never changed. `dfs_to_row` and `rank_of_row` are inverse permutations;
every node owns a half-open DFS interval. Coincident coordinates remain distinct
rows. Partition cells describe topology, while separate support bounds carry an explicit definition. The default
all-view definition covers all possible decoded offsets and Gaussian support;
the experimental native anchor proxy below covers a different, narrower object.

## API

```python
from gdmgs.unified_index import AnchorIndex, SupportSettings
index = AnchorIndex.from_finalized(scene, leaf_capacity=64)
index.save("anchor_index.npz")
index = AnchorIndex.load("anchor_index.npz", finalized_scene=scene)
result = index.query(fov_ids, ori.depth_bounds, camera.w2c,
                     ori.angular_domain, ori.image_size,
                     mode="tree", camera_token=camera.camera_id,
                     mesh_token=ori.mesh_token, scene_token=scene.token)
```

`fov_ids` is a unique `int64` vector. ORI depths are a `float64 [H,W]` array:
positive finite values are an upper camera-z bound and `+inf` is Unknown.
Continuous ORI supplies full-cell triangle-union coverage; the separately named
native-pixel ORI supplies coverage of every native pixel center in each tile.
The experiment must record which contract produced the buffer; pixel-center
coverage is not presented as continuous coverage. W2C is a `float64 [4,4]` affine matrix
multiplying column vectors. Angular bounds are `(xmin,xmax,ymin,ymax)` in `x/z,
y/z`. For production PixelORI, use `ori.angular_domain` and `ori.image_size` together:
these retain calibrated pixel spacing while padding only the bottom/right.
For example, Truck's original height 891 becomes 896 in this query grid; mixing
its padded depth cells with the original image height would misalign them.
The caller binds ORI camera/mesh identity before handing over this raw
immutable depth buffer. The result preserves the input FoV order, including
nonmonotonic input IDs. `mode="linear"` certifies each FoV row individually and
serves as R1's anchor baseline.

`raw_ranges` and `formal_ranges` both denote exactly the selected set in DFS
rank order; formalization only merges adjacent ranges. Optional `RangeState`
reuses unchanged range storage *after* fresh discovery. It never supplies or
restricts candidate discovery and stores no decoded Gaussians.

## Optional native anchor proxy experiment

The default remains `gsplat-1.4-pinhole-classic-alpha-support-v3`. Development
comparisons may explicitly request:

```python
settings = SupportSettings(profile="native_anchor_proxy_v1")
proxy_index = AnchorIndex.from_finalized(scene, support_settings=settings)
```

`native_anchor_proxy_v1` means
`native_fov_anchor_first3_scale_image_space_proxy`. Its only center is the
original anchor, and its radius is `3.5 * max(activated_scale[:3])`, rounded
outward in FP32 with the same world/screen numerical slack. This matches the
position and scale sources in `gaussian_renderer/visibility.py`:
`pc.get_anchor`, `pc.get_scaling[:,:3]`, and the normalized `pc.get_rotation`.
The maximum-axis sphere bounds any such rotation. FoV passes those parameters
directly as FP32; the decoder's FP16 batch rounding does not apply here.
The original FoV call and its selected IDs remain unchanged.

This proxy **does not include decoded offsets or the last three decoder
scales, and does not bound the full decoded Gaussian bundle**. Its image-space
occlusion decision is an explicit approximation whose adoption requires every
final frame to pass the accepted PSNR-drop <=0.1 dB and SSIM-drop <=0.002 gate.
A synthetic fixture intentionally places an anchor behind a wall while one
of its decoded offsets is in front: the proxy removes that anchor and the
all-view definition retains it. This distinction must remain visible in
experimental reports; geometric tests for the proxy do not establish image
quality.

R1 and R2 must use the same profile, original FoV IDs and ORI. Their tree versus
linear removal/range contracts are unchanged. Each query returns
`support_profile` and `support_definition`. Format-4 artifacts store profile,
definition and rounding together, reject inconsistent combinations, and may
be loaded with `support_profile=...` to require an exact match. Changing the
support definition or world-bound settings of an existing index raises an
error; rebuilding is required. Earlier format-2/3 all-view artifacts retain
their original definition when loaded. No experiment outcome has established
this optional profile as the preferred production setting.

## Renderer support derivation (default all-view definition)

The inspected production decoder computes centers as
`anchor + offset * torch.exp(log_scale[:3])`. All offset slots participate in the
support, independent of camera-dependent opacity. Its covariance axes are
`torch.exp(log_scale[3:]) * sigmoid(...)`. `from_finalized` evaluates the actual
device `torch.exp` before copying activated values. The batch converts axes and
quaternions to FP16; gsplat then upcasts to FP32 and normalizes the quaternion.
We outward-round each maximum possible axis to the next FP16 value; an overflow
or any unknown/nonfinite offset or scale produces unbounded support and Keep.

The inspected gsplat version is `1.4.0+pt24cu124`. Its
`cuda/csrc/rasterize_to_pixels_fwd.cu` rejects alpha below `1/255`; opacity is at
most one, so surviving samples lie within `sqrt(2 log(255)) = 3.329...` projected
sigma. The default support uses 3.5 sigma plus floating-point slack. Its
`fully_fused_projection_fwd.cu` adds `eps2d=0.3`, computes the radius from
`ceil(3 * sqrt(b + sqrt(max(0.01, b*b-det)))))`, and `isect_tiles.cu` expands to
16-pixel tiles. The query keeps an additional pixel pad of
`3.5 * sqrt(0.3 + 0.1) + 1 = 3.213594...` pixels on each side in the v3
`gsplat-1.4-pinhole-classic-alpha-support-v3` profile. A tile only determines
where a Gaussian is evaluated; the same alpha cutoff still rejects samples
outside its projected ellipse. Adding a complete tile a second time was
unnecessary. Legacy v2 artifacts retain their recorded 19.213594-pixel profile
when loaded, so loading never silently changes an earlier experiment.

World support is the union of all offset centers expanded by the maximum
rotated-axis radius. Projection uses all eight corners and outward FP32
transform-error bounds; crossing or approaching the near plane produces Keep.
A perspective world box alone does **not** necessarily bound gsplat's
linearized covariance footprint. Therefore the tree also stores all-offset
center boxes and maximum radii. Native projection separately bounds the
clamped pinhole Jacobian row norm by `sqrt(1 + clamped_ratio^2) / minimum_z`,
including a spectral upper bound on the camera rotation. The final screen
support encloses both that footprint and the projected world box, plus the
pixel pad. The large off-axis regression exercises a real gap in the naive
world-box-only bound.

A node or anchor is removed only when every ORI cell intersecting this screen
support is finite and the largest occluder depth plus margin is strictly before
the nearest world support depth. Unknown support, cells and near-plane cases
remain candidates and have separate counters. Parent bounds include every
child center, radius and support; a tree certificate cannot remove an anchor
that the same per-anchor certificate would retain.

## Build, persistence and timing

Run `native/build.py --output-dir <native directory>` with pybind11 installed,
then set `GDMGS_NATIVE_DIR` to that directory. This independent CPU C++17 module
requires neither torch compilation nor CUDA. No hashes/checksums are used.

Saving records actual positions, support/center/radius buffers, nodes, DFS
permutations, format/settings and source path/iteration/size/mtime. Loading
restores that topology without repartition and validates its bijection,
reachability, interval partition and descendant-bound containment. A persisted
index can bind a new finalized session only after source metadata agrees.

`prepare_ms` includes FoV rank marking and the O(N) prefix count. Traversal uses
that prefix to skip empty nodes in O(1). Static support records are prevalidated
and stored contiguously in DFS order for leaf traversal; these are index
geometry metadata, with no decoded Gaussian payload. An Unknown cell at the
center-box midpoint rejects a certificate before its full projection. This
shortcut only retains candidates: a finite midpoint never permits deletion.
The remaining projection evaluates positive-denominator extrema by numerator
sign and retains outward error slack without per-corner libm calls.
`traversal_ms` includes node and leaf certification. `materialization_ms` includes range generation/formalization and
returning original-order IDs. `native_total_ms` and `python_total_ms` enclose
all their respective work. Preparing full-row flags and prefix counts is still
O(N); traversal pruning does not imply a sublinear complete query. Report the
full timings, not just certified-node reductions.

## Frozen Truck CPU optimization evidence

The full saved Truck frame 0 FoV contains 130,217 anchors; the pixel ORI has
541 finite cells out of 22,400. Every comparison uses those same saved input
arrays, one warmup per mode, then five retained alternating-order repetitions.
Only CPU anchor query time is measured here.

| Implementation | Linear median ms | Tree median ms | Removed anchors |
| --- | ---: | ---: | ---: |
| Original v2 | 125.033 | 153.292 | 0 |
| Negative-only Unknown shortcut, v2 support | 17.377 | 34.268 | 0 |
| Faster outward arithmetic, v2 support | 9.510 | 18.852 | 0 |
| v3 alpha-support padding | 8.928 | 17.364 | 3 |
| Contiguous DFS support metadata, final v3 | 9.875 | 9.406 | 3 |

All same-profile optimized IDs exactly match the original; the final linear
and tree IDs match each other. The v3 profile removes rows 6363, 70315 and
107373, while retaining the other 130,214 in their original FoV order. Ratios
of all five elapsed sums are 12.285x for old/new linear, 16.064x for old/new
tree, and 1.081x for final linear/tree. These are development component
measurements, not complete-query or fresh-render speedups. The sparse mesh
coverage and small number of removals remain material limitations for this
candidate.

The retained remote evidence is
`/ssddata/lun/gdmgs_artifacts/index_20260908/anchor_cpu_optimization/`:
`comparison.json`, every stage's timing/counter JSON, and explicit before/after
ID arrays. `native/benchmark_snapshot.py` reproduces the paired CPU benchmark.
The combined anchor, mesh and independent CPU suite passed 57 tests after the
optimization, including large off-axis support, FP16 rounding, full ranges and
known-occluder removal.

## GPU query implementation

GPUAnchorIndex.from_cpu(anchor_index, device) uploads the existing validated
binding, topology and support metadata once. It keeps no decoder output.
Build the independent extension with cuda/build.py --output-dir <directory>
and set GDMGS_GPU_NATIVE_DIR, or use $GDMGS_NATIVE_DIR/anchor_cuda.
The CPU extension remains independent and unchanged. CUDA compilation uses
normal precision with FMA contraction disabled; no fast-math option is used.

    from gdmgs.unified_index.gpu_index import GPUAnchorIndex
    index_gpu = GPUAnchorIndex.from_cpu(index, device="cuda:0")
    result = index_gpu.query(fov_ids_gpu, ori.gpu_depth_bounds,
                             camera.w2c, ori.angular_domain, ori.image_size,
                             mode="tree", camera_token=camera.camera_id,
                             mesh_token=ori.mesh_token)
    # Consume selected_anchor_ids directly in the fresh decoder.
    # After the enclosing frame synchronizes:
    counters = result.collect_counters()
    timings = result.timings

Each query builds an exact two-dimensional sparse max table over its actual
float32 or float64 ORI values. Horizontal and vertical dyadic levels require
log2(width) + log2(height) parallel launches. Any support rectangle is exactly
the union of four overlapping dyadic blocks inside that rectangle, so four
reads return its exact maximum; Unknown +inf propagates without snapping or
additional conservative coverage. The full table-build cost belongs to query
time. Its values are the same uploaded pixel-tile upper depths, not ray samples
or a redefined mesh-coverage proof.

GPU R1 certifies every original FoV row in parallel. GPU R2 first marks the
FoV's DFS ranks and computes prefix counts, evaluates all nonempty tree nodes
in parallel, retains maximal certified intervals through parent checks, and
uses a difference/prefix array to mark their rows. It then certifies every
remaining FoV anchor with the same certifier. This is explicitly a batch-node
GPU strategy, not early-exit DFS. visited_nodes, empty_nodes,
maximal_certified_nodes and anchor_checks report its actual work.

Selected IDs retain original FoV order. GPU raw and formal ranges are canonical
selected DFS runs, with exact identical denotation. Their construction,
including required dynamic-output/nonzero synchronization, occurs inside the
query. Only small counters and event-duration collection are deferred until
after the timed fresh frame; no missing range algorithm is performed then.
gpu_total_ms is event elapsed time and host_launch_and_sync_ms is host time
through submission and necessary synchronization. The benchmark's enclosing
synchronized wall measurement remains the complete query timing.

Eleven real GPU tests passed: all rectangles of a non-power-of-two sparse table
against an independent full scan; CPU/GPU ID equality with float32 and float64
ORI; both support definitions; rotated cameras, holes, near-plane, FP16 and
large off-axis fixtures; maximal intervals without duplicate removal; and
exact final ranges. On frozen Truck frame 0, five retained GPU measurements
produced median complete anchor-query times of 1.129 ms (linear) and 1.285 ms
(tree), with all 130,214 selected IDs exactly matching the CPU reference and
three removals. The node method is slightly slower on this sparse-coverage
fixture. Resident index metadata uses 34,157,409 bytes and the query's sparse
table uses 5,017,600 bytes. These are component development results, not a
complete-render acceleration claim.

Evidence lives in remote
/ssddata/lun/gdmgs_artifacts/index_20260908/anchor_gpu_development/:
gpu_comparison.json contains every warmup, timing repetition and counter;
gpu_ids.npz retains full FoV and output IDs. The reproducible entrypoint is
cuda/benchmark_snapshot.py.

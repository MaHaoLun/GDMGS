# Mesh retrieval and pixel depth index

`MeshIndex` owns an object-split BVH. Every triangle row occurs once in its leaf
references. `median` and `binned_sah` are genuine alternative builders. Node
bounds enclose complete triangles; an unrelated anchor partition never limits
the mesh domain. The linear scan and BVH use the same conservative six-plane
triangle predicate and return sorted unique `int64` face-row IDs. Corner false
positives are allowed; triangle centroids never decide relevance.

```python
from gdmgs.mesh_index import MeshIndex
from gdmgs.query.depth_pyramid import MeshDepthRasterizer

# vertices: numpy.float64 [V,3]; faces: numpy.int64 [F,3]
index = MeshIndex(vertices, faces, method="binned_sah", leaf_size=8,
                  mesh_token="run-label:scene:mesh-method")
index.save("mesh_bvh.npz")
index = MeshIndex.load("mesh_bvh.npz", mesh_token=index.mesh_token)
query = index.query(camera_domain, backend="bvh")
rasterizer = MeshDepthRasterizer(index, device="cuda:0", tile_size=8)
ori = rasterizer.build(query, camera_domain, image_size=(width, height))
# The anchor consumer uses ori.angular_domain and ori.image_size together.
```

The camera mapping contains finite `float64` `w2c[4,4]`, multiplying column
vectors; `angular_domain=(xmin,xmax,ymin,ymax)` describes camera `x/z,y/z`;
`near` is positive, `far` may be infinity, and `camera_id` is an explicit label.
For a centered pinhole camera the angular bounds are ±tan(FoV/2).

The adopted ORI definition is `native_pixel_depth_tiles_v1`. Nvdiffrast CUDA
rasterizes every original pixel center. Camera-z is perspective-correctly
interpolated; empty or invalid pixels become infinity. Tile reduction takes the
maximum, so every sample must be covered before a tile can be finite. Image
padding preserves the original calibration and adds unknown samples only on the
right/bottom. The returned angular domain and padded dimensions describe those
same tile boundaries. This is a discrete pixel visibility criterion, not a
continuous coverage proof; per-frame image-quality acceptance remains required.

Geometry resides on the selected GPU. Each query uploads its own triangle IDs,
projects/rasterizes the current camera, reduces depths, and downloads fresh
float64 tile values. Enclosing and stage timings include this work. Full pixel
downloads requested with `keep_pixel_depth=True` are separate diagnostics.

`gdmgs.query.ori.build_ori` is the older continuous-union reference, retained for
small exact geometry fixtures and earlier diagnostic evidence. Its CGAL lazy
exact arithmetic is not used by the production pixel-depth path. The native
module still includes that reference for independent fixture comparison.

## Build

Build `native/CMakeLists.txt` with CMake, pybind11 and a C++17 compiler. CGAL,
GMP and MPFR development headers/libraries support the retained exact reference.
`GDMGS_CGAL_ROOT` may point to a scoped package extraction containing `usr/`;
no system installation is required. Set `GDMGS_NATIVE_DIR` to the directory
containing the resulting `GDMGS_mesh_native` extension.

Pixel rasterization uses the validated nvdiffrast `0.3.3` CUDA backend with the
existing Torch environment. Its initial CUDA extension build is an offline
setup cost. Set `CUDA_HOME` and `TORCH_EXTENSIONS_DIR` explicitly, and ensure
the environment's `ninja` executable is on `PATH`. Use `RasterizeCudaContext`;
an OpenGL context is not needed on the A100.

Persistence contains actual vertices, faces, node bounds, child/leaf intervals
and the reference permutation. Loading validates counts, dtype, topology,
geometry enclosure and every face reference without rebuilding the BVH.

## Optional GPU query path

`gdmgs.mesh_index.gpu_index.GPUMeshIndex` uploads the validated triangle table
and BVH leaf partition once. Its `brute_force` backend runs the complete
triangle predicate on every face in parallel. Its `bvh` backend scans all leaf
AABBs in parallel, then skips the expensive triangle predicate for faces whose
leaf is irrelevant. It is a **GPU BVH leaf-cluster scan**, not traversal from
the root. Both launch a global face mask, and stable compaction produces sorted
CUDA `int64` global triangle-row IDs. Counters report the full triangle thread
slots separately from faces that actually ran the geometric predicate.

```python
from gdmgs.mesh_index.gpu_index import GPUMeshIndex

gpu_index = GPUMeshIndex(index, device="cuda:0")
gpu_query = gpu_index.query(camera_domain, backend="bvh")
gpu_ori = rasterizer.build(gpu_query, camera_domain, (width, height),
                           download_tiles=False)
# Pass gpu_ori.gpu_depth_bounds (CUDA float32), gpu_ori.angular_domain,
# and gpu_ori.image_size directly to the GPU anchor query.
# Collect these small statistics after the enclosing frame has been timed:
mesh_counts = gpu_query.collect_counters()
ori_counts = gpu_ori.collect_counters()
```

This path uploads only small camera parameters per query. Triangle IDs and
tile depths remain on the GPU; `depth_bounds` is `None` when tile download is
disabled. Necessary compaction/output-size synchronization remains part of the
algorithm. Logging counters and CUDA event reads are deferred; their collection
does not supply inputs to the next query stage. `host_enqueue_ms` is explicitly
distinct from the completed CUDA event interval and the caller's synchronized
frame wall time.

The CUDA query extension is compiled in the explicitly configured
`TORCH_EXTENSIONS_DIR`. It uses float64 coordinates with coefficient and
arithmetic uncertainty margins. Nonfinite intermediate dot products retain
uncertain triangles. The CPU implementation remains a reference; GPU full scan
and GPU leaf-cluster scan are the same-device comparison for GPU experiments.

## Validation evidence

- `tests/gdmgs/test_mesh_index.py`: builder/reference agreement, topology,
  persistence, geometry/camera identity and retained continuous reference.
- `tests/gdmgs/test_depth_pyramid.py`: original pixel centers, depth clipping,
  padded angular-domain calibration.
- Independent `tests/gdmgs/test_pixel_depth_oracles.py`: full-pixel CPU ray
  oracle versus real CUDA rasterization, including asymmetric poses, sloped
  depth, near/far clipping, pixel holes, between-pixel holes, padding and empty
  geometry. GPU cases require `GDMGS_TEST_PIXEL_GPU=1` and a leased device.
- `tests/gdmgs/test_gpu_mesh_index.py`: complete leaf binding, GPU full/leaf
  equality, CPU reference inclusion, active pruning, extreme finite-coordinate
  overflow, immutable query IDs and direct GPU pixel-depth handoff. Its GPU
  cases require `GDMGS_TEST_GPU_INDEX=1` and a leased device.

Nvdiffrast's primary documentation specifies its clip-space, point-sampled
coverage, interpolation and memory conventions:
<https://nvlabs.github.io/nvdiffrast/>.

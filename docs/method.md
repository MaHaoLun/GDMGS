# Method alignment

The supplied method text is the specification for this consolidation. The
[Notion project record](https://www.notion.so/3d8efdb220d0818d8259ef0bfe6a1d0e)
and its [sharing](https://www.notion.so/3e3efdb220d081f3a10ae330ddb679a1) and
[scheduling](https://www.notion.so/3ddefdb220d081a1a86ccd20332ee528) pages provide
source lineage and historical experimental scope, not validation of this code.

| Method operation | Implementation | Consolidation choice |
| --- | --- | --- |
| Pose-independent anchor bounds | `anchor_bounds` | Candidate-center envelope plus caller-certified radius |
| Morton sort and radix grouping | `AnchorIndex.build` | New portable builder with stable equal-key handling and contiguous subtree intervals |
| Cull / Keep / Descend | `anchor_walk` | Extracted reference, sorted at the public boundary |
| Boundary and free-space classification | `make_grid` | Extracted conservative triangle-AABB marking and 26-neighbor flood, including outside seeds |
| Merge eight solid children | `make_grid`, `OccluderIndex` | Maximal full cells, implicit full-child subdivision at the near plane |
| Hole construction and active-hole propagation | `hole_planes`, `anchor_walk` | Extracted single-hole proof predicate; no union-coverage approximation |
| Sorted group union and membership | `union_plan`, `SharedRows` | Adapted from G4, dynamic Boolean table supports actual tail length |
| Center source pose | `source_camera` | Adapted center interpolation/SLERP with explicit world-to-camera convention |
| One source decode | `ProxyGSDecoder` | Extracted opacity/attribute split; explicit union is never reselected |
| Per-target projection and sorting | `GSplatRenderer` | Shared attributes, private membership-masked opacity; gsplat performs target projection/sort |
| Selection on both processors | `CPUSelector`, `TensorSelector` | New portable tensor port of the reference predicates |
| Batch barrier and rate allocation | `Schedule`, `cpu_quota` | Adapted integer allocation, complete groups, CPU transfer finishes before barrier |
| Retained-row capacity | `Admission` | FIFO, exact opacity count, release after final target completes |

## Versions deliberately kept separate

`GDMGS_Codebase` is an earlier fresh/compatibility pipeline. Later ProxyGS
experiments added a matching gsplat backend and several cache/schedule variants.
The October 1 combined anchor/triangle traversal runs with occlusion disabled;
it is not the supplied method's solid-cell holed-frustum implementation. The
older depth-map filtering path and pair2 cache policy are also different methods.
They are therefore not silently selected as this package's default.

The old optimized low-level gsplat renderer fused membership into preprocessing.
This package reuses the simpler extracted gsplat API and masks opacity before
that call. It still projects shared rows, so it does not claim identical work,
memory use, or performance to the optimized kernels.

## Preconditions and limits

- The mesh must describe opaque solid interiors and training cameras must seed
  the relevant free-space components. Flood fill alone cannot prove every
  unreachable cavity is material. Holes/open surfaces can reduce pruning.
- Triangle AABBs overmark boundary cells. The reference grid costs O(8^L) memory
  and supports L=0..8 explicitly. It does not silently reduce a requested level.
- Anchor and occluder domains must match. Caller-certified radii must account
  for the decoder and renderer support; a naive world-space three-sigma bound
  is not automatically a proof for a rasterizer with screen-space dilation.
- CPU/GPU LoD eligibility must match. The archive recorded isolated floating
  LoD-boundary exceptions; this package does not accept mismatches automatically.
- Torch traversal is host-driven and synchronization-heavy. The package contains
  executable CUDA-device code, but CUDA was unavailable for local qualification.
- A row capacity bounds retained shared attributes only. Decoder candidates,
  ID lists, masks, projection, sorting, and image workspaces need separate memory.
- Render callbacks must be read-only, synchronous, and return independent
  outputs. The supplied adapter returns host images. Host output storage grows
  with trajectory length; only selected-ID batches and shared rows are bounded.
- Core grouping accepts any positive K. The inherited experiments qualified
  particular K values and workloads; arbitrary K has no inherited quality claim.
- This is inference code, not a replacement for ProxyGS training or its loader.

New integration code is listed explicitly in the source manifest. Existing
experiment directories and artifacts were left intact.

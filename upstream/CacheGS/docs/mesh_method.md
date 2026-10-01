# Mesh construction for the unified-index experiment

Proxy-GS uses different geometry sources for different datasets. Dense LiDAR
points are reconstructed into surfaces; synthetic ground-truth depths are fused
with TSDF; sparse indoor COLMAP inputs use MoGe2 plus PGSR; sparse outdoor inputs
use CityGS-X. These are described in the authors' [Appendix A.4](https://arxiv.org/html/2509.24421v2#A1.SS4).

Our twelve in-setting frozen checkpoints have sparse COLMAP inputs. Exporting
their expected depths and fusing them is a separate candidate method; it is not
the paper's sparse-input reconstruction workflow or a ground-truth-depth source.
MatrixCity is outside this experiment, following the user's scope decision.

The first comparison uses exactly the same complete train-camera inventory and
native configured render dimensions for both candidates:

1. Fresh `RGB+ED` and alpha export, followed by confidence-masked Open3D TSDF.
2. Continuous depth-image grid triangles from all views. Each rectangle checks
   every interior source pixel, alpha, and depth spread. A neighboring-view
   support filter checks all triangle vertices. Missing and discontinuous regions
   remain holes; no views are welded and no holes are filled.

Depth is positive camera-z in the loaded model coordinate system. The export
saves the actual W2C matrix and pinhole intrinsics. GS raster pixels are centered
at `(column + 0.5, row + 0.5)`; Open3D's integer-pixel convention receives an
intrinsic principal-point offset of `-0.5`. Analytic tests check the convention,
rotation, translation, holes, and depth discontinuities.

`tools/gdmgs/build_mesh.py` has separate `export`, `export-scenes`, `build`,
`filter-consistency`, `weld-source-grid`, `simplify`, and `assess` commands.
Meshes use float64 vertices, int64 faces, and stable cleaned face-row triangle
IDs. Each `.npz` has a neighboring `.json` containing the explicit run label,
checkpoint path/iteration, complete fusion inventory, parameters, and extraction
cost. There are no content digests. Source checkpoints are read-only; file
size/mtime inventories check that export did not modify them.

The geometry assessment measures ray coverage and depth disagreement on every
fusion view. Its explicitly sampled pixel grid is a diagnostic, never an ORI
continuous-coverage certificate or final image-quality acceptance. Selection is
pending measured construction cost, coverage, cross-view consistency, and the
full fresh-render gate: each evaluation frame loses at most 0.1 dB PSNR and 0.002
SSIM against the same ground truth. Neither candidate is currently called best.

## Measured candidate evidence, 2026-09-08

All twelve scenes exported their complete native-resolution train inventories:
2,254 frames in total. Every depth manifest is complete and records an unchanged
checkpoint size/mtime inventory. MatrixCity was never included in this export.
Artifacts are below `/ssddata/lun/gdmgs_artifacts/index_20260908/<scene>/` on
`zxcpu2`. `depth_manifest.json` enumerates every input and exported observation;
each method directory contains `mesh.npz`, `mesh.json`, and its geometry report.

The global-consistency variant checks every candidate vertex against every
fusion view. It requires support from at least two views and rejects a vertex
if any high-alpha observation places it over 2% in front of the observed depth.
Every triangle must satisfy this rule at all its vertices. Unknown or occluded
observations do not count as disagreement. This filter does **not** certify a
triangle's projected interior; the residual front-depth discrepancies in the
table demonstrate that limitation. Native render quality remains mandatory.

| Scene | Complete frames | Grid faces | Global-consistency faces | Observed coverage after global filter | Front discrepancy after global filter |
| --- | ---: | ---: | ---: | ---: | ---: |
| amsterdam | 161 | 618,269 | 520,318 | 80.55% | 3.64% |
| barcelona | 160 | 259,487 | 243,841 | 68.94% | 5.69% |
| bilbao | 129 | 552,279 | 519,457 | 84.92% | 10.83% |
| chicago | 160 | 164,902 | 129,438 | 53.09% | 3.69% |
| drjohnson | 263 | 170,078 | 66,059 | 24.90% | 1.53% |
| hollywood | 125 | 216,632 | 132,326 | 53.33% | 3.01% |
| playroom | 225 | 165,405 | 59,080 | 32.66% | 0.74% |
| pompidou | 161 | 437,104 | 323,656 | 62.85% | 11.53% |
| quebec | 160 | 559,863 | 490,988 | 80.75% | 8.47% |
| rome | 158 | 374,583 | 306,611 | 69.93% | 9.80% |
| train | 301 | 40,162 | 3,463 | 2.86% | 0.18% |
| truck | 251 | 81,697 | 19,317 | 15.53% | 0.38% |

Coverage is the fraction of sampled high-alpha source pixels with a mesh ray
hit. Front discrepancy is the fraction of paired hits whose mesh depth is more
than 2% nearer than source ED. Every source view is measured, using diagnostic
pixel stride 8. These columns are not native-pixel image acceptance or complete
continuous geometry proofs.

Truck's raw grid took 61.4 seconds after its 40.6-second complete depth export;
it covered 43.06% of observed diagnostic rays, with 23.21% front discrepancy.
Global vertex filtering took another 12.9 seconds and reduced the discrepancy
to 0.38%, at the coverage cost shown above.

Truck's fine TSDF used voxel length 0.0134998. All 251 views integrated, but
extraction still had no mesh after more than 16 minutes and resident memory
exceeded 190 GiB. This development candidate was deliberately terminated for
cost. It is neither a successful extraction nor a demonstrated capacity failure;
`tsdf_fine_cost_rejection.json` preserves its process evidence.

The independently recorded coarse TSDF uses a 4x voxel length, 0.0539992, with
the same 251 views and confidence threshold. It produced 4,714,999 triangles in
99.7 seconds (252.8 MB). Its observed diagnostic coverage was 91.11%, with 21.30%
front discrepancy. Full-view vertex filtering retained 2,010,613 triangles in
an additional 86.2 seconds. No quality threshold changed between candidates.

## Simplification and boundary verification

For depth grids, identical coordinates are merged **only within the same source
view**. Sorted source-ID/coordinate records determine the merged vertices. A
direct array comparison verifies that every ordered triangle retains exactly
the same three world coordinates; no triangle is added or removed by welding.

The QEM candidate uses PyMeshLab 2023.12.post3 with boundary, topology, and normal
preservation enabled, optimal placement disabled, and additional automatic
cleanup disabled. The [PyMeshLab filter documentation](https://pymeshlab.readthedocs.io/en/latest/filter_list.html#meshing-decimation-quadric-edge-collapse)
describes these controls. The implementation additionally verifies connected
components, Euler characteristic, nonmanifold-edge count, and the exact multiset
of geometric boundary segments. It does not trust a library flag alone or
silently disable a constraint to reach the requested 100,000 faces.

Amsterdam's globally filtered grid dropped from 520,318 to 266,110 triangles
in 10.56 seconds after source-view welding, with the audit passing. The coarse
Truck TSDF dropped to 2,536,873 triangles in 110 seconds, also passing the audit.
It has 465,358 disconnected components, so a topology-preserving 100,000-face
result is already impossible from component count alone. These sizes favor
the grid candidate for index cost, but final adoption still requires the
paired fresh-render image gate.

The geometry unit suite currently contains 24 passing analytic tests, including
non-affine calibration rejection, source-view-only welding, and coincident
boundary components. Nonauthor reviews checked the depth/calibration, hole, and
provenance contracts; the review's strict affine-row validation suggestion was
implemented and tested. A second review found that mesh/depth-manifest source
binding needed enforcement: assessment and filtering now reject a differing
checkpoint path, iteration, or explicit depth-run label before reading frames.
The source-binding audit passed for all 31 meshes that existed at that audit;
subsequent construction uses the same validation at its entrypoint.

## Representative-scene comparison

The coarse TSDF comparison also completed every Amsterdam and drjohnson fusion
view. Amsterdam produced 30,378 faces in 7.03 seconds, with 99.35% observed
coverage and 19.61% front discrepancy. Drjohnson produced 407,588 faces in
10.41 seconds, with 98.98% coverage and 21.02% front discrepancy. TSDF therefore
has very different costs across these scenes; Truck's cost does not justify
rejecting it universally.

Global consistency reduced Amsterdam's TSDF to 13,860 triangles in another
2.74 seconds. Its coverage is 27.19%, with 0.63% front discrepancy. Drjohnson's
globally filtered TSDF has 37,538 triangles after a 7.23-second filter.

Amsterdam's QEM result provides a warning against using triangle count alone:
despite unchanged geometric boundary segments and topology statistics, its
interior surfaces changed enough that front discrepancy increased from 3.64%
to 26.81%. Coverage increased from 80.55% to 82.62%. Boundary/statistical checks
therefore do not establish geometry or rendering accuracy, and QEM is not
adopted on the basis of its smaller face count.

The useful candidates for the actual fresh-render comparison are the complete
grid with global consistency and the coarse TSDF with global consistency.
Choosing between their coverage, discrepancy, and cost requires the complete
per-frame image gate and actual index measurements. No mesh has yet been
declared the final winner by this geometry-only report.

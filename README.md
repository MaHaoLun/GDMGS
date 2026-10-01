# GDM-GS — full source integration

This repository uses **CacheGS/GDMGS_Codebase as its default training and model
backend**, together with the GDM-GS native indices, shared decoding and scheduling.
The independently trained models used by later experiments remain available
through an explicit `proxygs` compatibility backend.

The integrated pipeline loads a real checkpoint and camera inventory, builds
native anchor indices, retrieves solid occluder cells, selects original anchor
IDs, materializes each group's union once, and renders every target with the
retained staged gsplat/Triton renderer. It does not require the old experiment
directory, a saved selected-ID trace, or a prebuilt private `.so`.

See [validation](docs/validation.md) for completed checks and remaining GPU
qualification, and [method alignment](docs/method.md) for implementation details.

## Contents

- `upstream/CacheGS/`: read-only snapshot downloaded from zxcpu2
  `/home/zyl/lun/GDMGS_Codebase`: original training, YAML model configuration,
  checkpoint loading, pose-local explicit-ID decoding and fresh renderer.
- `upstream/ProxyGS/`: historical compatibility snapshot: training, loaders, original inference,
  metrics, mesh-depth tools, model code, native rasterizer/backpropagation,
  simple-knn, mesh/anchor indices, tests, and third-party source dependencies.
- `system/`: retained full-bundle cache, split decoder, staged renderer and
  intersections, native CPU/CUDA selection, solid-cell retrieval, complete-group
  batch scheduling, and configurable checkpoint loading.
- `train.py`: original CacheGS training by default; `--backend proxygs` is explicit.
- `render_fresh.py`: the preserved CacheGS fresh path, independent of the new shared pipeline.
- `research/retained/`: source snapshots of the reviewed query, sharing and worker-48 schedule implementations.
- `docs/source_audit.json`: both checkpoint lineages, source paths and anchor counts.
- `render.py`: integrated checkpoint-to-image entrypoint.
- `scripts/`: native builds, solid-cell preparation, measured-rate calibration.
- `tests/`: real compiled C++ regression tests and device-geometry predicate tests.
- `docs/source_inventory.json`: per-file source inventory and adaptation records.

## Build the full CUDA environment

Use Linux, Python 3.10, a working NVIDIA CUDA toolkit, a C++ compiler and OpenMP.
The historical measured environment used Torch 2.4.0+cu124, torchvision
0.19.0+cu124 and gsplat 1.4.0+pt24cu124. The installer builds from the bundled
source and has not been validated as a fresh install in this task.

```sh
bash scripts/install_cuda.sh cachegs
```

CacheGS requires the existing compatible **fVDB 0.0.1 build**, whose commit and
local build fixes are recorded in `dependencies/fvdb-observed.json` and
`dependencies/fvdb-local.patch`. The installer checks for it and does not replace
it with an arbitrary package of the same name. The observed source is
`/home/zyl/XCube/openvdb/fvdb`; reproducing this dependency is a separate build.
For the historical backend use `bash scripts/install_cuda.sh proxygs`.

`requirements-runtime.txt` deliberately uses Python-3.10-compatible NumPy/SciPy
constraints instead of blindly copying the newer constraints in the historical
upstream README. To exercise the optional legacy CPU triangle-mesh index,
install CGAL/GMP/MPFR and build `upstream/ProxyGS/gdmgs/mesh_index/native` with
CMake/pybind11; set `GDMGS_NATIVE_DIR` to the resulting module directory.
The integrated solid-cell pipeline does not instantiate that legacy mesh index.
The Vulkan viewer has its separate preserved CMake build and is optional.

## Original CacheGS training

```sh
python train.py --help
python train.py --config /absolute/path/to/training.yaml
```

Use the original `model_params`, `pipeline_params`, and `optim_params` YAML
structure. The original training algorithm is unchanged. Its entrypoint retains
its own GPU-selection and timestamped-output behavior; inspect the preserved
training source before scheduling jobs on a shared host.

The eight January CacheGS training backups match this training file after only
the output-directory literal is changed. The later `proxygs_step2` checkpoints
have separate commands, different anchor counts and a different loader; files
with the same `point_cloud.ply`/MLP names must not be assumed interchangeable.
Their training path requires explicit `--backend proxygs` and its original flags.

## Prepare solid cells

```sh
python scripts/prepare_occluders.py --mesh /data/mesh.npz \
  --cameras /data/training_cameras.json --level 9 --output /data/cells.npz
```

The mesh NPZ has `vertices` and `triangles`. Camera JSON is a list of records
with `w2c` matrices. Mesh file formats supported by trimesh are also accepted.
All input cameras/triangles are used. Dense occupancy costs O(8^level) memory;
the script does not reduce the requested level or cap the number of cells.
For old cell files lacking a level field, set `occluder_level` explicitly.
Opaque-solid interiors and free-space camera seeds are required assumptions.

## Render all targets

Copy `configs/render.example.json` and set the actual model, dataset and cell
paths. `model_backend` defaults to `cachegs`, which loads `config.yaml` through
the original loader. `configs/render.proxygs.example.json` explicitly selects
`cfg_args` compatibility. Mixing both source namespaces in one process fails.
CacheGS scale/rotation rounding and anchor-based LoD positions are preserved.
By default, all checkpoint camera IDs are rendered; an explicit
`camera_ids` list can specify a frozen trajectory. `interpolation=4` produces
125 targets from 32 supplied cameras. Set `expected_targets` to enforce the
intended denominator. No target limit or silent capacity fallback exists.

```sh
CUDA_VISIBLE_DEVICES=0 python render.py --config my_scene.json \
  --output outputs/scene --verify
```

`--verify` checks **every** CPU/GPU selection, each group against the original
unsplit decoder, and every shared-render output against a materialized subset
rendered by the original gsplat API. Failures abort the run. RGB, alpha and
depth are checked; image comparison uses explicit `atol=rtol=1e-5`, while IDs
and decoder tensors require exact equality. Source-pose reuse versus per-target
fresh decoding is a separate quality experiment, not this equivalence check.

The renderer writes numbered PNGs, progress, configuration and a report. Existing
output directories are rejected. Validation/image saving are inside the reported
wall time, so that value must not be presented as historical benchmark FPS.

## Hybrid selection

Calibrate on the full trajectory at the configured CPU/GPU worker counts:

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/calibrate_selection.py \
  --config my_scene.json --output my_scene.calibrated.json
CUDA_VISIBLE_DEVICES=0 python render.py --config my_scene.calibrated.json \
  --output outputs/hybrid --verify
```

The CPU quota uses the better of floor/ceil of `n*c_GPU/(c_CPU+c_GPU)` and
interleaves assignments. Each batch contains complete groups. Every query and
CPU ID transfer finishes before materialization starts. Group admission is FIFO
and reserves exact opacity-retained rows. A single over-capacity group fails;
rows are released after its targets complete. This is a row budget, not a bound
on total GPU workspace or image storage. The retained renderer supports K=1..4
and one or two target workers; unsupported settings fail explicitly.

## Local checks

```sh
python -m pip install numpy scipy torch pytest einops
python -m pytest -q tests
python render.py --help
```

Native tests require a C++ compiler/OpenMP. On macOS set `LIBOMP_PREFIX` if libomp
is not in its usual Homebrew prefix. The test build links the same OpenMP runtime
as PyTorch when available; it does not suppress duplicate-runtime errors.

## License

The original root `LICENSE` is preserved. `upstream/ProxyGS` and the derived
`system` code retain the inherited research-use license and third-party notices.
The root Apache license does not relicense those components. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

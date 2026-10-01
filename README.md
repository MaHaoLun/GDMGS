# GDM-GS — full source integration

This repository contains the **complete ProxyGS training/model/checkpoint code,
original native extensions, and the GDM-GS inference pipeline** assembled from
the VISTA experiment versions. The earlier reference-only draft has been
removed from the working tree; it remains in Git history.

The integrated pipeline loads a real checkpoint and camera inventory, builds
native anchor indices, retrieves solid occluder cells, selects original anchor
IDs, materializes each group's union once, and renders every target with the
retained staged gsplat/Triton renderer. It does not require the old experiment
directory, a saved selected-ID trace, or a prebuilt private `.so`.

**Validation status:** local native CPU and geometry tests are available.
The new integration and its CUDA changes have not been run on a GPU. Do not
interpret source completeness as completed image-quality/performance acceptance.
See [validation](docs/validation.md) and [method alignment](docs/method.md).

## Contents

- `upstream/ProxyGS/`: full source snapshot: training, loaders, original inference,
  metrics, mesh-depth tools, model code, native rasterizer/backpropagation,
  simple-knn, mesh/anchor indices, tests, and third-party source dependencies.
- `system/`: retained full-bundle cache, split decoder, staged renderer and
  intersections, native CPU/CUDA selection, solid-cell retrieval, complete-group
  batch scheduling, and configurable checkpoint loading.
- `train.py`: original proxy-aware training entrypoint.
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
bash scripts/install_cuda.sh
```

`requirements-runtime.txt` deliberately uses Python-3.10-compatible NumPy/SciPy
constraints instead of blindly copying the newer constraints in the historical
upstream README. To exercise the optional legacy CPU triangle-mesh index,
install CGAL/GMP/MPFR and build `upstream/ProxyGS/gdmgs/mesh_index/native` with
CMake/pybind11; set `GDMGS_NATIVE_DIR` to the resulting module directory.
The integrated solid-cell pipeline does not instantiate that legacy mesh index.
The Vulkan viewer has its separate preserved CMake build and is optional.

## Proxy-aware training

All original training options remain in the bundled entrypoint:

```sh
python train.py --help
python train.py -s /data/scene -m /models/scene \
  --ply_path /data/scene/points.ply --ply_mesh /data/scene/proxy_mesh.ply \
  --depth_npy_dir /data/scene/proxy_depth --iterations 40000
```

Use the dataset/model settings from the actual experiment's saved training
protocol. This command shows path wiring; it is not a replacement for those
settings. Existing checkpoints are loaded without retraining or format conversion.
Original metrics and native renderer entrypoints remain under `upstream/ProxyGS`.

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
paths. By default, all checkpoint camera IDs are rendered; an explicit
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
python -m pip install numpy scipy torch pytest
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

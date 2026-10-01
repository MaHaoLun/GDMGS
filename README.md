# GDM-GS

A compact method-core consolidation of the  GDM-GS research code.
It follows the supplied method's three stages: spatial selection, group-shared
materialization, and a CPU/GPU selection barrier followed by bounded rendering.


## Layout

| Module | Responsibility |
| --- | --- |
| `index.py` | Morton-packed binary radix anchor index; original row IDs |
| `occupancy.py`, `occluders.py`, `geometry.py` | Conservative mesh-cell construction, octree retrieval, single-hole pruning |
| `selection.py`, `tensor_selection.py` | Same geometric predicates on NumPy CPU and Torch CPU/CUDA |
| `materialization.py` | Sorted union, membership, one source-opacity pass and one attribute decode |
| `schedule.py` | Rate-based CPU quota, complete-group batch barrier, FIFO exact-row admission |
| `adapters/` | Extracted ProxyGS decoder/row contract and gsplat backend |

No experiment logs, datasets, weights, private absolute paths, prebuilt native
libraries, old cache policies, viewer, or benchmark matrices are required by
the core package. Training/checkpoint loading stays with the original ProxyGS
runtime; this package consumes an already-loaded model.

## Install and check

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python examples/query.py
```

The query example needs only NumPy/SciPy (`pip install -e .`). It is a small
geometry demonstration, not a real-scene quality or speed benchmark. On Linux
with a matching CUDA/PyTorch installation, `pip install -e '.[render]'` adds the
optional gsplat adapter. The historical baseline used Torch 2.4.0+cu124 and
gsplat 1.4.0; this release does not install or recreate that full environment.

## Use the pipeline

See [the integration example](examples/render.py) for the complete API wiring:
build both indices in the same domain, attach the same LoD eligibility rule to
both selectors, bind an existing ProxyGS model, and run `Schedule.run`.
All cameras use **`x_camera = R x_world + t`**, with center **`-Rᵀt`**.
Anchor bounds are caller-supplied conservative bounds covering decoded support;
they must use the checkpoint's original PLY row order.

`TensorSelector(device='cuda')` is a host-driven Torch correctness backend.
It includes scalar synchronization and is not the fast fused CUDA implementation
from the experiment archives. Calibrate effective per-target selection costs
at the chosen concurrency, including CPU ID transfer, before supplying
`cpu_cost` / `gpu_cost`. The defaults are illustrative equal rates.

The default ProxyGS adapter supports the original restricted inference model:
no feature bank, appearance, level-input, or distance-input heads; `round` LoD.
Unsupported model settings fail explicitly. Source-pose reuse remains an
approximation and needs a dataset-specific quality check.

## Method and evidence

- [Method-to-code mapping and differences](docs/method.md)
- [Source provenance](docs/source_manifest.json)
- [Validation and remaining qualification](docs/validation.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

The original repository license remains at `LICENSE`. The extracted model and
raster adapters in `gdmgs/adapters/` retain their inherited research-use license;
the root Apache license does not relicense those files.

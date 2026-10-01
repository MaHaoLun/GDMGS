# Validation status — full integration

## Work performed locally

Date: 2026-10-01. Host: macOS arm64. Test environment: Python 3.13,
NumPy 2.5.3, SciPy 1.18.1, PyTorch 2.14.1. This is a CPU test environment,
not the historical CUDA rendering environment.

```sh
python -m pytest -q tests
python render.py --help
python -m compileall -q system scripts
```

**13 tests passed.** Native tests compile and load the actual C++/OpenMP source.
They cover geometric rejection, scale-expanded hole containment, native hole
planes versus the scalar implementation, indexed-versus-dense agreement on all
513 generated records with 1/2/4 threads, integer allocation, round-to-even LoD
thresholds, sparse-cell near-plane subdivision and tensor hole geometry.
Four model-boundary tests additionally check the unchanged CacheGS container
precision/ownership, avoidance of mutable fVDB source-state updates, explicit
backend namespace isolation, and the CacheGS training default.
Tensor geometry runs on CPU here. It is not evidence that CUDA kernels execute.

An initial combined test run exposed two OpenMP runtimes on macOS. The build
helper now links the same runtime used by PyTorch and corrects its embedded
loader path on the newly built library. No global Torch installation was edited
and no duplicate-runtime-suppression flag was used. The subsequent suite passed.

The CacheGS source snapshot was retrieved read-only from zxcpu2 and its fVDB
source version/build patch recorded. Both model families have explicit loaders;
CacheGS is the default. The tensor-container test executes the original
CacheGS container classes without invoking their fVDB-dependent methods. It
does not claim a real checkpoint load or rendering validation.

The source tree and original license files are preserved. Machine-readable
source mappings and byte comparisons are in `source_inventory.json`.

## Full CUDA validation provided, but not executed

`python render.py --config SCENE.json --output NEW_DIR --verify` is the actual
checkpoint-driven validation entrypoint. It checks all selected IDs on both
processors, every group's split decode against the original decoder, and every
RGB/alpha/depth output against the original gsplat handoff for that same shared
source. Original-ID and decoder comparisons are exact; output comparison uses
`atol=rtol=1e-5`. No reduced-view or synthetic substitute is used by that command.

The user explicitly requested local work and declined sending this source to
zxcpu2. No source was uploaded and no GPU test was performed on that server;
existing sources and dependency metadata were only read/downloaded. There is no
NVIDIA GPU in the local validation environment. Therefore:

- CUDA source compilation and runtime import closure remain unverified.
- The new threshold LoD rule and support-expanded hole tests need all-scene
  comparison; old archived PASS results do not cover those changes.
- No real checkpoint was loaded, no new training was run, and no new RGB/depth
  quality or performance result was produced by this task.
- Source-reuse quality versus independent fresh target decoding remains a
  separate acceptance gate. The renderer-equivalence check does not prove it.
- Fresh environment installation and optional viewer/legacy mesh-index builds
  remain untested. Their complete sources and build instructions are included.

The PR must remain a draft until those checks pass. In particular, this document
does not label the complete integration as GPU-validated or publication-ready.

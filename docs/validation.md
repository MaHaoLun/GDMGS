# Validation of the consolidation

Local validation on 2026-10-01 used macOS arm64, Python 3.13, NumPy 2.5.3,
SciPy 1.18.1 and PyTorch 2.14.1 in a separate temporary environment.

Commands:

```sh
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q
python examples/query.py
python -m compileall -q gdmgs examples
git diff --check
```

Result: **19 passed, 1 skipped**. The skipped test requires CUDA hardware.
The example returned original anchor ID `[0]`. Editable package installation
and compilation passed. CI repeats the CPU tests on Linux/Python 3.11;
its result must be checked separately after publication.

Tests cover dense-oracle agreement of the indexed query and tensor port,
near-plane subdivision, partial-hole conservativeness, duplicate Morton keys,
empty inputs, closed/open mesh flood fill, camera centers, integer allocation,
row ownership, target membership, exact row admission, failure cancellation,
complete-batch barriers, output ordering, and partial tail groups. A mocked
raster API test checks shared-attribute identity and target opacity isolation;
it does not validate gsplat image output.

## Required before claiming a production or paper-result release

1. Run CUDA selector parity on the intended GPU and all frozen scene cameras,
   using the same eligibility rule and certified bounds on both processors.
2. Load the original frozen ProxyGS checkpoints. Compare split decode tensors
   and row identity to the original decoder, including empty and zero-row cases.
3. Run real gsplat output comparisons for RGB, alpha and depth, then the full
   source-reuse quality protocol. The scalar clip settings are now passed from
   `Camera` to gsplat explicitly and must be included in the comparison.
4. Measure selection calibration, CPU transfer, batch storage, shared rows,
   workspace and end-to-end wall time at the selected worker counts.
5. Validate model/dataset lineage and unchanged full denominators. Unit tests
   and synthetic examples do not replace those scene-level checks.

No GPU run, model training, checkpoint render, historical FPS reproduction,
or full-data quality acceptance was performed by this consolidation task.

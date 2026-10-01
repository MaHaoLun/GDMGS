# GDM-GS Codebase

An isolated extension of CacheGS implementing stages A and B: compatible
entrypoints, PLY-row anchor identity, pose-local selection, complete decoder
ownership, and explicit-selection fresh rendering.

- [Usage and runtime](docs/gdmgs_ab_usage.md)
- [Implementation and validation results](docs/validation.md)
- [Reviewed cleanup](docs/cleanup.md)

The default renderer and cache retain the legacy path. `--pipeline gdmgs`
selects the new fresh pipeline. Mesh/BVH/ORI, bundle caching and lookahead
scheduling belong to later stages.

## Entry points

```sh
python train.py --config config/scaffoldgs/lod_model.yaml
python render.py -m <checkpoint-run> --iteration 40000 --output-root <render-output>
python render.py -m <checkpoint-run> --iteration 40000 --pipeline gdmgs --output-root <fresh-output>
python render2.py -m <checkpoint-run> --iteration 40000 --skip_train --output-root <split-output>
python precompute_cache.py -m <checkpoint-run> --iteration 40000 --output-root <precompute-output>
python metrics.py -m <render-output> --output-root <metric-output>
```

Set `CUDA_VISIBLE_DEVICES` before launching and keep project outputs outside
the checkpoint directory. Existing cache flags and precompute payload format
remain available; `--help` describes the compatible options. Original training
formulas and checkpoint format are unchanged.

Tests run in the existing rendering environment with `python -m pytest`.
Real CUDA fixtures require the explicit environment variables described in
the tests. The validation report distinguishes 13 actual model loads from
12 numerically valid inputs and the original MatrixCity checkpoint blocker.

See [LICENSE.md](LICENSE.md) for the inherited research-use license.

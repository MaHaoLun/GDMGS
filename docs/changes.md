# Changes from the initial PR draft

The initial Python/Torch reference package is removed from the current tree.
The original commit remains in Git history. It is superseded by:

- Complete original training, model, loaders, metrics, and native dependencies.
- Retained split decoder, fused bundle identity, shared membership, staged
  intersections and target-specific gsplat rendering.
- Real native index construction and querying instead of a host-driven tensor
  replacement for the anchor index.
- Configurable checkpoint-to-image loading, no archived source-directory imports.
- Full trajectory execution, explicit denominator checks, real verification
  entrypoint, calibration, mesh-cell preprocessing and build tooling.
- Corrected support-aware hole predicates and a common algebraic LoD rule.

These integration/correctness changes intentionally invalidate inherited
performance or image-quality claims. Source completeness and runtime acceptance
are recorded separately in `validation.md`.

## Training/model source correction

The default is now the remotely verified CacheGS/GDMGS_Codebase source, not
ProxyGS training. Root training and model configuration select CacheGS by
default, with explicit compatibility for the later experimental checkpoints.
The adapter preserves CacheGS scale/rotation rounding, original row ownership,
and anchor-based LoD positions. Source-pose preparation does not mutate shared
fVDB attributes. The original fresh entrypoint remains separately runnable.

The server's actual fVDB build commit and patch were recorded. Reviewed query,
sharing and worker-48 scheduling source snapshots are retained for traceability.
The user-requested query/schedule distinction remains deferred.

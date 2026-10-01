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

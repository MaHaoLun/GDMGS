# Retained research implementations

These files were downloaded read-only from zxcpu2 on 2026-10-01. They preserve
original query, sharing and CPU/GPU scheduling implementations plus their design
records. They are source references; their historical launchers still contain
experiment paths and hardware choices and are not the portable root entrypoints.

- `joint_query_table_20261001/vendor/`: original combined-tree CUDA construction
  and v7 query implementation.
- `proxygs_step7d_factor_cache_20260925/centered_k/`: retained union, epoch and
  priority-render implementations.
- `experiment_correction_20260930/fullblock_timed.py`: split source decode and
  exact row admission used by later schedules.
- `joint_schedule_worker48_controlled_20261001/`: worker-48 driver, selection
  workers and review inputs from the user-specified conversation.

No experiment is launched by importing or storing this directory. Selection
variant decisions requested to be deferred by the user remain deferred; this
snapshot does not assign new acceptance status or rewrite experimental results.

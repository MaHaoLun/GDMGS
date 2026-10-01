"""CUDA-resident pair publication; value audits run outside timing.

Publish only after the producer has completed its query. This latch does not
synchronize CUDA or create an event: the synchronous GPU query caller owns
completion. Tensor storage is retained by reference and must remain immutable.
"""
from dataclasses import dataclass
from threading import Lock
from time import perf_counter_ns
from typing import Optional, Tuple

import torch

from .pair_schedule import PairIdentity


@dataclass(frozen=True)
class GPUPairDemand:
    identity: PairIdentity
    selected_ids: Tuple[torch.Tensor, torch.Tensor]
    selection_records: Tuple[dict, dict]
    submitted_ns: int
    ready_ns: int

    def validate(self):
        """Validate metadata only; never inspect device tensor values."""
        self.identity.validate()
        if self.ready_ns < self.submitted_ns:
            raise ValueError("pair readiness precedes submission")
        if len(self.selected_ids) != 2 or len(self.selection_records) != 2:
            raise ValueError("pair demand must contain exactly two frames")
        device = None
        for ids in self.selected_ids:
            if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64
                    or ids.ndim != 1 or ids.device.type != "cuda"
                    or not ids.is_contiguous()):
                raise TypeError("selected IDs must be contiguous rank-one CUDA int64 tensors")
            if device is not None and ids.device != device:
                raise ValueError("pair demand tensors must share a CUDA device")
            device = ids.device
        if any(not isinstance(record, dict) for record in self.selection_records):
            raise TypeError("selection records must be dictionaries")

    def validate_values(self, anchor_count: Optional[int] = None):
        """Explicit synchronizing audit. Call after timed replay, not in publish."""
        self.validate()
        for ids in self.selected_ids:
            if ids.numel() and (bool((ids < 0).any()) or bool((ids[1:] <= ids[:-1]).any())):
                raise ValueError("selected IDs must be sorted unique nonnegative rows")
            if anchor_count is not None and ids.numel() and bool((ids >= anchor_count).any()):
                raise ValueError("selected ID outside immutable anchor table")


class GPUPairDemandLatch:
    """Single atomic publication after completed producer query, without D2H."""
    def __init__(self, identity: PairIdentity, *, submitted_ns: Optional[int] = None):
        identity.validate()
        self.identity = identity
        self.submitted_ns = perf_counter_ns() if submitted_ns is None else int(submitted_ns)
        self._lock = Lock()
        self._demand = None
        self._error = None

    def publish(self, selected_ids, selection_records, *, ready_ns: Optional[int] = None):
        with self._lock:
            if self._demand is not None or self._error is not None:
                raise RuntimeError("pair latch is already terminal")
            demand = GPUPairDemand(
                self.identity, tuple(selected_ids), tuple(selection_records), self.submitted_ns,
                perf_counter_ns() if ready_ns is None else int(ready_ns))
            demand.validate()
            self._demand = demand
            return demand

    def fail(self, error):
        with self._lock:
            if self._demand is not None or self._error is not None:
                raise RuntimeError("pair latch is already terminal")
            self._error = repr(error) if isinstance(error, BaseException) else str(error)

    @property
    def ready(self):
        with self._lock:
            return self._demand is not None

    @property
    def error(self):
        with self._lock:
            return self._error

    def demand(self, expected: PairIdentity):
        expected.validate()
        with self._lock:
            if expected != self.identity:
                raise ValueError("stale or mismatched pair identity")
            if self._error is not None:
                raise RuntimeError(f"pair selection failed: {self._error}")
            if self._demand is None:
                raise RuntimeError("pair demand is not ready")
            return self._demand

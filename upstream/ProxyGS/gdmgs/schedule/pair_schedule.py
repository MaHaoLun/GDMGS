"""Atomic one-pair lookahead publication used by Step 8.

The latch deliberately does not know about oracle traces or cache policy.  It
only publishes a complete pair after both online selection results have passed
identity validation.  Consumers may inspect readiness at a deadline, but a
partial pair is never observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import perf_counter_ns
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PairIdentity:
    scene: str
    pair_index: int
    frame_ids: Tuple[int, int]
    camera_ids: Tuple[str, str]

    def validate(self) -> None:
        if not self.scene:
            raise ValueError("scene must be nonempty")
        if self.pair_index < 0:
            raise ValueError("pair index must be nonnegative")
        expected = (2 * self.pair_index, 2 * self.pair_index + 1)
        if self.frame_ids != expected:
            raise ValueError("frame IDs do not match pair index")
        if len(self.camera_ids) != 2 or not all(self.camera_ids):
            raise ValueError("a pair requires two nonempty camera IDs")


@dataclass(frozen=True)
class PairDemand:
    identity: PairIdentity
    selected_ids: Tuple[np.ndarray, np.ndarray]
    selection_records: Tuple[dict, dict]
    submitted_ns: int
    ready_ns: int

    def validate(self) -> None:
        self.identity.validate()
        if self.ready_ns < self.submitted_ns:
            raise ValueError("pair readiness precedes submission")
        if len(self.selected_ids) != 2 or len(self.selection_records) != 2:
            raise ValueError("pair demand must contain exactly two frames")
        for ids in self.selected_ids:
            if not isinstance(ids, np.ndarray) or ids.dtype != np.int64 or ids.ndim != 1:
                raise TypeError("selected IDs must be rank-one int64 numpy arrays")
            if ids.size and (ids[0] < 0 or np.any(ids[1:] <= ids[:-1])):
                raise ValueError("selected IDs must be sorted unique nonnegative rows")


class PairDemandLatch:
    """Single-publication latch with fail-closed identity checks."""

    def __init__(self, identity: PairIdentity, *, submitted_ns: Optional[int] = None) -> None:
        identity.validate()
        self.identity = identity
        self.submitted_ns = perf_counter_ns() if submitted_ns is None else int(submitted_ns)
        self._lock = Lock()
        self._demand: Optional[PairDemand] = None
        self._error: Optional[str] = None

    def publish(self, selected_ids, selection_records, *, ready_ns: Optional[int] = None) -> PairDemand:
        with self._lock:
            if self._demand is not None or self._error is not None:
                raise RuntimeError("pair latch is already terminal")
            demand = PairDemand(
                identity=self.identity,
                selected_ids=tuple(selected_ids),
                selection_records=tuple(selection_records),
                submitted_ns=self.submitted_ns,
                ready_ns=perf_counter_ns() if ready_ns is None else int(ready_ns),
            )
            demand.validate()
            self._demand = demand
            return demand

    def fail(self, error: BaseException | str) -> None:
        with self._lock:
            if self._demand is not None or self._error is not None:
                raise RuntimeError("pair latch is already terminal")
            self._error = repr(error) if isinstance(error, BaseException) else str(error)

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._demand is not None

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def demand(self, expected: PairIdentity) -> PairDemand:
        expected.validate()
        with self._lock:
            if expected != self.identity:
                raise ValueError("stale or mismatched pair identity")
            if self._error is not None:
                raise RuntimeError(f"pair selection failed: {self._error}")
            if self._demand is None:
                raise RuntimeError("pair demand is not ready")
            return self._demand

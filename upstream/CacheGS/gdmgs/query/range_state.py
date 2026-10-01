"""Exact range-state reconciliation, independent of Gaussian caching."""
from dataclasses import dataclass, field
import numpy as np


def validate_ranges(ranges, rank_count=None):
    if not isinstance(ranges, np.ndarray) or ranges.dtype != np.int64 or ranges.ndim != 2 or ranges.shape[1] != 2:
        raise TypeError("ranges must be an int64 Rx2 array")
    if len(ranges) and (np.any(ranges[:, 0] < 0) or np.any(ranges[:, 0] >= ranges[:, 1]) or
                        np.any(ranges[1:, 0] < ranges[:-1, 1])):
        raise ValueError("ranges must be ordered disjoint nonempty half-open intervals")
    if rank_count is not None and len(ranges) and ranges[-1, 1] > rank_count:
        raise ValueError("range exceeds DFS rank count")
    return ranges


def expand_ranges(ranges, dfs_to_row):
    validate_ranges(ranges, len(dfs_to_row))
    if not isinstance(dfs_to_row, np.ndarray) or dfs_to_row.dtype != np.int64 or dfs_to_row.ndim != 1:
        raise TypeError("dfs_to_row must be a one-dimensional int64 array")
    if not len(ranges):
        return np.empty(0, dtype=np.int64)
    return np.concatenate([dfs_to_row[first:last] for first, last in ranges])


@dataclass
class RangeState:
    """Reuse unchanged formal range storage only after complete fresh discovery.

    This is index-state reuse, with no Gaussian/decoder payload. It cannot serve
    as input candidate discovery and cannot remove a newly visible anchor.
    """
    index_token: str = ""
    _ranges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64), repr=False)
    reconciliations: int = 0
    unchanged_reuses: int = 0

    def reconcile(self, current_ranges, index_token):
        validate_ranges(current_ranges)
        if self.index_token and self.index_token != index_token:
            raise ValueError("Range state belongs to a different anchor index")
        self.index_token = index_token
        self.reconciliations += 1
        if np.array_equal(self._ranges, current_ranges):
            self.unchanged_reuses += 1
        else:
            self._ranges = current_ranges.copy()
            self._ranges.flags.writeable = False
        return self._ranges

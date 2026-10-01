"""Morton-packed anchor bounds with contiguous binary radix subtrees."""
from dataclasses import dataclass
import numpy as np

from .occluders import anchor_walk


def anchor_bounds(positions, offsets, offset_scales, support_radii):
    """Bounds of candidate centers expanded by a caller-certified radius.

    Radius certification depends on the decoder and rasterizer. A nominal
    three-sigma radius alone does not cover every screen-space raster filter.
    """
    p, o, s, r = [np.asarray(x, dtype=np.float64)
                  for x in (positions, offsets, offset_scales, support_radii)]
    if (p.ndim != 2 or p.shape[1] != 3 or o.ndim != 3 or
            o.shape[0] != len(p) or o.shape[2] != 3 or o.shape[1] < 1 or
            s.shape != p.shape or r.shape != (len(p),)):
        raise ValueError("expected positions/scales [N,3], offsets [N,O,3], radii [N]")
    if not all(np.isfinite(x).all() for x in (p, o, s, r)) or (r < 0).any():
        raise ValueError("finite geometry and nonnegative support radii required")
    centers = p[:, None] + o * s[:, None]
    return np.c_[centers.min(1) - r[:, None], centers.max(1) + r[:, None]]


def morton_codes(centers, lo, hi, bits=21):
    """Stable 63-bit keys; real-valued bounds remain the query keys."""
    extent = np.maximum(np.asarray(hi) - lo, np.finfo(float).eps)
    q = np.floor(np.clip((centers - lo) / extent, 0, 1) * ((1 << bits) - 1)).astype(np.uint64)
    code = np.zeros(len(q), dtype=np.uint64)
    for bit in range(bits):
        for axis in range(3):
            code |= ((q[:, axis] >> np.uint64(bit)) & np.uint64(1)) << np.uint64(3 * bit + 2 - axis)
    return code


@dataclass(frozen=True)
class AnchorIndex:
    bounds: np.ndarray
    nodes: np.ndarray
    left: np.ndarray
    right: np.ndarray
    order: np.ndarray
    intervals: np.ndarray
    domain_lo: np.ndarray
    domain_hi: np.ndarray

    @classmethod
    def build(cls, bounds, leaf_size=32, domain=None):
        b = np.array(bounds, dtype=np.float64, copy=True)
        if b.ndim != 2 or b.shape[1] != 6 or not np.isfinite(b).all() or (b[:, :3] > b[:, 3:]).any():
            raise ValueError("bounds must be finite ordered [N,6] boxes")
        if isinstance(leaf_size, bool) or not isinstance(leaf_size, int) or leaf_size < 1:
            raise ValueError("leaf_size must be a positive integer")
        if domain is None:
            lo, hi = (b[:, :3].min(0), b[:, 3:].max(0)) if len(b) else (np.zeros(3), np.ones(3))
            hi = np.maximum(hi, lo + 1e-12)
        else:
            lo, hi = (np.asarray(x, dtype=float) for x in domain)
            if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all()
                    or np.any(hi <= lo) or np.any(b[:, :3] < lo) or np.any(b[:, 3:] > hi)):
                raise ValueError("domain must contain all anchor bounds")
        keys = morton_codes((b[:, :3] + b[:, 3:]) / 2, lo, hi)
        order = np.argsort(keys, kind="stable").astype(np.int64)
        leaves = list(range(0, len(b), leaf_size))
        nodes, left, right, intervals = [], [], [], []

        def build(first, end):
            start_id = len(nodes)
            begin, stop = leaves[first], min(end * leaf_size, len(b))
            nodes.append(None); left.append(-1); right.append(-1)
            intervals.append((begin, stop))
            if end - first == 1:
                rows = b[order[begin:stop]]
                nodes[start_id] = np.r_[rows[:, :3].min(0), rows[:, 3:].max(0)]
            else:
                a, z = int(keys[order[leaves[first]]]), int(keys[order[leaves[end - 1]]])
                if a == z:
                    mid = (first + end) // 2
                else:
                    shift = (a ^ z).bit_length() - 1
                    low, high = first + 1, end
                    while low < high:
                        middle = (low + high) // 2
                        if (int(keys[order[leaves[middle]]]) >> shift) == (a >> shift):
                            low = middle + 1
                        else:
                            high = middle
                    mid = low
                l, r = build(first, mid), build(mid, end)
                left[start_id], right[start_id] = l, r
                nodes[start_id] = np.r_[np.minimum(nodes[l][:3], nodes[r][:3]),
                                       np.maximum(nodes[l][3:], nodes[r][3:])]
            return start_id

        if leaves:
            build(0, len(leaves))
        arrays = [b, np.asarray(nodes).reshape(-1, 6), np.asarray(left, np.int64),
                  np.asarray(right, np.int64), order, np.asarray(intervals, np.int64).reshape(-1, 2),
                  np.array(lo, copy=True), np.array(hi, copy=True)]
        for array in arrays:
            array.setflags(write=False)
        return cls(*arrays)

    def query(self, planes, cells=(), eye=None, eligible_ids=None):
        """Return increasing original IDs, independent of tree traversal order."""
        ids, diagnostics = anchor_walk(self.bounds, self.nodes, self.left, self.right,
                                       self.order, self.intervals, planes, cells, eye)
        ids.sort()
        if eligible_ids is not None:
            ids = np.intersect1d(ids, eligible_ids)
        return ids, diagnostics

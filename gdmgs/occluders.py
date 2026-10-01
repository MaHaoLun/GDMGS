"""Separate solid-cell octree and exact reference holed-frustum anchor walk.

This is a correctness implementation.  It makes no GPU speed claim; the
Amsterdam mesh has too few usable solid cells to justify a CUDA port yet.
"""
from dataclasses import dataclass, field

import numpy as np

from .geometry import box_plane_min, box_plane_max, hole_planes


@dataclass
class CellNode:
    full: bool = False
    children: dict = field(default_factory=dict)


class OccluderIndex:
    def __init__(self, domain_lo, domain_hi, level, full_nodes):
        self.lo = np.asarray(domain_lo, np.float64)
        self.hi = np.asarray(domain_hi, np.float64)
        self.level = level
        if (self.lo.shape != (3,) or self.hi.shape != (3,)
                or not np.isfinite([self.lo, self.hi]).all() or np.any(self.hi <= self.lo)
                or isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 30):
            raise ValueError('invalid octree domain or level')
        self.root = CellNode()
        for depth, x, y, z in full_nodes:
            if (not all(isinstance(v, (int, np.integer)) for v in (depth, x, y, z))
                    or not 0 <= depth <= level or any(v < 0 or v >= (1 << depth) for v in (x, y, z))):
                raise ValueError('cell outside the leaf grid')
            node = self.root
            for shift in range(depth - 1, -1, -1):
                child = (((x >> shift) & 1) << 2) | (((y >> shift) & 1) << 1) | ((z >> shift) & 1)
                if node.full:
                    raise ValueError('overlapping full cells')
                node = node.children.setdefault(child, CellNode())
            if node.children or node.full:
                raise ValueError('duplicate or overlapping full cell')
            node.full = True

    def cell_box(self, depth, xyz):
        size = (self.hi - self.lo) / (1 << depth)
        lower = self.lo + np.asarray(xyz) * size
        return np.r_[lower, lower + size]

    def query(self, planes, w2c, near, tolerance=1e-9):
        """Return coarsest visible full cells entirely past the near plane."""
        near_plane = np.r_[w2c[2, :3], w2c[2, 3] - near]
        selected = []

        def walk(node, depth, xyz):
            box = self.cell_box(depth, xyz)
            if any(box_plane_max(box, p) < -tolerance for p in planes):
                return
            if node.full and box_plane_min(box, near_plane) > tolerance:
                selected.append(box)
                return
            if depth == self.level:
                return
            children = range(8) if node.full else node.children.keys()
            for child in children:
                bits = ((child >> 2) & 1, (child >> 1) & 1, child & 1)
                nxt = CellNode(full=True) if node.full else node.children[child]
                walk(nxt, depth + 1, tuple(2 * xyz[a] + bits[a] for a in range(3)))

        walk(self.root, 0, (0, 0, 0))
        return np.asarray(selected, np.float64).reshape(-1, 6)


def frustum_class(box, planes, tolerance=1e-9):
    if any(box_plane_max(box, p) < -tolerance for p in planes):
        return 0
    if all(box_plane_min(box, p) > tolerance for p in planes):
        return 1
    return 2


def hole_class(box, planes, tolerance=1e-8):
    if all(box_plane_min(box, p) > tolerance for p in planes):
        return 1
    if any(box_plane_max(box, p) < -tolerance for p in planes):
        return 0
    return 2


def anchor_walk(anchor_bounds, node_bounds, left, right, order, leaf_start,
                frustum_planes, visible_cells, eye):
    """Cull / Keep / Descend; refine leaves against the remaining holes."""
    anchor_bounds = np.asarray(anchor_bounds)
    node_bounds = np.asarray(node_bounds)
    left = np.asarray(left)
    right = np.asarray(right)
    order = np.asarray(order)
    leaf_start = np.asarray(leaf_start)
    hole_sets = [hole_planes(cell, eye) for cell in visible_cells]
    result = []
    visits = 0
    hidden_nodes = 0

    def walk(node, active):
        nonlocal visits, hidden_nodes
        visits += 1
        box = node_bounds[node, :6]
        fstate = frustum_class(box, frustum_planes)
        if fstate == 0:
            return
        nearby = []
        for hp in active:
            state = hole_class(box, hp)
            if state == 1:
                hidden_nodes += 1
                return
            if state == 2:
                nearby.append(hp)
        if fstate == 1 and not nearby:
            # Source order here is the index's Morton order, as in the paper.
            result.extend(order[leaf_start[node, 0]:leaf_start[node, 1]])
            return
        if left[node] < 0:
            for original_id in order[leaf_start[node, 0]:leaf_start[node, 1]]:
                bounds = anchor_bounds[original_id, :6]
                if frustum_class(bounds, frustum_planes) == 0:
                    continue
                if any(hole_class(bounds, hp) == 1 for hp in nearby):
                    continue
                result.append(original_id)
            return
        walk(left[node], nearby)
        walk(right[node], nearby)

    if len(node_bounds):
        walk(0, hole_sets)
    return np.asarray(result, dtype=np.int64), dict(visited_nodes=visits, hidden_nodes=hidden_nodes,
                                                      holes=len(hole_sets))

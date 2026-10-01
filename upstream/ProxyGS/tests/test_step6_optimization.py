from __future__ import annotations

import numpy as np
import pytest

from gdmgs.anchor_index import AnchorPointIndex
from gdmgs.mesh_index import CameraDomain, MeshIndex


def test_optimized_mesh_matches_frozen_bvh_for_multiple_thread_counts():
    vertices = np.asarray(
        [
            [-1.0, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [0.0, 1.0, 2.0],
            [5.0, 5.0, 2.0],
            [6.0, 5.0, 2.0],
            [5.0, 6.0, 2.0],
            [0.0, 0.0, -2.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [3, 4, 5], [0, 2, 6]], dtype=np.int64)
    index = MeshIndex(vertices, triangles, method="binned_sah", leaf_size=1)
    domain = CameraDomain.parse(
        {
            "w2c": np.eye(4, dtype=np.float64),
            "angular_domain": (-1.0, 1.0, -1.0, 1.0),
            "near": 0.01,
            "far": 100.0,
            "camera_id": "unit",
        }
    )
    frozen = index.query(domain, backend="bvh")
    for threads in (1, 2, 4):
        optimized = index.query(domain, backend="optimized_bvh", threads=threads)
        assert np.array_equal(optimized.triangle_ids, frozen.triangle_ids)
        assert optimized.counters["optimized"] is True
        assert optimized.counters["output_ordering"] in {
            "original_row_bitmap_scan",
            "sorted_sparse_ids",
        }


def test_optimized_anchor_reuses_scratch_without_changing_outputs():
    rng = np.random.default_rng(20260915)
    positions = rng.uniform([-2.0, -2.0, -1.0], [2.0, 2.0, 5.0], size=(4097, 3)).astype(
        np.float32
    )
    index = AnchorPointIndex.build(
        positions,
        scene="unit",
        leaf_capacity=256,
        max_depth=16,
    )
    candidates = np.arange(len(positions), dtype=np.int64)
    depth = np.full((64, 64), 3.0, dtype=np.float32)
    view = np.eye(4, dtype=np.float32)
    projection = np.eye(4, dtype=np.float32)
    frozen = index.query(candidates, depth, view, projection, mode="tree", camera="unit")
    for _ in range(3):
        optimized = index.query(
            candidates,
            depth,
            view,
            projection,
            mode="tree",
            camera="unit",
            trusted_buffers=True,
            reuse_buffers=True,
        )
        assert np.array_equal(optimized.selected_anchor_ids, frozen.selected_anchor_ids)
        assert np.array_equal(optimized.formal_ranges, frozen.formal_ranges)
        assert optimized.timings["trusted_buffers"] == 1.0
        assert optimized.timings["reused_buffers"] == 1.0


def test_trusted_anchor_boundary_still_rejects_out_of_range_endpoints():
    positions = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    index = AnchorPointIndex.build(positions, scene="unit", leaf_capacity=1, max_depth=1)
    with pytest.raises(ValueError, match="trusted candidate endpoint"):
        index.query(
            np.asarray([1], dtype=np.int64),
            np.full((1, 1), 2.0, dtype=np.float32),
            np.eye(4, dtype=np.float32),
            np.eye(4, dtype=np.float32),
            mode="tree",
            trusted_buffers=True,
            reuse_buffers=True,
        )

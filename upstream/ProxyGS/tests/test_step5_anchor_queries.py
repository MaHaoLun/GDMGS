"""Contract tests for Step 5 reference/linear/tree anchor selection."""

from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anchor_query_runtime import run_anchor_queries
from gdmgs.anchor_index import AnchorPointIndex
from gdmgs.anchor_index.point_index import _native


def camera_matrices():
    return np.eye(4, dtype=np.float32).T.copy(), np.eye(4, dtype=np.float32).T.copy()


def perspective_matrices():
    view = np.eye(4, dtype=np.float32)
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = 1.0
    projection[1, 1] = 1.0
    projection[2, 2] = 1.0
    projection[2, 3] = 1.0
    return view, projection


def test_reference_linear_tree_and_source_order_parity():
    anchors = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [3.0, 0.0, 2.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    candidates = np.arange(4, dtype=np.int64)
    depth = np.full((4, 4), np.inf, dtype=np.float32)
    depth[2, 2] = 1.5
    view, projection = camera_matrices()
    index = AnchorPointIndex.build(anchors, scene="fixture", leaf_capacity=2, max_depth=8)
    result = run_anchor_queries(
        index=index,
        candidate_ids=candidates,
        anchor_positions=anchors,
        world_view_transform=view,
        full_proj_transform=projection,
        indexed_depth=depth,
        camera="fixture",
        tree_first=False,
    )
    np.testing.assert_array_equal(result["reference_ids"], [0, 2])
    assert all(result["parity"].values())


def test_finite_depth_cull_uses_exact_pointwise_fallback():
    grid = np.linspace(-0.01, 0.01, 18, dtype=np.float32)
    anchors = np.array(
        [(x, y, 3.0) for x in grid for y in grid], dtype=np.float32
    )
    depth = np.ones((8, 8), dtype=np.float32)
    candidates = np.arange(len(anchors), dtype=np.int64)
    view, projection = camera_matrices()
    index = AnchorPointIndex.build(anchors, scene="fixture", leaf_capacity=1, max_depth=8)
    result = index.query(
        candidates, depth, view, projection, mode="tree", camera="known-cull"
    )
    assert len(result.selected_anchor_ids) == 0
    assert result.counters["certified_nodes"] == 0
    assert result.counters["anchor_fallback_checks"] == len(anchors)


def test_unknown_depth_forces_fallback_without_false_cull():
    anchors = np.array([[0.0, 0.0, 3.0], [0.01, 0.01, 3.0]], dtype=np.float32)
    depth = np.full((8, 8), np.inf, dtype=np.float32)
    candidates = np.arange(len(anchors), dtype=np.int64)
    view, projection = camera_matrices()
    index = AnchorPointIndex.build(anchors, scene="fixture", leaf_capacity=8, max_depth=8)
    result = index.query(candidates, depth, view, projection, mode="tree")
    np.testing.assert_array_equal(result.selected_anchor_ids, candidates)
    assert result.counters["certified_nodes"] == 0
    assert result.counters["anchor_fallback_checks"] == len(candidates)


def test_exact_depth_range_max_for_non_power_of_two_shapes():
    values = np.arange(35, dtype=np.float32).reshape(5, 7) + 1
    values[3, 4] = np.inf
    native = _native()
    for row0, column0, row1, column1 in (
        (0, 0, 0, 0),
        (0, 0, 4, 6),
        (1, 2, 2, 5),
        (3, 4, 3, 4),
        (2, 0, 4, 3),
    ):
        actual = native.depth_range_max(values, row0, column0, row1, column1)
        selected = values[row0 : row1 + 1, column0 : column1 + 1]
        assert actual["maximum"] == np.max(selected)
        assert actual["minimum"] == np.min(selected)


def test_depth_min_keep_ablation_is_disabled_without_changing_results():
    grid = np.linspace(-0.02, 0.02, 20, dtype=np.float32)
    anchors = np.array([(x, y, 1.0) for x in grid for y in grid], dtype=np.float32)
    depth = np.full((16, 16), 4.0, dtype=np.float32)
    candidates = np.arange(len(anchors), dtype=np.int64)
    view, projection = camera_matrices()
    index = AnchorPointIndex.build(anchors, scene="front", leaf_capacity=32, max_depth=16)
    result = index.query(candidates, depth, view, projection, mode="tree")
    np.testing.assert_array_equal(result.selected_anchor_ids, candidates)
    assert result.counters["terminal_depth_keep_nodes"] == 0


def test_tree_can_terminal_cull_nonpositive_camera_z():
    grid = np.linspace(-0.02, 0.02, 20, dtype=np.float32)
    anchors = np.array([(x, y, -1.0) for x in grid for y in grid], dtype=np.float32)
    depth = np.ones((16, 16), dtype=np.float32)
    candidates = np.arange(len(anchors), dtype=np.int64)
    view, projection = camera_matrices()
    index = AnchorPointIndex.build(anchors, scene="behind", leaf_capacity=32, max_depth=16)
    result = index.query(candidates, depth, view, projection, mode="tree")
    assert len(result.selected_anchor_ids) == 0
    assert result.counters["terminal_nonpositive_cull_nodes"] >= 1


def test_save_load_binding_and_topology(tmp_path):
    anchors = np.linspace(-1, 1, 99, dtype=np.float32).reshape(33, 3)
    source = {"path": "/fixture.ply", "bytes": 123, "mtime_ns": 456, "iteration": 40000}
    built = AnchorPointIndex.build(
        anchors,
        scene="fixture",
        leaf_capacity=4,
        max_depth=12,
        source=source,
    )
    path = tmp_path / "anchor_point_bvh.npz"
    metadata = built.save(path)
    loaded = AnchorPointIndex.load(path, scene="fixture", source=source)
    np.testing.assert_array_equal(loaded.positions, anchors)
    np.testing.assert_array_equal(loaded.dfs_to_row, built.dfs_to_row)
    np.testing.assert_array_equal(loaded.rank_of_row, built.rank_of_row)
    assert metadata["anchor_count"] == len(anchors)
    with pytest.raises(ValueError):
        AnchorPointIndex.load(path, scene="other")


def test_rejects_duplicate_candidates_and_bad_depth():
    anchors = np.ones((3, 3), dtype=np.float32)
    index = AnchorPointIndex.build(anchors, scene="fixture")
    depth = np.ones((2, 2), dtype=np.float32)
    view, projection = camera_matrices()
    with pytest.raises((ValueError, RuntimeError)):
        index.query(np.array([0, 0], dtype=np.int64), depth, view, projection, mode="linear")
    depth[0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError)):
        index.query(np.array([0], dtype=np.int64), depth, view, projection, mode="linear")


@pytest.mark.parametrize("seed", range(12))
def test_random_reference_linear_tree_property(seed):
    random = np.random.default_rng(seed)
    anchors = random.uniform([-8.0, -6.0, -2.0], [8.0, 6.0, 15.0], size=(4096, 3)).astype(
        np.float32
    )
    chosen = np.sort(random.choice(len(anchors), size=3072, replace=False)).astype(np.int64)
    depth = random.uniform(0.05, 12.0, size=(73, 117)).astype(np.float32)
    depth[random.random(depth.shape) < 0.13] = np.inf
    view, projection = perspective_matrices()
    index = AnchorPointIndex.build(
        anchors, scene=f"random-{seed}", leaf_capacity=16, max_depth=24
    )
    result = run_anchor_queries(
        index=index,
        candidate_ids=chosen,
        anchor_positions=anchors,
        world_view_transform=view,
        full_proj_transform=projection,
        indexed_depth=depth,
        camera=f"random-{seed}",
        tree_first=bool(seed % 2),
    )
    assert all(result["parity"].values())

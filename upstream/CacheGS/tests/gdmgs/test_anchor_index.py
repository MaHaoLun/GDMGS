"""Anchor native binding, conservative support and materialization contracts."""
import os
from pathlib import Path
import numpy as np
import pytest

from gdmgs.unified_index import AnchorIndex, SupportSettings, support_bounds
from gdmgs.query.range_state import RangeState, expand_ranges


def build(points, scales=None, offsets=None, leaf=2):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    return AnchorIndex.from_arrays(points, np.zeros((n, 1, 3)) if offsets is None else offsets,
                                   np.full((n, 6), .01) if scales is None else scales,
                                   leaf_capacity=leaf, scene_token="anchor-fixture")


def query(index, ids=None, depth=None, mode="tree", **kwargs):
    ids = np.arange(index.anchor_count, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
    depth = np.full((32, 32), 2.0) if depth is None else depth
    return index.query(ids, depth, np.eye(4), (-1., 1., -1., 1.), (1024, 1024), mode=mode, **kwargs)


def assert_ranges(index, result):
    raw = expand_ranges(result.raw_ranges, index.dfs_to_row)
    formal = expand_ranges(result.formal_ranges, index.dfs_to_row)
    assert np.array_equal(raw, formal)
    assert np.array_equal(np.sort(raw), np.sort(result.selected_anchor_ids))


def test_duplicate_positions_root_boundaries_and_binding_bijection():
    points = [[0, 0, 0], [0, 0, 0], [-1, -1, -1], [1, 1, 1], [-1, 1, -1], [1, -1, 1]]
    index = build(points, leaf=1)
    layout = index.layout()
    assert np.array_equal(np.sort(index.dfs_to_row), np.arange(6))
    assert np.array_equal(index.rank_of_row[index.dfs_to_row], np.arange(6))
    for node, (start, end) in enumerate(layout["intervals"]):
        descendants = index.dfs_to_row[start:end]
        coordinates = np.asarray(points)[descendants]
        assert np.all(coordinates >= layout["partition_bounds"][node, :3])
        assert np.all(coordinates <= layout["partition_bounds"][node, 3:])
        children = layout["children"][node]
        children = children[children >= 0]
        if len(children):
            assert layout["intervals"][children[0], 0] == start
            assert layout["intervals"][children[-1], 1] == end
            assert np.array_equal(layout["intervals"][children[:-1], 1], layout["intervals"][children[1:], 0])


def test_known_wall_effective_removal_preserves_original_fov_order():
    index = build([[0, 0, 5], [.2, 0, 5], [.1, 0, 1], [0, .1, 1]], leaf=1)
    result = query(index, [2, 1, 3, 0])
    assert result.selected_anchor_ids.tolist() == [2, 3]
    assert result.counters["certified_nodes"] > 0
    assert result.counters["certified_anchors"] == 2
    assert_ranges(index, result)


def test_support_includes_far_offset_and_crosses_partition_cell():
    points = np.array([[0., 0, 5], [.1, 0, 5]])
    offsets = np.zeros((2, 2, 3)); offsets[0, 1, 2] = -4.5
    scales = np.full((2, 6), .01); scales[:, :3] = 1
    index = build(points, scales, offsets, leaf=1)
    assert query(index).selected_anchor_ids.tolist() == [0]
    layout = index.layout()
    assert layout["support_bounds"][0, 2] < 1
    assert layout["partition_bounds"][0, 2] == 5
    assert layout["anchor_support_bounds"][0, 2] < .5


def test_fp16_upper_rounding_and_all_rotations_bound():
    scales = np.array([[1, 1, 1, .9998, .1, .2]], dtype=np.float64)
    bound = support_bounds(np.zeros((1, 3)), np.zeros((1, 1, 3)), scales)
    assert bound[0, 3] > 3.5 * np.float16(.9998).astype(np.float64)
    # A 90-degree rotation can place the longest scale on any axis.
    assert np.all(bound[0, :3] <= -3.5)
    assert np.all(bound[0, 3:] >= 3.5)


def test_unknown_scales_overflow_and_near_plane_keep():
    scales = np.full((3, 6), .01); scales[0, 3] = np.inf; scales[1, 4] = 70000
    index = build([[0, 0, 5], [.1, 0, 5], [0, 0, .01]], scales)
    result = query(index)
    assert result.selected_anchor_ids.tolist() == [0, 1, 2]
    assert index.unbounded_support_count == 2
    assert result.counters["unbounded"] > 0 and result.counters["near_plane"] > 0


def test_unknown_ori_hole_and_screen_blur_keep():
    index = build([[0., 0, 5]])
    complete = np.full((32, 32), 2.)
    assert len(query(index, depth=complete).selected_anchor_ids) == 0
    # The small Gaussian center is in one cell, but the complete expanded
    # pixel footprint reaches its neighbor. A hole there forbids a certificate.
    complete[15, 15] = np.inf
    assert query(index, depth=complete).selected_anchor_ids.tolist() == [0]


def test_randomized_tree_removals_subset_linear_and_ranges_exact():
    rng = np.random.default_rng(192)
    points = rng.uniform([-4, -4, .1], [4, 4, 12], (600, 3))
    scales = rng.uniform(.001, .1, (600, 6))
    offsets = rng.normal(size=(600, 4, 3))
    index = build(points, scales, offsets, leaf=8)
    for trial in range(10):
        ids = rng.permutation(600)[:400]
        depth = rng.uniform(1., 3., (32, 32))
        depth[rng.random((32, 32)) < .02] = np.inf
        tree = query(index, ids, depth, "tree")
        linear = query(index, ids, depth, "linear")
        assert set(ids) - set(tree.selected_anchor_ids) <= set(ids) - set(linear.selected_anchor_ids)
        assert set(tree.selected_anchor_ids) <= set(ids)
        assert tree.selected_anchor_ids.tolist() == [row for row in ids if row in set(tree.selected_anchor_ids)]
        assert_ranges(index, tree)
        assert_ranges(index, linear)


def test_empty_all_fov_raw_and_formal_denotation():
    index = build([[0, 0, 1], [.1, 0, 1], [.2, 0, 1], [.3, 0, 1]], leaf=1)
    empty = query(index, [])
    assert empty.selected_anchor_ids.shape == (0,)
    assert empty.raw_ranges.shape == empty.formal_ranges.shape == (0, 2)
    all_ids = query(index, [3, 1, 0, 2], np.full((32, 32), np.inf))
    assert all_ids.selected_anchor_ids.tolist() == [3, 1, 0, 2]
    assert all_ids.counters["raw_range_count"] > all_ids.counters["formal_range_count"]
    assert all_ids.formal_ranges.tolist() == [[0, 4]]
    assert_ranges(index, empty); assert_ranges(index, all_ids)
    empty_tree = build([])
    assert query(empty_tree).selected_anchor_ids.size == 0


def test_ranges_reuse_never_restricts_new_candidate_discovery():
    index = build([[0, 0, 1], [.1, 0, 1], [.2, 0, 1]])
    state = RangeState()
    first = query(index, [0], range_state=state)
    second = query(index, [2, 1, 0], range_state=state)
    fresh = query(index, [2, 1, 0])
    assert first.selected_anchor_ids.tolist() == [0]
    assert second.selected_anchor_ids.tolist() == [2, 1, 0]
    assert np.array_equal(second.formal_ranges, fresh.formal_ranges)
    query(index, [2, 1, 0], range_state=state)
    assert state.unchanged_reuses == 1
    with pytest.raises(ValueError, match="different anchor index"):
        state.reconcile(fresh.formal_ranges, "other")


def test_saved_topology_roundtrip_and_tampering_rejected(tmp_path):
    index = build([[0, 0, 5], [.2, 0, 5], [0, 0, 1]], leaf=1)
    destination = tmp_path / "anchor_index.npz"
    index.save(destination)
    restored = AnchorIndex.load(destination, scene_token=index.scene_token)
    for key, array in index.layout().items():
        assert np.array_equal(array, restored.layout()[key])
    assert np.array_equal(query(index).selected_anchor_ids, query(restored).selected_anchor_ids)
    assert restored.load_seconds >= 0
    with pytest.raises(ValueError, match="another finalized scene"):
        AnchorIndex.load(destination, scene_token="wrong")
    with np.load(destination) as saved:
        contents = {key: saved[key] for key in saved.files}
    contents["dfs_to_row"][1] = contents["dfs_to_row"][0]
    np.savez(destination, **contents)
    with pytest.raises(ValueError, match="bijective"):
        AnchorIndex.load(destination)


@pytest.mark.parametrize("ids", [np.array([-1], dtype=np.int64), np.array([5], dtype=np.int64), np.array([0, 0], dtype=np.int64)])
def test_invalid_ids_rejected(ids):
    index = build([[0, 0, 5]])
    with pytest.raises(ValueError):
        query(index, ids)


def test_typed_camera_boundary_rejects_invalid_buffers():
    index = build([[0, 0, 5]])
    with pytest.raises(TypeError):
        index.query(np.array([0], dtype=np.int32), np.ones((2, 2)), np.eye(4), (-1, 1, -1, 1), (100, 100))
    for depth in (np.array([[np.nan]]), np.array([[-np.inf]]), np.array([[0.]])):
        with pytest.raises(ValueError):
            query(index, depth=depth)
    matrix = np.eye(4); matrix[3, 0] = 1
    with pytest.raises(ValueError, match="world-to-camera"):
        index.query(np.array([0], dtype=np.int64), np.ones((2, 2)), matrix, (-1, 1, -1, 1), (100, 100))


def test_linearized_covariance_footprint_exceeds_perspective_world_box():
    # gsplat uses a linearized covariance. With a large off-axis splat, an
    # actual alpha>=1/255 sample can lie outside the perspective world box.
    # x/z=1, sigma=1.5, z=10: x radius is 3.329*1.5*sqrt(2)/10=.706.
    # Thus x/z=.30 is inside the ellipse even though a projected 3.5-sigma
    # world box begins near .311. Its ORI hole must prevent deletion.
    scales = np.array([[1., 1, 1, 1.5, 1.5, 1.5]])
    index = build([[10., 0, 10]], scales)
    depth = np.ones((256, 1024))
    args = (np.array([0], dtype=np.int64), depth, np.eye(4), (-2., 2., -2., 2.), (1000000, 1000000))
    assert index.query(*args).selected_anchor_ids.size == 0
    depth[128, int((.30 + 2) / 4 * 1024)] = np.inf
    assert index.query(*args).selected_anchor_ids.tolist() == [0]


def test_persisted_source_binding_survives_session_change_but_rejects_checkpoint_change(tmp_path):
    from types import SimpleNamespace
    import torch
    ply = tmp_path / "point_cloud.ply"
    ply.write_text("fixture rows only")
    scene = SimpleNamespace(_rows={"_anchor": torch.tensor([[0., 0., 5.]]),
                                  "_offset": torch.zeros(1, 1, 3),
                                  "_scaling": torch.full((1, 6), -4.)},
                            token="first-session", anchor_count=1, iteration=30000,
                            checkpoint_path=str(tmp_path), _loaded_ply_path=str(ply))
    index = AnchorIndex.from_finalized(scene)
    path = tmp_path / "bound_index.npz"
    index.save(path)
    scene.token = "next-session"
    loaded = AnchorIndex.load(path, finalized_scene=scene)
    assert loaded.scene_token == "next-session"
    assert loaded.index_token == index.index_token
    ply.write_text("changed fixture row payload")
    with pytest.raises(ValueError, match="checkpoint provenance"):
        AnchorIndex.load(path, finalized_scene=scene)


def test_alpha_support_profile_removes_only_redundant_tile_evaluation_padding():
    import math
    settings = SupportSettings()
    legacy = SupportSettings(profile="gsplat-1.4-pinhole-classic-fp32-fp16-batch")
    assert settings.profile.endswith("alpha-support-v3")
    assert math.isclose(settings.pixel_pad, 3.5 * math.sqrt(.4) + 1)
    assert math.isclose(legacy.pixel_pad - settings.pixel_pad, 16)
    # Raster tiles can evaluate samples outside the ellipse, but these never
    # survive the opacity<=1 / alpha>=1/255 rule. A tile is not added support.
    distances = np.linspace(settings.sigma, settings.sigma + 16, 1000)
    assert np.all(np.exp(-.5 * distances ** 2) < 1 / 255)
    positions = np.array([[0., 0, 5.]])
    offsets, scales = np.zeros((1, 1, 3)), np.full((1, 6), .01)
    np.testing.assert_array_equal(support_bounds(positions, offsets, scales, settings),
                                  support_bounds(positions, offsets, scales, legacy))


def test_native_anchor_proxy_is_explicit_and_default_stays_full_decoder_support():
    proxy = SupportSettings(profile="native_anchor_proxy_v1")
    assert proxy.definition == "native_fov_anchor_first3_scale_image_space_proxy"
    assert proxy.scale_rounding == "float32-outward"
    assert not SupportSettings().is_anchor_proxy
    assert SupportSettings().profile == "gsplat-1.4-pinhole-classic-alpha-support-v3"
    positions = np.array([[0., 0, 5.]])
    offsets = np.array([[[0., 0, -150.]]])
    scales = np.array([[.01, .02, .03, .5, .5, .5]])
    full = AnchorIndex.from_arrays(positions, offsets, scales)
    index = AnchorIndex.from_arrays(positions, offsets, scales, support_settings=proxy)
    # The original anchor proxy is behind the wall; its decoded center is in
    # front. Different decisions are intentional and require the image gate.
    assert query(full).selected_anchor_ids.tolist() == [0]
    result = query(index)
    assert result.selected_anchor_ids.tolist() == []
    assert result.support_profile == proxy.profile
    assert result.support_definition == proxy.definition
    centers = index.layout()["anchor_center_bounds"]
    assert np.all(centers[:, :3] <= positions) and np.all(centers[:, 3:] >= positions)
    assert centers[0, 2] > 4.999
    assert index.layout()["anchor_radii"][0] > 3.5 * np.float32(.03)
    assert index.layout()["anchor_radii"][0] < .106


def test_native_anchor_proxy_bounds_fov_fp32_scales_not_decoder_fp16_or_offsets():
    proxy = SupportSettings(profile="native_anchor_proxy_v1")
    positions = np.array([[0., 0, 1e6]])
    offsets = np.full((1, 2, 3), np.nan)
    scales = np.array([[70000., 1., 1., np.inf, np.inf, np.inf]])
    index = AnchorIndex.from_arrays(positions, offsets, scales, support_settings=proxy)
    assert index.unbounded_support_count == 0
    bound = index.layout()["anchor_support_bounds"][0]
    assert np.isfinite(bound).all()
    assert bound[0] < -245000 and bound[3] > 245000
    # Unknown *used* first-three scales still mean Keep.
    scales[0, 0] = np.inf
    unbounded = AnchorIndex.from_arrays(positions, offsets, scales, support_settings=proxy)
    assert unbounded.unbounded_support_count == 1
    assert query(unbounded).selected_anchor_ids.tolist() == [0]


def test_native_proxy_profile_definition_and_persistence_cannot_be_mixed(tmp_path):
    import json
    proxy = SupportSettings(profile="native_anchor_proxy_v1")
    with pytest.raises(ValueError, match="disagree"):
        SupportSettings(profile=proxy.profile, definition=SupportSettings().definition)
    with pytest.raises(ValueError, match="disagree"):
        SupportSettings(profile=proxy.profile, scale_rounding="float16-outward")
    index = AnchorIndex.from_arrays(np.array([[0., 0, 5.]]), np.zeros((1, 1, 3)),
                                    np.full((1, 6), .01), support_settings=proxy)
    path = tmp_path / "proxy.npz"
    index.save(path)
    restored = AnchorIndex.load(path, support_profile=proxy.profile)
    assert restored.support_settings == proxy
    np.testing.assert_array_equal(query(index).selected_anchor_ids, query(restored).selected_anchor_ids)
    with pytest.raises(ValueError, match="differs from the requested"):
        AnchorIndex.load(path, support_profile=SupportSettings().profile)
    with np.load(path) as saved:
        contents = {key: saved[key] for key in saved.files}
    metadata = json.loads(str(contents["metadata"]))
    assert metadata["format_version"] == 4
    assert metadata["support_settings"]["definition"] == proxy.definition
    metadata["support_settings"]["definition"] = SupportSettings().definition
    contents["metadata"] = np.asarray(json.dumps(metadata))
    np.savez(path, **contents)
    with pytest.raises(ValueError, match="disagree"):
        AnchorIndex.load(path)


def test_native_proxy_randomized_linear_tree_same_profile_and_ranges():
    rng = np.random.default_rng(194)
    positions = rng.uniform([-3, -3, .1], [3, 3, 10], (500, 3))
    scales = rng.uniform(.001, .1, (500, 6))
    index = AnchorIndex.from_arrays(positions, rng.normal(size=(500, 3, 3)), scales,
                                    support_settings=SupportSettings(profile="native_anchor_proxy_v1"),
                                    leaf_capacity=8)
    for _ in range(5):
        ids = rng.permutation(500)[:300]
        depths = rng.uniform(1, 3, (32, 32))
        depths[rng.random((32, 32)) < .01] = np.inf
        linear = query(index, ids, depths, "linear")
        tree = query(index, ids, depths, "tree")
        assert linear.support_profile == tree.support_profile == "native_anchor_proxy_v1"
        assert set(ids) - set(tree.selected_anchor_ids) <= set(ids) - set(linear.selected_anchor_ids)
        assert_ranges(index, tree)
        assert_ranges(index, linear)


def test_relabeling_existing_geometry_as_another_support_definition_is_rejected(tmp_path):
    index = build([[0., 0, 5.]])
    index.support_settings = SupportSettings(profile="native_anchor_proxy_v1")
    with pytest.raises(ValueError, match="rebuild anchor geometry"):
        query(index)
    with pytest.raises(ValueError, match="rebuild anchor geometry"):
        index.save(tmp_path / "mislabeled.npz")

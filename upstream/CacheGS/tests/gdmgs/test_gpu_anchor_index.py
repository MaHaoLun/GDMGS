"""GPU sparse-max and anchor certificates against CPU/analytic oracles."""
import numpy as np
import pytest
import torch

from gdmgs.unified_index import AnchorIndex, SupportSettings
from gdmgs.unified_index.gpu_index import GPUAnchorIndex, _native
from gdmgs.query.range_state import expand_ranges

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")


def make_index(points, *, profile=None, offsets=None, scales=None, leaf=8):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(p)
    settings = SupportSettings() if profile is None else SupportSettings(profile=profile)
    return AnchorIndex.from_arrays(p, np.zeros((n, 1, 3)) if offsets is None else offsets,
        np.full((n, 6), .01) if scales is None else scales, support_settings=settings, leaf_capacity=leaf)


def paired(cpu, depth, ids, mode, dtype=torch.float64, matrix=None):
    gpu = GPUAnchorIndex.from_cpu(cpu)
    matrix = np.eye(4) if matrix is None else matrix
    gpu_depth = torch.tensor(depth, dtype=dtype, device="cuda")
    exact_depth = gpu_depth.cpu().numpy().astype(np.float64)
    cpu_result = cpu.query(ids, exact_depth, matrix, (-1., 1., -1., 1.), (1024, 1024), mode=mode)
    result = gpu.query(torch.tensor(ids, device="cuda"), gpu_depth, matrix,
                       (-1., 1., -1., 1.), (1024, 1024), mode=mode,
                       camera_token="fixture-camera", mesh_token="fixture-mesh")
    torch.cuda.synchronize()
    selected = result.selected_anchor_ids.cpu().numpy()
    np.testing.assert_array_equal(selected, cpu_result.selected_anchor_ids)
    raw, formal = result.raw_ranges.cpu().numpy(), result.formal_ranges.cpu().numpy()
    np.testing.assert_array_equal(expand_ranges(raw, cpu.dfs_to_row), expand_ranges(formal, cpu.dfs_to_row))
    np.testing.assert_array_equal(np.sort(expand_ranges(formal, cpu.dfs_to_row)), np.sort(selected))
    counts = result.collect_counters()
    assert counts["fov_count"] == len(ids)
    assert counts["selected_count"] == len(selected)
    assert result.timings["gpu_total_ms"] >= 0
    assert result.camera_token == "fixture-camera" and result.mesh_token == "fixture-mesh"
    assert result.support_definition == cpu.support_settings.definition
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sparse_max_exact_all_dyadic_blocks_and_all_rectangles(dtype):
    base = np.arange(77, dtype=np.float64).reshape(7, 11) + 1
    base[3, 4] = np.inf
    table = _native().sparse_max(torch.tensor(base, dtype=dtype, device="cuda")).cpu().numpy()
    for ky in range(table.shape[0]):
        for kx in range(table.shape[1]):
            hh, ww = 1 << ky, 1 << kx
            for y in range(7):
                for x in range(11):
                    expected = np.max(base[y:y+hh, x:x+ww]) if y+hh <= 7 and x+ww <= 11 else np.inf
                    assert table[ky, kx, y, x] == expected
    for y0 in range(7):
        for y1 in range(y0, 7):
            for x0 in range(11):
                for x1 in range(x0, 11):
                    ky, kx = (y1-y0+1).bit_length()-1, (x1-x0+1).bit_length()-1
                    by, bx = y1-(1 << ky)+1, x1-(1 << kx)+1
                    actual = max(table[ky, kx, y0, x0], table[ky, kx, y0, bx],
                                 table[ky, kx, by, x0], table[ky, kx, by, bx])
                    assert actual == np.max(base[y0:y1+1, x0:x1+1])


@pytest.mark.parametrize("mode", ["linear", "tree"])
@pytest.mark.parametrize("profile", [None, "native_anchor_proxy_v1"])
def test_wall_actual_removal_front_offsets_unknown_and_order(mode, profile):
    points = [[0, 0, 5], [.1, 0, 5], [0, .2, 1], [0, 0, .005], [.2, 0, 5]]
    offsets = np.zeros((5, 2, 3)); offsets[4, 1, 2] = -450
    cpu = make_index(points, profile=profile, offsets=offsets, leaf=1)
    result = paired(cpu, np.full((32, 32), 2.), np.array([4, 0, 3, 1, 2], dtype=np.int64), mode)
    assert result.collect_counters()["certified_anchors"] >= 2
    unknown = paired(cpu, np.full((32, 32), np.inf), np.array([4, 0, 3, 1, 2], dtype=np.int64), mode)
    assert unknown.selected_anchor_ids.tolist() == [4, 0, 3, 1, 2]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_randomized_rotated_camera_gpu_matches_cpu_and_tree_matches_linear(dtype):
    rng = np.random.default_rng(414)
    p = rng.uniform([-3, -3, .03], [3, 3, 15], (800, 3))
    cpu = make_index(p, offsets=rng.normal(size=(800, 3, 3)), scales=rng.uniform(.001, .05, (800, 6)))
    angle = .2
    matrix = np.array([[np.cos(angle), 0, np.sin(angle), .1], [0, 1, 0, -.05],
                       [-np.sin(angle), 0, np.cos(angle), .2], [0, 0, 0, 1]], dtype=np.float64)
    ids = rng.permutation(800)[:600].astype(np.int64)
    for holes in (False, True):
        depth = rng.uniform(1, 3, (31, 43))
        if holes:
            depth[rng.random(depth.shape) < .01] = np.inf
        a = paired(cpu, depth, ids, "linear", dtype, matrix)
        b = paired(cpu, depth, ids, "tree", dtype, matrix)
        assert set(ids) - set(b.selected_anchor_ids.tolist()) <= set(ids) - set(a.selected_anchor_ids.tolist())


def test_maximal_node_interval_does_not_count_descendant_anchors_twice():
    points = [[x / 100., y / 100., 5.] for y in range(12) for x in range(12)]
    cpu = make_index(points, leaf=2)
    result = paired(cpu, np.full((32, 32), 2.), np.arange(len(points), dtype=np.int64), "tree")
    counts = result.collect_counters()
    assert counts["visited_nodes"] == len(cpu.intervals)
    assert counts["maximal_certified_nodes"] == 1
    assert counts["certified_nodes"] > 1
    assert counts["anchor_checks"] == 0
    assert counts["certified_anchors"] == len(points)


def test_empty_index_and_fov_and_readonly_binding_guard():
    empty = make_index([])
    for mode in ("linear", "tree"):
        result = paired(empty, np.full((3, 5), 2.), np.empty(0, dtype=np.int64), mode)
        assert result.raw_ranges.shape == (0, 2)
    cpu = make_index([[0, 0, 5]])
    paired(cpu, np.full((3, 5), 2.), np.empty(0, dtype=np.int64), "tree")
    gpu = GPUAnchorIndex.from_cpu(cpu)
    gpu._tensors["anchor_radii"].add_(1)
    with pytest.raises(RuntimeError, match="modified"):
        gpu.query(torch.tensor([0], device="cuda"), torch.ones((3, 5), device="cuda"), np.eye(4), (-1, 1, -1, 1), (100, 100))


def test_offaxis_jacobian_hole_fp16_overflow_and_dtype_rejection():
    cpu = make_index([[10., 0, 10]], scales=np.array([[1., 1, 1, 1.5, 1.5, 1.5]]))
    gpu = GPUAnchorIndex.from_cpu(cpu)
    depth = torch.ones((256, 1024), dtype=torch.float64, device="cuda")
    ids = torch.tensor([0], dtype=torch.int64, device="cuda")
    first = gpu.query(ids, depth, np.eye(4), (-2., 2., -2., 2.), (1000000, 1000000))
    assert first.selected_anchor_ids.numel() == 0
    depth[128, int((.30+2)/4*1024)] = torch.inf
    assert gpu.query(ids, depth, np.eye(4), (-2., 2., -2., 2.), (1000000, 1000000)).selected_anchor_ids.tolist() == [0]
    with pytest.raises(TypeError, match="int64"):
        gpu.query(ids.int(), depth, np.eye(4), (-2., 2., -2., 2.), (1000000, 1000000))
    scales = np.full((1, 6), .01); scales[0, 3] = 70000
    paired(make_index([[0, 0, 5]], scales=scales), np.full((4, 4), 2.), np.array([0], dtype=np.int64), "tree")


def test_gpu_binding_replacement_or_key_change_is_rejected_before_query():
    cpu = make_index([[0, 0, 5], [.1, 0, 5]])
    for mutation in ("replacement", "missing_key", "storage_rebind"):
        gpu = GPUAnchorIndex.from_cpu(cpu)
        if mutation == "replacement":
            gpu._tensors["rank_of_row"] = gpu._tensors["rank_of_row"].flip(0)
        elif mutation == "missing_key":
            del gpu._tensors["rank_of_row"]
        else:
            gpu._tensors["rank_of_row"].data = gpu._tensors["rank_of_row"].clone()
        with pytest.raises(RuntimeError, match="modified"):
            gpu.query(torch.tensor([0], device="cuda"), torch.ones((3, 5), device="cuda"),
                      np.eye(4), (-1, 1, -1, 1), (100, 100))

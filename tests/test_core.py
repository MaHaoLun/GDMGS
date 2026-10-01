import itertools
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gdmgs.camera import Camera, source_camera
from gdmgs.geometry import prune_bounds
from gdmgs.index import AnchorIndex, anchor_bounds
from gdmgs.materialization import GroupMaterializer, union_plan
from gdmgs.occluders import OccluderIndex, frustum_class
from gdmgs.occupancy import make_grid
from gdmgs.schedule import Schedule, assignment, cpu_quota
from gdmgs.selection import CPUSelector
from gdmgs.tensor_selection import TensorSelector
from gdmgs.adapters.proxygs import ProxyGSDecoder


def camera(x=0):
    w = np.eye(4); w[0, 3] = -x
    return Camera(w, 64, 64, 32., 32., .1, 20.)


class Model:
    """Small analytic decoder fixture, never a quality/performance benchmark."""
    use_feat_bank = add_level = appearance_dim = add_opacity_dist = add_color_dist = add_cov_dist = False
    dist2level = 'round'
    n_offsets = 2

    def __init__(self):
        self.get_anchor = torch.tensor([[0., 0., 3.], [1., 0., 3.], [0., 1., 3.]])
        self.get_anchor_feat = torch.tensor([[1.], [-1.], [2.]])
        self.get_level = torch.zeros(3, 1)
        self._offset = torch.zeros(3, 2, 3)
        self.get_scaling = torch.ones(3, 6) * .1
        self.calls = 0

    def get_opacity_mlp(self, x):
        self.calls += 1
        return torch.cat((x[:, :1], -x[:, :1]), dim=1)

    def get_color_mlp(self, x):
        return torch.sigmoid(x[:, :1]).repeat(1, 6)

    def get_cov_mlp(self, x):
        return torch.ones(len(x), 14)

    def rotation_activation(self, x):
        return torch.nn.functional.normalize(x, dim=1)


def test_camera_translation_and_midpoint():
    a, b = camera(2), camera(4)
    np.testing.assert_allclose(a.center, [2, 0, 0])
    np.testing.assert_allclose(source_camera([a, b]).center, [3, 0, 0])
    assert source_camera([a]) is a


@pytest.mark.parametrize('leaf_size', [1, 4, 32])
def test_radix_walk_matches_independent_dense_predicate(leaf_size):
    rng = np.random.default_rng(71)
    lo = rng.uniform([-5, -5, .2], [5, 5, 12], (120, 3))
    bounds = np.c_[lo, lo + rng.uniform(.01, .5, (120, 3))]
    domain = ([-8, -8, -1], [16, 16, 16])
    index = AnchorIndex.build(bounds, leaf_size, domain)
    occ = OccluderIndex(*domain, 3, [(3, 2, 2, 1)])
    for c in [camera(), camera(-2), camera(3)]:
        cells = occ.query(c.planes, c.w2c, c.near)
        expected = np.flatnonzero(np.array([frustum_class(b, c.planes) != 0 for b in bounds])
                                  & ~prune_bounds(bounds, cells, c.center))
        np.testing.assert_array_equal(CPUSelector(index, occ)(c), expected)
        np.testing.assert_array_equal(TensorSelector(index, occ, 'cpu')(c).numpy(), expected)


def test_duplicate_morton_codes_empty_and_bounds():
    bounds = np.tile([0, 0, 2, 1, 1, 3], (100, 1))
    index = AnchorIndex.build(bounds, 3)
    np.testing.assert_array_equal(index.query(camera().planes)[0], np.arange(100))
    assert AnchorIndex.build(np.empty((0, 6))).query(camera().planes)[0].size == 0
    b = anchor_bounds([[0, 0, 0]], [[[1, 0, 0], [-1, 0, 0]]], [[2, 1, 1]], [.5])
    np.testing.assert_allclose(b, [[-2.5, -.5, -.5, 2.5, .5, .5]])


def test_occupancy_closed_cube_and_camera_seed():
    vertices = np.array(list(itertools.product([-1., 1.], repeat=3)))
    faces = []
    for axis in range(3):
        for sign in [-1, 1]:
            ids = np.flatnonzero(vertices[:, axis] == sign)
            faces.extend([[ids[0], ids[1], ids[3]], [ids[0], ids[3], ids[2]]])
    faces = np.array(faces)
    record = make_grid(vertices, faces, [[0, 0, -2]], 4, ([-3]*3, [3]*3))
    assert record['solid_cells'] > 0
    opened = make_grid(vertices, faces[2:], [[0, 0, -2]], 4, ([-3]*3, [3]*3))
    assert opened['solid_cells'] == 0
    seeded = make_grid(vertices, faces, [[0, 0, -2], [0, 0, 0]], 4, ([-3]*3, [3]*3))
    assert seeded['solid_cells'] == 0


def test_union_owner_membership_and_opacity_once():
    model = Model(); materializer = GroupMaterializer(ProxyGSDecoder(model))
    requests = [torch.tensor([0, 2]), torch.tensor([1, 2])]
    state, rows = materializer.prepare(camera(), requests)
    shared = materializer.finish(state)
    assert rows == 3 and model.calls == 1
    assert shared.batch.bundle_metadata.row_offset_slots.tolist() == [0, 1, 0]
    assert shared.row_mask(0).tolist() == [True, False, True]
    assert shared.row_mask(1).tolist() == [False, True, True]
    assert shared.batch.anchor_indices.tolist() == [0, 1, 2]


def test_empty_materialization_and_bad_ids():
    m = GroupMaterializer(ProxyGSDecoder(Model()))
    state, rows = m.prepare(camera(), [torch.empty(0, dtype=torch.long)])
    assert rows == 0 and not len(m.finish(state).row_mask(0))
    for values in [[2, 1], [1, 1], [-1]]:
        with pytest.raises(ValueError):
            union_plan([torch.tensor(values)])


def test_allocation_global_integer_optimum():
    assert cpu_quota(120, 1/60, 1/40) == 72
    for n in range(1, 27):
        for cc, cg in [(1, 1), (3.7, .2), (.3, 8.)]:
            x = cpu_quota(n, cc, cg)
            assert max(cc*x, cg*(n-x)) == min(max(cc*i, cg*(n-i)) for i in range(n+1))
            assert sum(assignment(n, x)) == x


def test_barrier_capacity_tail_and_order():
    views = [camera(i*.01) for i in range(9)]
    selected = set(); lock = threading.Lock()
    decoder = ProxyGSDecoder(Model())
    original = decoder.prepare

    def select(c):
        time.sleep(.002)
        with lock:
            selected.add(id(c))
        return np.array([0, 1, 2], np.int64)

    def prepare(c, ids):
        assert len(selected) == 9  # Every target before any materialization.
        return original(c, ids)

    decoder.prepare = prepare

    def render(c, shared, j):
        assert shared.row_mask(j).all()
        time.sleep(.001)
        return float(c.center[0])

    outputs, report = Schedule(group_size=4, batch_groups=3, capacity_rows=3).run(
        views, select, select, GroupMaterializer(decoder), render)
    np.testing.assert_allclose(outputs, np.arange(9)*.01)
    assert report[0]['peak_retained_rows'] == 3
    assert decoder.model.calls == 3


@pytest.mark.parametrize('failure', ['overflow', 'render', 'prepare'])
def test_failures_cancel_waiting_groups(failure):
    decoder = ProxyGSDecoder(Model())
    if failure == 'prepare':
        def broken(*args):
            raise ValueError('prepare failure')
        decoder.prepare = broken

    def render(*args):
        if failure == 'render':
            raise ValueError('render failure')

    schedule = Schedule(group_size=1, capacity_rows=2 if failure == 'overflow' else 3)
    with pytest.raises((ValueError, RuntimeError)):
        schedule.run([camera()]*4, lambda c: np.arange(3, dtype=np.int64), None,
                     GroupMaterializer(decoder), render)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_cuda_selector_parity():
    bounds = np.array([[0, 0, 2, .2, .2, 3], [10, 10, 2, 11, 11, 3.]])
    index = AnchorIndex.build(bounds)
    assert TensorSelector(index, device='cuda')(camera()).cpu().tolist() == CPUSelector(index)(camera()).tolist()


def test_renderer_masks_opacity_without_copying_shared_attributes(monkeypatch):
    import sys
    from gdmgs.adapters.render import GSplatRenderer
    m = GroupMaterializer(ProxyGSDecoder(Model()))
    prepared, _ = m.prepare(camera(), [torch.tensor([0]), torch.tensor([1, 2])])
    shared = m.finish(prepared)
    old_opacity = shared.batch.opacity.clone()

    def backend(view, batch, background, mode):
        assert batch.xyz is shared.batch.xyz
        assert batch.color is shared.batch.color
        assert batch.opacity.flatten().tolist() == [0., 1., 2.]
        return dict(render=torch.zeros(3, 2, 2), render_alpha=torch.zeros(1, 2, 2), render_depth=None)

    monkeypatch.setitem(sys.modules, 'gdmgs.adapters.gsplat', SimpleNamespace(render_gdmgs_backend=backend))
    result = GSplatRenderer()(camera(), shared, 1)
    assert set(result) == {'render', 'render_alpha'}
    assert torch.equal(old_opacity, shared.batch.opacity)


def test_multiple_batches_preserve_all_targets():
    c = [camera(x*.01) for x in range(11)]
    selected = []

    def select(view):
        selected.append(view)
        return np.array([0], np.int64)

    outputs, report = Schedule(group_size=3, batch_groups=2, group_slots=1).run(
        c, select, None, GroupMaterializer(ProxyGSDecoder(Model())),
        lambda view, shared, j: float(view.center[0]))
    np.testing.assert_allclose(outputs, np.arange(11)*.01)
    assert [r['targets'] for r in report] == [6, 5]
    assert len(selected) == 11

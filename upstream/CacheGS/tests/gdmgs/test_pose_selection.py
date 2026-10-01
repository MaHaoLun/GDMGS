"""CPU contract tests using the actual model decoder and LoD implementation.

Run in the existing render environment. Projection is replaced with a known
visibility oracle; real gsplat projection is covered by the GPU regression.
"""

import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from plyfile import PlyData

from gaussian_renderer import visibility
from scene.gs_model_scaffoldgs.lod_model import GaussianLoDModel, PoseLocalState
from utils import fvdb_visibility
from utils.fvdb_visibility import JaggedVisibilityDescriptor, _infer_batch_size, validate_anchor_ids


def camera(x=0.0, resolution_scale=1.0):
    return SimpleNamespace(
        camera_center=torch.tensor([x, 0.0, 0.0]), resolution_scale=resolution_scale,
        uid=0, FoVx=1.0, FoVy=0.8, image_width=32, image_height=24,
        world_view_transform=torch.eye(4),
    )


def model_fixture(mode="round", progressive=False):
    """Construct real CPU parameters without the production CUDA allocator."""
    model = GaussianLoDModel.__new__(GaussianLoDModel)
    model.setup_functions()
    model.n_offsets = 3
    model.feat_dim = 3
    model.use_feat_bank = False
    model.appearance_dim = 0
    model.dist2level = mode
    model.progressive = progressive
    model.coarse_intervals = [10, 20, 30]
    model.init_level = 0
    model.levels = 4
    model.standard_dist = 4.0
    model.fork = 2
    model._anchor = nn.Parameter(torch.tensor([[1., 0., 0.], [2., 0., 0.], [4., 0., 0.], [8., 0., 0.]]))
    model._level = torch.tensor([[2], [0], [1], [0]], dtype=torch.int32)
    model._extra_level = torch.tensor([.2, .25, .5, 0.])
    model._anchor_feat = nn.Parameter(torch.tensor([[-1., -1., -1.], [1., -1., 1.], [1., 1., 1.], [-1., -1., -1.]]))
    model._offset = nn.Parameter(torch.arange(36, dtype=torch.float32).reshape(4, 3, 3) / 100.)
    model._scaling = nn.Parameter(torch.zeros(4, 6))
    model._rotation = torch.tensor([[1., 0., 0., 0.]]).repeat(4, 1)
    model._anchor_mask = torch.tensor([True, False, True, False])
    model._prog_ratio = torch.full((4, 1), .37)
    model.transition_mask = torch.tensor([False, True, False, True])
    model._last_visibility_descriptor = object()
    model._ensure_fvdb_ready = lambda: None
    model._sync_grid_attributes = lambda names: None
    model._grid_attribute_handle = lambda name: None
    model.mlp_opacity = nn.Linear(6, 3)
    model.mlp_color = nn.Sequential(nn.Linear(6, 9), nn.Sigmoid())
    model.mlp_cov = nn.Linear(6, 21)
    with torch.no_grad():
        model.mlp_opacity.weight.zero_()
        model.mlp_opacity.weight[:, :3] = torch.eye(3)
        model.mlp_opacity.bias.zero_()
        model.mlp_cov.weight.zero_()
        model.mlp_cov.bias.copy_(torch.tensor([0., 0., 0., 1., 0., 0., 0.]).repeat(3))
    return model


class PoseSelectionTests(unittest.TestCase):
    def test_lod_modes_match_legacy_at_interval_boundaries(self):
        for mode in ("floor", "round", "ceil", "progressive"):
            for progressive in (False, True):
                for iteration in (0, 10, 11, 20, 21, 31):
                    with self.subTest(mode=mode, progressive=progressive, iteration=iteration):
                        model = model_fixture(mode, progressive)
                        view = camera(-0.4, 1.5)
                        model.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
                        expected = model._anchor_mask.clone()
                        ratio = model._prog_ratio.clone()
                        transition = model.transition_mask.clone()
                        model.map_to_int_level = lambda *args: self.fail("Pure pose computation called the mutating mapper")
                        state = model.compute_pose_local_state(view, iteration)
                        self.assertTrue(torch.equal(state.anchor_mask, expected))
                        if mode == "progressive":
                            self.assertTrue(torch.equal(state.prog_ratio, ratio))
                            self.assertTrue(torch.equal(state.transition_mask, transition))
                        else:
                            self.assertIsNone(state.prog_ratio)
                            self.assertIsNone(state.transition_mask)

    def test_interleaved_poses_leave_legacy_fields_and_prior_pose_unchanged(self):
        model = model_fixture("progressive")
        original = (model._anchor_mask, model._prog_ratio, model.transition_mask, model._last_visibility_descriptor)
        state_a = model.compute_pose_local_state(camera(), 40)
        saved = (state_a.anchor_mask.clone(), state_a.prog_ratio.clone(), state_a.transition_mask.clone())
        state_b = model.compute_pose_local_state(camera(-6.), 40)
        self.assertFalse(torch.equal(state_a.prog_ratio, state_b.prog_ratio))
        for actual, expected in zip((model._anchor_mask, model._prog_ratio, model.transition_mask, model._last_visibility_descriptor), original):
            self.assertIs(actual, expected)
        for actual, expected in zip((state_a.anchor_mask, state_a.prog_ratio, state_a.transition_mask), saved):
            self.assertTrue(torch.equal(actual, expected))

    def test_explicit_pose_fov_uses_its_mask_without_descriptor_or_model_mutation(self):
        model = model_fixture()
        state = PoseLocalState(torch.tensor([False, True, True, True]))
        sentinel = model._last_visibility_descriptor
        original_mask = model._anchor_mask
        seen = []

        def projection(means, *args, **kwargs):
            seen.append(means.detach().clone())
            return (torch.tensor([[0, 3, 4]]),)

        with patch.object(visibility, "fully_fused_projection", projection):
            sample = visibility.sample_visibility_pose_local(camera(), model, None, None, pose_state=state, return_mask=True)
        self.assertEqual(sample.indices.tolist(), [2, 3])
        self.assertEqual(sample.mask.tolist(), [False, False, True, True])
        self.assertTrue(torch.equal(seen[0], model._anchor[[1, 2, 3]]))
        self.assertIsNone(sample.descriptor)
        self.assertIs(model._anchor_mask, original_mask)
        self.assertIs(model._last_visibility_descriptor, sentinel)
        self.assertEqual(state.anchor_mask.tolist(), [False, True, True, True])

    def test_empty_pose_does_not_call_projection(self):
        model = model_fixture()
        state = PoseLocalState(torch.zeros(4, dtype=torch.bool))
        with patch.object(visibility, "fully_fused_projection", side_effect=AssertionError("empty projection")):
            result = visibility.sample_visibility_pose_local(camera(), model, None, None, pose_state=state, return_mask=True)
        self.assertEqual(result.indices.tolist(), [])
        self.assertEqual(result.mask.tolist(), [False] * 4)

    def test_bundle_rows_follow_unsorted_requests_and_keep_zero_counts(self):
        model = model_fixture()
        state = model.compute_pose_local_state(camera(), 40)
        model._ensure_fvdb_ready = lambda: self.fail("Descriptor-free decode touched fVDB state")
        batch = model.generate_neural_gaussians(camera(), torch.tensor([2, 0, 1]), build_descriptor=False, pose_state=state, return_bundle_metadata=True)
        metadata = batch.bundle_metadata
        self.assertEqual(metadata.request_anchor_ids.tolist(), [2, 0, 1])
        self.assertEqual(metadata.request_level_ids.tolist(), [1, 2, 0])
        self.assertEqual(metadata.counts.tolist(), [3, 0, 2])
        self.assertEqual(metadata.offsets.tolist(), [0, 3, 3, 5])
        self.assertEqual(metadata.row_owner_ids.tolist(), [2, 2, 2, 1, 1])
        self.assertEqual(metadata.row_owner_levels.tolist(), [1, 1, 1, 0, 0])
        self.assertEqual(metadata.row_offset_slots.tolist(), [0, 1, 2, 0, 2])
        expected = model._anchor[[2, 2, 2, 1, 1]] + model._offset[[2, 2, 2, 1, 1], [0, 1, 2, 0, 2]]
        self.assertTrue(torch.equal(batch.xyz, expected))

    def test_empty_request_and_all_zero_bundles(self):
        model = model_fixture()
        state = model.compute_pose_local_state(camera(), 40)
        for ids, counts, offsets in (([], [], [0]), ([3, 0], [0, 0], [0, 0, 0])):
            with self.subTest(ids=ids):
                batch = model.generate_neural_gaussians(camera(), torch.tensor(ids, dtype=torch.long), build_descriptor=False, pose_state=state, return_bundle_metadata=True)
                self.assertEqual(batch.xyz.shape, (0, 3))
                self.assertEqual(batch.bundle_metadata.request_anchor_ids.tolist(), ids)
                self.assertEqual(batch.bundle_metadata.counts.tolist(), counts)
                self.assertEqual(batch.bundle_metadata.offsets.tolist(), offsets)

    def test_next_pose_does_not_change_progressive_decoder_for_current_pose(self):
        model = model_fixture("progressive")
        view = camera()
        ids = torch.tensor([2, 1])
        state = model.compute_pose_local_state(view, 40)
        first = model.generate_neural_gaussians(view, ids, build_descriptor=False, pose_state=state, return_bundle_metadata=True)
        model.compute_pose_local_state(camera(-8.), 40)
        second = model.generate_neural_gaussians(view, ids, build_descriptor=False, pose_state=state, return_bundle_metadata=True)
        self.assertTrue(torch.equal(first.opacity, second.opacity))
        self.assertTrue(torch.equal(first.xyz, second.xyz))

    def test_decoder_and_training_gradients_match_legacy(self):
        old = model_fixture("progressive")
        new = copy.deepcopy(old)
        view = camera()
        ids = torch.tensor([2, 0, 1])
        old.set_anchor_mask(view.camera_center, 40, view.resolution_scale)
        state = new.compute_pose_local_state(view, 40)
        legacy = old.generate_neural_gaussians(view, ids, build_descriptor=False)
        current = new.generate_neural_gaussians(view, ids, build_descriptor=False, pose_state=state, return_bundle_metadata=True)
        self.assertEqual(len(tuple(legacy)), 7)
        self.assertEqual(len(tuple(current)), 7)
        for a, b in zip(tuple(legacy), tuple(current)):
            if isinstance(a, torch.Tensor):
                self.assertEqual(a.dtype, b.dtype)
                self.assertTrue(torch.equal(a, b))
            else:
                self.assertEqual(a, b)
        parameters = lambda m: [m._anchor, m._anchor_feat, m._offset, m._scaling] + list(m.mlp_opacity.parameters()) + list(m.mlp_color.parameters()) + list(m.mlp_cov.parameters())
        for batch in (legacy, current):
            loss = batch.xyz.sum() + batch.color.sum() + batch.opacity.sum() + batch.scaling.float().sum() + batch.rotation.float().sum()
            loss.backward()
        for a, b in zip(parameters(old), parameters(new)):
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.equal(a.grad, b.grad))
        for model in (old, new):
            torch.optim.SGD(parameters(model), lr=.01).step()
        for a, b in zip(parameters(old), parameters(new)):
            self.assertTrue(torch.equal(a, b))

    def test_legacy_boolean_masks_remain_compatible(self):
        model = model_fixture()
        ids = torch.tensor([1, 2])
        for mask in (torch.tensor([False, True, True, False]), torch.tensor([[False], [True], [True], [False]])):
            a = model.generate_neural_gaussians(camera(), ids, build_descriptor=False)
            b = model.generate_neural_gaussians(camera(), mask, build_descriptor=False)
            self.assertTrue(torch.equal(a.xyz, b.xyz))
            self.assertIsNone(b.bundle_metadata)

    def test_ply_export_preserves_row_order_and_offset_binding(self):
        model = model_fixture()
        model.voxel_size = 1.0
        model.init_pos = torch.zeros(3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point_cloud.ply"
            model.save_ply(str(path), 40)
            rows = PlyData.read(str(path)).elements[0]
        self.assertEqual(list(rows["x"]), model._anchor[:, 0].detach().tolist())
        self.assertEqual(list(rows["level"]), [2, 0, 1, 0])
        self.assertEqual(list(rows["f_offset_0"]), model._offset[:, 0, 0].detach().tolist())
        self.assertEqual(list(rows["f_anchor_feat_1"]), model._anchor_feat[:, 1].detach().tolist())

    def test_appearance_override_accepts_scalar_and_legacy_single_integer_sequence(self):
        model = model_fixture()
        model.appearance_dim = 2
        model.embedding_appearance = nn.Embedding(4, 2)
        model.mlp_color = nn.Sequential(nn.Linear(8, 9), nn.Sigmoid())
        state = model.compute_pose_local_state(camera(), 40)
        batches = [model.generate_neural_gaussians(camera(), torch.tensor([1, 2]), ape_code=code, build_descriptor=False, pose_state=state) for code in (2, [2], (2,), torch.tensor([2]))]
        for batch in batches[1:]:
            self.assertTrue(torch.equal(batch.color, batches[0].color))
        uid_batch = model.generate_neural_gaussians(camera(), torch.tensor([1, 2]), build_descriptor=False, pose_state=state)
        zero_batch = model.generate_neural_gaussians(camera(), torch.tensor([1, 2]), ape_code=0, build_descriptor=False, pose_state=state)
        self.assertTrue(torch.equal(uid_batch.color, zero_batch.color))
        for code in (1.5, [], [1, 2], torch.tensor([1.0]), True):
            with self.subTest(code=code):
                with self.assertRaises((TypeError, ValueError)):
                    model.generate_neural_gaussians(camera(), torch.tensor([1, 2]), ape_code=code, build_descriptor=False, pose_state=state)

    def test_invalid_explicit_ids_raise_instead_of_coercion(self):
        model = model_fixture()
        state = model.compute_pose_local_state(camera(), 40)
        for ids in (torch.tensor([-1]), torch.tensor([4]), torch.tensor([1, 1]), torch.tensor([1.]), torch.tensor([[1]])):
            with self.subTest(ids=ids):
                with self.assertRaises((TypeError, ValueError)):
                    model.generate_neural_gaussians(camera(), ids, build_descriptor=False, pose_state=state, return_bundle_metadata=True)

    def test_single_anchor_boolean_mask_preserves_true_selection(self):
        model = model_fixture()
        model._anchor = nn.Parameter(model._anchor[:1].detach().clone())
        for strict in (False, True):
            for selected in (False, True):
                with self.subTest(strict=strict, selected=selected):
                    ids, _ = model._normalize_visibility_input(torch.tensor([selected]), strict=strict)
                    self.assertEqual(ids.tolist(), [0] if selected else [])


class StrictIDTests(unittest.TestCase):
    def test_empty_unsorted_and_narrow_integer_ids(self):
        original = torch.tensor([254, 1], dtype=torch.uint8)
        actual = validate_anchor_ids(original, 1000)
        self.assertEqual(actual.dtype, torch.long)
        self.assertEqual(actual.tolist(), [254, 1])
        original[0] = 0
        self.assertEqual(actual.tolist(), [254, 1])
        self.assertEqual(validate_anchor_ids(torch.tensor([], dtype=torch.int16), 0).tolist(), [])
        self.assertEqual(validate_anchor_ids(torch.tensor([127], dtype=torch.int8), 1000).tolist(), [127])
        with self.assertRaises(ValueError):
            validate_anchor_ids(torch.tensor([-1, 127], dtype=torch.int8), 1000)

    def test_strict_descriptor_rejects_malformed_ids_before_native_construction(self):
        grid = SimpleNamespace(ijk=SimpleNamespace(jdata=torch.zeros(4, 3, dtype=torch.int32)))
        for ids in (torch.tensor([-1]), torch.tensor([4]), torch.tensor([1., 2.]), torch.tensor([1, 1]), torch.tensor([[1]])):
            with self.subTest(ids=ids):
                with self.assertRaises((ValueError, TypeError)):
                    JaggedVisibilityDescriptor.from_indices(grid, ids, strict=True)

    def test_grid_count_preserves_empty_lod_grids(self):
        from utils.fvdb_conversion import build_precompute_payload_header

        grid = SimpleNamespace(grid_count=8, batch_size=1, ijk=SimpleNamespace(jidx=torch.tensor([0, 4])))
        self.assertEqual(_infer_batch_size(grid, level_data=torch.tensor([1, 2])), 8)
        grid.voxel_sizes = torch.ones(8, 3)
        grid.origins = torch.zeros(8, 3)
        header = build_precompute_payload_header("checkpoint", 40000, 2, 1, grid)
        self.assertEqual(header["grid_meta"]["batch_size"], 8)
        self.assertEqual(header["version"], 2)
        grid = SimpleNamespace(grid_count=8, ijk=SimpleNamespace(jidx=torch.empty(0, dtype=torch.long)))
        self.assertEqual(_infer_batch_size(grid), 8)
        self.assertEqual(_infer_batch_size(SimpleNamespace(batch_size=6)), 6)
        self.assertEqual(_infer_batch_size(SimpleNamespace(), level_data=torch.tensor([0, 4])), 5)

    def test_corrupt_serialized_descriptor_is_rejected_before_native_construction(self):
        coords = torch.zeros(2, 3, dtype=torch.int32)
        levels = torch.tensor([0, 4], dtype=torch.int16)
        cases = [
            (coords, levels, value) for value in (0, -1, True, 1.5, "8")
        ] + [
            (coords, levels, 1),
            (coords, torch.tensor([-1, 0]), 8),
            (coords, torch.tensor([0., 1.]), 8),
            (coords, torch.tensor([[0], [1]]), 8),
            (coords, torch.tensor([0]), 8),
            (coords.float(), levels, 8),
            (torch.zeros(2, 4, dtype=torch.int32), levels, 8),
        ]
        with patch.object(fvdb_visibility, "fvdb", SimpleNamespace(JaggedTensor=SimpleNamespace(from_data_and_jidx=lambda *a, **kw: self.fail("Malformed descriptor reached native code")))):
            for coord, level, batch_size in cases:
                with self.subTest(batch_size=batch_size, coords=coord, levels=level):
                    with self.assertRaises((TypeError, ValueError)):
                        JaggedVisibilityDescriptor.from_serialized(object(), ijk_jdata=coord, ijk_jidx=level, batch_size=batch_size)

    def test_valid_serialized_descriptor_keeps_dtype_and_empty_grids(self):
        coords = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        levels = torch.tensor([0, 4], dtype=torch.int16)
        grid = SimpleNamespace(ijk=SimpleNamespace(jdata=coords, jidx=levels))
        calls = []
        def construct(data, jidx, *, batch_size):
            calls.append((data, jidx, batch_size))
            return SimpleNamespace(jdata=data, jidx=jidx)
        with patch.object(fvdb_visibility, "fvdb", SimpleNamespace(JaggedTensor=SimpleNamespace(from_data_and_jidx=construct))):
            descriptor = JaggedVisibilityDescriptor.from_serialized(grid, ijk_jdata=coords, ijk_jidx=levels, batch_size=8)
            empty = JaggedVisibilityDescriptor.from_serialized(grid, ijk_jdata=coords[:0], ijk_jidx=levels[:0], batch_size=8)
        self.assertTrue(torch.equal(descriptor.ijk, coords))
        self.assertEqual(calls[0][0].dtype, torch.int32)
        self.assertEqual(calls[0][1].dtype, torch.int16)
        self.assertEqual([call[2] for call in calls], [8, 8])
        self.assertEqual(empty.numel(), 0)


if __name__ == "__main__":
    unittest.main()

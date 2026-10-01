"""Independent CPU checks of inference identity and explicit-selection boundaries.

The fake decoder emits IDs as its payload; a recording rasterizer then makes
selection/data disagreements observable without CUDA or production selection
helpers acting as their own oracle. Real model/kernel coverage lives separately.
"""

from dataclasses import replace
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gdmgs.adapters import materialize_selected, prepare_selection, rasterize_materialized
from gdmgs.pipeline import FreshPipeline
from gdmgs.runtime.camera_key import CameraQueryKey
from gdmgs.runtime.finalized_scene import validate_anchor_ids
from gdmgs.runtime.session import InferenceSession


class ToyMetadata:
    def __init__(self, ids):
        self.request_anchor_ids = ids.clone()

    def validate(self, row_count=None):
        if row_count is not None and row_count != self.request_anchor_ids.numel():
            raise ValueError("Toy payload must emit one row per request.")


class ToyModel:
    """Small differentiable loaded model whose payload identifies every request."""

    def __init__(self):
        self._anchor = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3))
        self._level = torch.tensor([[0], [1], [0], [2]])
        self._extra_level = torch.zeros(4)
        self._offset = torch.zeros(4, 2, 3)
        self._anchor_feat = torch.ones(4, 3)
        self._scaling = torch.zeros(4, 6)
        self._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
        self._fvdb_grid = object()
        self._fvdb_dirty = False
        self.fvdb_renderer = object()
        self.mlp_opacity = torch.nn.Linear(3, 2).eval()
        self.mlp_cov = torch.nn.Linear(3, 14).eval()
        self.mlp_color = torch.nn.Linear(3, 6).eval()
        self.n_offsets = 2
        self.dist2level = "floor"
        self.decode_calls = []
        self.pose_calls = []

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def fvdb_grid(self):
        return self._fvdb_grid

    def _ensure_fvdb_ready(self):
        return None

    def compute_pose_local_state(self, camera, iteration):
        state = types.SimpleNamespace(
            anchor_mask=torch.ones(4, dtype=torch.bool),
            camera_center=camera.camera_center.clone(),
            prog_ratio=torch.ones(4, 1),
            transition_mask=torch.zeros(4, dtype=torch.bool),
        )
        self.pose_calls.append(state)
        return state

    def generate_neural_gaussians(self, camera, ids, ape_code, **kwargs):
        self.decode_calls.append((ids.clone(), ape_code, kwargs))
        metadata = ToyMetadata(ids)
        return types.SimpleNamespace(
            bundle_metadata=metadata,
            payload_owner_ids=ids.clone(),
            xyz=ids.to(torch.float32).unsqueeze(-1).repeat(1, 3),
            sh_degree=None,
        )


def camera():
    return types.SimpleNamespace(
        world_view_transform=torch.eye(4), camera_center=torch.zeros(3),
        FoVx=1.0, FoVy=0.8, image_width=8, image_height=6,
        znear=0.01, zfar=100.0, resolution_scale=1.0, uid=3,
    )


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        self.model = ToyModel()
        self.session = InferenceSession(self.model, "/read-only/checkpoint", 40000)
        self.camera = camera()
        self.background = torch.zeros(3)
        self.raster_calls = []
        self.visibility_calls = []
        render_module = types.ModuleType("gaussian_renderer.render")
        visibility_module = types.ModuleType("gaussian_renderer.visibility")
        package = types.ModuleType("gaussian_renderer")
        package.__path__ = []

        def record_raster(view, model, batch, background, render_mode, **kwargs):
            self.raster_calls.append((batch, kwargs))
            return {"render": batch.payload_owner_ids.clone()}

        def record_visibility(view, model, pipe, background, **kwargs):
            self.visibility_calls.append(kwargs)
            return types.SimpleNamespace(indices=torch.tensor([0, 2]))

        render_module.rasterize_batch = record_raster
        visibility_module.sample_visibility_pose_local = record_visibility
        replacements = {
            "gaussian_renderer": package,
            "gaussian_renderer.render": render_module,
            "gaussian_renderer.visibility": visibility_module,
        }
        # Restore only the three substituted import entries. patch.dict on the
        # entire registry removes unrelated modules lazily loaded during a test;
        # unloading Torch/native registrations can then corrupt interpreter exit.
        missing = object()
        original_modules = {name: sys.modules.get(name, missing) for name in replacements}
        sys.modules.update(replacements)

        def restore_modules():
            for name, original in original_modules.items():
                if original is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

        self.addCleanup(restore_modules)
        self.environment = patch.dict("os.environ", {"CACHE_ENABLE": "0", "PRECOMP_INDICES_PATH": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_explicit_selection_preserves_order_without_fov_reselection(self):
        with FreshPipeline(self.model, "/read-only/checkpoint", 40000) as pipeline:
            output = pipeline.render(self.camera, None, self.background,
                                     selected_ids=torch.tensor([3, 0, 2]))
        self.assertEqual(output["render"].tolist(), [3, 0, 2])
        self.assertEqual(output["selected_anchor_ids"].tolist(), [3, 0, 2])
        self.assertEqual(len(self.model.decode_calls), 1)
        self.assertEqual(self.visibility_calls, [])
        self.assertEqual(self.raster_calls[0][1]["visible_mask"].tolist(), [True, False, True, True])

    def test_selection_takes_a_copy_of_callers_ids(self):
        ids = torch.tensor([3, 1])
        result = materialize_selected(self.session, ids, self.camera)
        ids[0] = 0
        self.assertEqual(result.anchor_ids.tolist(), [3, 1])
        self.assertEqual(result.batch.payload_owner_ids.tolist(), [3, 1])

    def test_empty_explicit_selection_is_not_replaced_by_fov(self):
        with FreshPipeline(self.model, "/read-only/checkpoint", 40000) as pipeline:
            output = pipeline.render(self.camera, None, self.background,
                                     selected_ids=torch.empty(0, dtype=torch.long))
        self.assertEqual(output["render"].numel(), 0)
        self.assertEqual(self.visibility_calls, [])
        self.assertFalse(bool(self.raster_calls[0][1]["visible_mask"].any()))

    def test_wrong_ids_are_rejected_before_decoder(self):
        cases = (
            torch.tensor([0.5]), torch.tensor([True, False]),
            torch.tensor([[0, 1]]), torch.tensor([0, 0]),
            torch.tensor([-1]), torch.tensor([4]), [0, 1],
        )
        for ids in cases:
            with self.subTest(ids=ids):
                with self.assertRaises((TypeError, ValueError)):
                    materialize_selected(self.session, ids, self.camera)
        self.assertEqual(self.model.decode_calls, [])

    def test_small_integer_dtypes_do_not_wrap_large_anchor_count(self):
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                ids = validate_anchor_ids(torch.tensor([100, 0, 45], dtype=dtype), 300)
                self.assertEqual(ids.tolist(), [100, 0, 45])
                self.assertEqual(ids.dtype, torch.int64)

    def test_new_session_rejects_old_prepared_and_materialized_objects(self):
        prepared = prepare_selection(self.session, self.camera, None, self.background)
        materialized = materialize_selected(self.session, torch.tensor([2]), self.camera)
        other = InferenceSession(self.model, "/read-only/checkpoint", 40000)
        self.assertNotEqual(self.session.scene.token, other.scene.token)
        with self.assertRaises(ValueError):
            materialize_selected(other, torch.tensor([2]), self.camera, prepared=prepared)
        with self.assertRaises(ValueError):
            rasterize_materialized(other, materialized, self.camera, self.background)

    def test_close_rejects_validation_materialization_and_rasterization(self):
        materialized = materialize_selected(self.session, torch.tensor([2]), self.camera)
        self.session.close()
        self.session.close()
        for operation in (
            self.session.validate,
            lambda: materialize_selected(self.session, torch.tensor([2]), self.camera),
            lambda: rasterize_materialized(self.session, materialized, self.camera, self.background),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(RuntimeError):
                    operation()

    def test_finalization_does_not_disable_training_gradients(self):
        params = [self.model._anchor, *self.model.mlp_opacity.parameters()]
        self.assertTrue(all(value.requires_grad for value in params))
        loss = self.model.mlp_opacity(self.model._anchor).square().sum()
        loss.backward()
        self.assertTrue(all(value.grad is not None for value in params))
        before = self.model._anchor.detach().clone()
        torch.optim.SGD(params, lr=0.01).step()
        self.assertFalse(torch.equal(before, self.model._anchor))
        with self.assertRaises(RuntimeError):
            self.session.validate()

    def test_reloaded_rows_invalidate_old_session(self):
        self.model._anchor = torch.nn.Parameter(self.model._anchor.detach().clone())
        with self.assertRaises(RuntimeError):
            self.session.validate()
        replacement = InferenceSession(self.model, "/read-only/checkpoint", 40000)
        replacement.validate()
        self.assertNotEqual(replacement.scene.token, self.session.scene.token)

    def test_finalization_rejects_unresolved_or_fractional_iteration(self):
        for iteration in (-1, 40000.5, True):
            with self.subTest(iteration=iteration):
                with self.assertRaises((TypeError, ValueError)):
                    InferenceSession(self.model, "/read-only/checkpoint", iteration)

    def test_zero_activated_scale_accepts_negative_infinity_log_scale(self):
        self.model._scaling[0, 0] = -torch.inf
        zero_scale = InferenceSession(self.model, "/read-only/checkpoint", 40000)
        zero_scale.validate()
        self.assertEqual(torch.exp(self.model._scaling[0, 0]).item(), 0.0)

    def test_invalid_or_overflowing_log_scale_is_rejected(self):
        for value in (torch.nan, torch.inf, 1000.0):
            with self.subTest(value=value):
                self.model._scaling[0, 0] = value
                with self.assertRaises(ValueError):
                    InferenceSession(self.model, "/read-only/checkpoint", 40000)

    def test_nonfinite_decoder_parameters_are_rejected_at_finalization(self):
        with torch.no_grad():
            self.model.mlp_color.weight[0, 0] = torch.nan
        with self.assertRaises(ValueError):
            InferenceSession(self.model, "/read-only/checkpoint", 40000)

    def test_declared_checkpoint_must_match_recorded_loaded_ply(self):
        self.model._loaded_ply_path = "/read-only/checkpoint/point_cloud/iteration_40000/point_cloud.ply"
        current = InferenceSession(self.model, "/read-only/checkpoint", 40000)
        current.validate()
        with self.assertRaises(ValueError):
            InferenceSession(self.model, "/another/checkpoint", 40000)
        with self.assertRaises(ValueError):
            InferenceSession(self.model, "/read-only/checkpoint", 40001)
        self.model._loaded_ply_path = "/another/checkpoint/point_cloud/iteration_40000/point_cloud.ply"
        with self.assertRaises(RuntimeError):
            current.validate()

    def test_inplace_row_update_invalidates_session(self):
        with torch.no_grad():
            self.model._anchor[0, 0].add_(1)
        with self.assertRaises(RuntimeError):
            self.session.validate()

    def test_inplace_decoder_update_invalidates_session(self):
        with torch.no_grad():
            self.model.mlp_color.weight.add_(1)
        with self.assertRaises(RuntimeError):
            self.session.validate()

    def test_replaced_grid_or_raster_backend_invalidates_session(self):
        self.model._fvdb_grid = object()
        with self.assertRaises(RuntimeError):
            self.session.validate()
        other = InferenceSession(self.model, "/read-only/checkpoint", 40000)
        self.model.fvdb_renderer = object()
        with self.assertRaises(RuntimeError):
            other.validate()

    def test_lod_configuration_and_training_mode_invalidate_session(self):
        self.model.dist2level = "progressive"
        with self.assertRaises(RuntimeError):
            self.session.validate()
        self.model.dist2level = "floor"
        self.model.mlp_opacity.train()
        with self.assertRaises(RuntimeError):
            self.session.validate()

    def test_camera_key_copies_pose_values(self):
        key = CameraQueryKey.from_camera(self.camera, 40000)
        original_values = key.world_to_camera
        self.camera.world_view_transform[3, 0] = 1
        self.assertEqual(key.world_to_camera, original_values)
        self.assertNotEqual(key, CameraQueryKey.from_camera(self.camera, 40000))

    def test_appearance_resolution_pose_and_iteration_are_identity(self):
        original = CameraQueryKey.from_camera(self.camera, 40000)
        changes = {
            "FoVx": 1.1, "FoVy": 1.0, "image_width": 10, "image_height": 7,
            "resolution_scale": 2.0, "uid": 4, "znear": 0.02, "zfar": 200.0,
            "camera_center": torch.tensor([1.0, 0.0, 0.0]),
        }
        for field, value in changes.items():
            changed = camera()
            setattr(changed, field, value)
            with self.subTest(field=field):
                self.assertNotEqual(original, CameraQueryKey.from_camera(changed, 40000))
        self.assertNotEqual(original, CameraQueryKey.from_camera(self.camera, 40001))
        self.assertNotEqual(original, CameraQueryKey.from_camera(self.camera, 40000, [2]))
        self.assertEqual(CameraQueryKey.from_camera(self.camera, 40000, [2]),
                         CameraQueryKey.from_camera(self.camera, 40000, 2))

    def test_invalid_camera_values_are_rejected(self):
        for field, value in (("image_width", 8.5), ("image_height", True),
                             ("FoVx", float("nan")), ("resolution_scale", 0.0)):
            changed = camera()
            setattr(changed, field, value)
            with self.subTest(field=field, value=value):
                with self.assertRaises((TypeError, ValueError)):
                    CameraQueryKey.from_camera(changed, 40000)
        with self.assertRaises((TypeError, ValueError)):
            CameraQueryKey.from_camera(self.camera, 40000.5)

    def test_prepared_pose_rejected_after_camera_change(self):
        prepared = prepare_selection(self.session, self.camera, None, self.background)
        self.camera.camera_center[0] = 0.25
        with self.assertRaises(ValueError):
            materialize_selected(self.session, torch.tensor([2]), self.camera, prepared=prepared)

    def test_prepared_pose_tensor_mutation_is_rejected(self):
        prepared = prepare_selection(self.session, self.camera, None, self.background)
        prepared.pose_state.prog_ratio[2, 0] = 0.1
        with self.assertRaises((ValueError, RuntimeError)):
            materialize_selected(self.session, torch.tensor([2]), self.camera, prepared=prepared)
        self.assertEqual(self.model.decode_calls, [])

    def test_materialized_pose_and_appearance_cannot_be_reused_as_fresh(self):
        result = materialize_selected(self.session, torch.tensor([2]), self.camera)
        with self.assertRaises(ValueError):
            rasterize_materialized(self.session, result, self.camera, self.background, ape_code=1)
        self.camera.world_view_transform[3, 0] = 1.0
        with self.assertRaises(ValueError):
            rasterize_materialized(self.session, result, self.camera, self.background)

    def test_mutated_materialized_ids_cannot_mislabel_rendered_payload(self):
        result = materialize_selected(self.session, torch.tensor([3, 1]), self.camera)
        result.anchor_ids[0] = 0
        with self.assertRaises((ValueError, RuntimeError)):
            rasterize_materialized(self.session, result, self.camera, self.background)
        self.assertEqual(self.raster_calls, [])

    def test_mutated_materialized_payload_cannot_be_rasterized(self):
        result = materialize_selected(self.session, torch.tensor([3, 1]), self.camera)
        result.batch.xyz[0, 0] = 999.0
        with self.assertRaises((ValueError, RuntimeError)):
            rasterize_materialized(self.session, result, self.camera, self.background)
        self.assertEqual(self.raster_calls, [])

    def test_mutated_sh_degree_cannot_change_fresh_raster_semantics(self):
        result = materialize_selected(self.session, torch.tensor([3, 1]), self.camera)
        result.batch.sh_degree = 3
        with self.assertRaises((ValueError, RuntimeError)):
            rasterize_materialized(self.session, result, self.camera, self.background)
        self.assertEqual(self.raster_calls, [])

    def test_replaced_materialized_ids_cannot_mislabel_rendered_payload(self):
        result = materialize_selected(self.session, torch.tensor([3, 1]), self.camera)
        result = replace(result, anchor_ids=torch.tensor([0, 1]))
        with self.assertRaises((ValueError, RuntimeError)):
            rasterize_materialized(self.session, result, self.camera, self.background)
        self.assertEqual(self.raster_calls, [])

    def test_decoder_reordering_is_rejected_before_raster(self):
        original_decode = self.model.generate_neural_gaussians

        def reversed_decode(view, ids, ape_code, **kwargs):
            return original_decode(view, ids.flip(0), ape_code, **kwargs)

        self.model.generate_neural_gaussians = reversed_decode
        with self.assertRaises(RuntimeError):
            materialize_selected(self.session, torch.tensor([3, 1]), self.camera)
        self.assertEqual(self.raster_calls, [])

    def test_legacy_cache_environment_conflict_is_explicit(self):
        for key, value in (("CACHE_ENABLE", "1"), ("PRECOMP_INDICES_PATH", "/old/indices.pt")):
            with self.subTest(key=key), patch.dict("os.environ", {key: value}):
                with self.assertRaises(ValueError):
                    FreshPipeline(self.model, "/read-only/checkpoint", 40000)


if __name__ == "__main__":
    unittest.main()

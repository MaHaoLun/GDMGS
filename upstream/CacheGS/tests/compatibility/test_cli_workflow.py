"""CPU-only contract tests for CLI isolation, camera mapping and frame pairing."""

import os
import contextlib
import json
import weakref
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.render_workflow import (enumerate_cameras, load_scene, make_frame_renderer,
                                   resolve_checkpoint_iteration, render_artifact_path)
from utils.runtime_options import (configure_device, output_root_for,
                                   output_subdirectory, validate_pipeline_options)
import metrics
import precompute_cache
import render
import render2


class CLIWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.model = self.root / "checkpoint"
        ply = self.model / "point_cloud" / "iteration_40000" / "point_cloud.ply"
        ply.parent.mkdir(parents=True)
        ply.touch()
        self.source = self.root / "dataset"
        self.source.mkdir()
        self.dataset = SimpleNamespace(model_path=str(self.model), source_path=str(self.source),
                                       eval=True, resolution_scales=[1.0])
        self.train = [SimpleNamespace(uid=10, image_name="c"), SimpleNamespace(uid=4, image_name="a")]
        self.test = [SimpleNamespace(uid=8, image_name="b")]
        self.scene = SimpleNamespace(getTrainCameras=lambda: self.train,
                                     getTestCameras=lambda: self.test)

    def test_import_does_not_import_gpu_stack_or_change_visibility(self):
        script = '''import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3,7"
import render, render2, precompute_cache, metrics
assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,7"
assert "torch" not in sys.modules
assert "gaussian_renderer" not in sys.modules
assert "gdmgs" not in sys.modules
'''
        subprocess.run([sys.executable, "-c", script], cwd=str(ROOT), check=True)

    def test_old_cli_defaults_and_new_options(self):
        args = render.build_parser().parse_args(["-m", "model"])
        self.assertEqual((args.pipeline, args.output_root, args.output_name), ("legacy", None, "test"))
        args = render2.build_parser().parse_args(["-m", "model", "--skip_train", "--device", "cuda:1", "--pipeline", "gdmgs"])
        self.assertTrue(args.skip_train)
        self.assertFalse(args.skip_test)
        self.assertEqual(args.device, "cuda:1")
        self.assertEqual(precompute_cache.build_parser().parse_args(["-m", "model"]).iteration, -1)
        self.assertEqual(metrics.build_parser().parse_args(["-m", "a", "b"]).model_paths, ["a", "b"])

    def test_legacy_cache_options_are_unchanged(self):
        validate_pipeline_options("legacy", True, {"CACHE_ENABLE": "1", "PRECOMP_INDICES_PATH": "old.pt"})

    def test_gdmgs_rejects_all_legacy_conflicts(self):
        for value in ("1", "true", "yes", "2", "false"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_pipeline_options("gdmgs", False, {"CACHE_ENABLE": value})
        with self.assertRaises(ValueError):
            validate_pipeline_options("gdmgs", True, {})
        with self.assertRaises(ValueError):
            validate_pipeline_options("gdmgs", False, {"PRECOMP_INDICES_PATH": "old.pt"})
        validate_pipeline_options("gdmgs", False, {"CACHE_ENABLE": "0"})

    def test_output_is_separate_and_does_not_create_directories(self):
        self.assertEqual(output_root_for(str(self.model)), str(self.model))
        target = self.root / "artifacts"
        self.assertEqual(output_root_for(self.model, target), str(target))
        self.assertFalse(target.exists())
        for target in (self.model, self.model / "test"):
            with self.assertRaises(ValueError):
                output_root_for(self.model, target)
        alias = self.root / "alias"
        alias.symlink_to(self.model, target_is_directory=True)
        with self.assertRaises(ValueError):
            output_root_for(self.model, alias / "test")

    def test_output_name_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            output_subdirectory(self.root / "artifacts", "../../checkpoint")
        self.assertEqual(output_subdirectory(self.root, "test/trajectory"), str(self.root / "test/trajectory"))

    def test_precompute_preserves_out_and_defaults(self):
        self.assertEqual(precompute_cache.precompute_output_path("model"), "model/precomputed_indices.pt")
        self.assertEqual(precompute_cache.precompute_output_path("model", out="old.pt"), "old.pt")
        with self.assertRaises(ValueError):
            precompute_cache.precompute_output_path("model", "new", "old.pt")

    def test_scene_read_only_guard_rejects_zero_or_missing_checkpoint(self):
        with self.assertRaises(ValueError):
            resolve_checkpoint_iteration(self.model, 0)
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint_iteration(self.model, 50000)
        self.assertEqual(resolve_checkpoint_iteration(self.model, -1), 40000)
        factory = unittest.mock.Mock()
        with self.assertRaises(ValueError):
            load_scene(self.dataset, object(), 0, scene_factory=factory)
        factory.assert_not_called()

    def test_city_merged_and_precompute_force_eval_then_restore(self):
        (self.source / "transforms.json").touch()
        captured = []
        def factory(dataset, model, **kwargs):
            captured.append((dataset.eval, kwargs))
            return self.scene
        loaded = load_scene(self.dataset, object(), -1, scene_factory=factory)
        self.assertIs(loaded, self.scene)
        self.assertFalse(captured[0][0])
        self.assertEqual(captured[0][1]["load_iteration"], 40000)
        self.assertFalse(captured[0][1]["shuffle"])
        self.assertTrue(self.dataset.eval)
        self.assertEqual(enumerate_cameras(self.dataset, loaded), self.train)
        with patch.object(precompute_cache, "load_scene", return_value=loaded) as loader:
            cameras, _ = precompute_cache._enumerate_cameras_in_order(self.dataset, object(), -1)
            self.assertEqual(loader.call_args.args[3], "merged")
            self.assertEqual([c.uid for c in cameras], [10, 4])

    def test_city_eval_restored_on_scene_failure(self):
        (self.source / "transforms.json").touch()
        def fail(*args, **kwargs):
            raise RuntimeError("load failed")
        with self.assertRaises(RuntimeError):
            load_scene(self.dataset, object(), -1, scene_factory=fail)
        self.assertTrue(self.dataset.eval)

    def test_split_policy_keeps_original_city_eval_and_uids(self):
        (self.source / "transforms.json").touch()
        seen = []
        load_scene(self.dataset, object(), -1, "split", lambda dataset, *a, **kw: seen.append(dataset.eval))
        self.assertEqual(seen, [True])
        groups = enumerate_cameras(self.dataset, self.scene, "split")
        self.assertEqual([(name, [c.uid for c in cams]) for name, cams in groups], [("train", [10, 4]), ("test", [8])])
        self.assertEqual(enumerate_cameras(self.dataset, self.scene, "split", True, False), [("test", self.test)])

    def test_colmap_merges_image_order_without_uid_reassignment(self):
        (self.source / "sparse" / "0").mkdir(parents=True)
        self.assertEqual([c.uid for c in enumerate_cameras(self.dataset, self.scene)], [4, 8, 10])
        self.dataset.eval = False
        self.assertIs(enumerate_cameras(self.dataset, self.scene), self.train)

    def test_colmap_uid_fallback_keeps_ids(self):
        (self.source / "sparse" / "0").mkdir(parents=True)
        del self.train[0].image_name
        self.assertEqual([c.uid for c in enumerate_cameras(self.dataset, self.scene)], [4, 8, 10])

    def test_frame_names_pair_actual_images_and_reject_missing_frames(self):
        renders, gt = self.root / "renders", self.root / "gt"
        renders.mkdir()
        gt.mkdir()
        for name in ("zz.png", "aa.png"):
            (renders / name).touch()
            (gt / name).touch()
        self.assertEqual([p[0] for p in metrics.image_pair_paths(renders, gt)], ["aa.png", "zz.png"])
        (gt / "missing.png").touch()
        with self.assertRaises(ValueError):
            metrics.image_pair_paths(renders, gt)

    def test_missing_visibility_is_not_silently_zero(self):
        with self.assertRaises(ValueError):
            metrics._parse_visibility_entry(None)
        self.assertEqual(metrics._parse_visibility_entry(12), (12, {}, [], None))
        self.assertEqual(metrics._parse_visibility_entry({"visible_gaussians": 0})[0], 0)

    def test_metric_output_destinations_are_independent(self):
        target = self.root / "results"
        self.assertEqual(metrics.metric_output_paths([self.model], target), [target])
        with self.assertRaises(ValueError):
            metrics.metric_output_paths([self.model, self.root / "other" / "checkpoint"], target)

    def test_legacy_frame_dispatch_keeps_optional_ape_and_cache_protocol(self):
        calls = []
        def renderer(*args, **kwargs):
            calls.append((args, kwargs))
            return {"render": "image"}
        renderer.__name__ = "render"
        module = SimpleNamespace(render=renderer)
        helpers = SimpleNamespace(get_render_func=lambda _: "render")
        with patch.dict(sys.modules, {"gaussian_renderer": module, "utils.general_utils": helpers}):
            frame = make_frame_renderer("scaffoldgs", "model", "pipe", "bg", 40000, "RGB", enable_cache=False)
            self.assertEqual(frame("camera"), {"render": "image"})
            frame = make_frame_renderer("scaffoldgs", "model", "pipe", "bg", 40000, "RGB", 4, True)
            frame("camera")
        self.assertEqual(calls[0], (("camera", "model", "pipe", "bg", 40000, "RGB"), {"disable_cache": True}))
        self.assertEqual(calls[1][0][-1], 4)
        self.assertEqual(calls[1][1], {"disable_cache": False})

    def test_gdmgs_dispatch_uses_fresh_pipeline_directly(self):
        events = []
        class Pipeline:
            def __init__(self, *args):
                events.append(args)
            def close(self):
                events.append("closed")
            def render(self, *args, **kwargs):
                events.append((args, kwargs))
                return "fresh"
        with patch.dict(sys.modules, {"gdmgs.pipeline": SimpleNamespace(FreshPipeline=Pipeline)}), patch.dict(os.environ, {"CACHE_ENABLE": "0", "PRECOMP_INDICES_PATH": ""}):
            frame = make_frame_renderer("scaffoldgs", "model", "pipe", "bg", 40000, "RGB", pipeline="gdmgs", checkpoint_path="checkpoint")
            with frame:
                self.assertEqual(frame("camera"), "fresh")
            frame.close()
        self.assertEqual(events.count("closed"), 1)
        self.assertEqual(events[0], ("model", "checkpoint", 40000))
        self.assertEqual(events[1], (("camera", "pipe", "bg", "RGB"), {"ape_code": -1}))


    def test_render_destination_rejects_checkpoint_via_parent_root_and_symlinks(self):
        with self.assertRaises(ValueError):
            render_artifact_path(self.model, self.root, self.model.name, 40000)
        artifacts = self.root / "artifacts"
        (artifacts / "test").mkdir(parents=True)
        (artifacts / "test" / "ours_40000").symlink_to(self.model, target_is_directory=True)
        with self.assertRaises(ValueError):
            render_artifact_path(self.model, artifacts, "test", 40000)
        (artifacts / "safe" / "ours_40000" / "renders").mkdir(parents=True)
        (artifacts / "safe" / "ours_40000" / "renders" / "00000.png").symlink_to(self.model / "config.yaml")
        with self.assertRaises(ValueError):
            render_artifact_path(self.model, artifacts, "safe", 40000, "renders", "00000.png")

    def test_fresh_session_is_closed_on_render_exception(self):
        from utils.render_workflow import FrameRenderer
        closed = []
        def fail(camera):
            raise RuntimeError("frame failed")
        with self.assertRaises(RuntimeError):
            with FrameRenderer(fail, lambda: closed.append(True)) as frame:
                frame("camera")
        self.assertEqual(closed, [True])

    def test_legacy_context_resets_once_per_scene_not_per_split(self):
        self.dataset.base_model = "scaffoldgs"
        self.dataset.render_mode = "RGB"
        self.scene.loaded_iter = 40000
        self.scene.background = "background"
        torch = SimpleNamespace(no_grad=contextlib.nullcontext)
        with patch.dict(sys.modules, {"torch": torch}), patch.object(render2, "load_inference_scene", return_value=(object(), self.scene)), patch.object(render2, "render_set") as draw, patch.object(render2, "reset_scene_render_context") as reset:
            render2.render_sets(self.dataset, object(), object(), -1, False, False, -1)
            self.assertEqual(reset.call_count, 1)
            self.assertEqual(draw.call_count, 2)
            render2.render_sets(self.dataset, object(), object(), -1, False, False, -1)
            self.assertEqual(reset.call_count, 2)

    def test_logical_device_selection_preserves_visible_devices(self):
        selected = []
        class Device:
            def __init__(self, value, index=None):
                value, _, suffix = value.partition(":")
                self.type = value
                self.index = index if index is not None else (int(suffix) if suffix else None)
        cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                               device_count=lambda: 2, set_device=lambda value: selected.append(value.index))
        with patch.dict(sys.modules, {"torch": SimpleNamespace(device=Device, cuda=cuda)}), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,7"}):
            self.assertEqual(configure_device("cuda:1").index, 1)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3,7")
            with self.assertRaises(ValueError):
                configure_device("cuda:2")
        self.assertEqual(selected, [1])

    def test_metrics_stream_pairs_and_keep_names_and_fields(self):
        method = self.root / "rendered" / "test" / "ours_40000"
        for folder in ("renders", "gt"):
            (method / folder).mkdir(parents=True)
            for name in ("b.png", "a.png"):
                (method / folder / name).touch()
        (method / "per_view_count.json").write_text(json.dumps({"b.png": 20, "a.png": 10}))
        class Scalar:
            def __init__(self, value):
                self.value = value
            def item(self):
                return self.value
        class Scores:
            def __init__(self, values):
                self.values = values
            def float(self):
                return self
            def mean(self):
                return Scalar(sum(self.values) / len(self.values))
        class Pair:
            pass
        references, loaded = [], []
        def load(render_path, gt_path, device):
            self.assertTrue(all(ref() is None for ref in references))
            render_image, truth = Pair(), Pair()
            references.extend((weakref.ref(render_image), weakref.ref(truth)))
            loaded.append(render_path.name)
            return render_image, truth
        class Evaluator:
            def eval(self):
                return self
            def __call__(self, *args):
                return Scalar(0.25)
        modules = {"torch": SimpleNamespace(no_grad=contextlib.nullcontext, tensor=Scores),
                   "tqdm": SimpleNamespace(tqdm=lambda values, **kw: values),
                   "utils.loss_utils": SimpleNamespace(ssim=lambda *args: Scalar(0.75)),
                   "utils.image_utils": SimpleNamespace(psnr=lambda *args: Scalar(30.0))}
        target = self.root / "metrics"
        with patch.dict(sys.modules, modules), patch.object(metrics, "configure_device", return_value="cpu"), patch.object(metrics, "_load_image_pair", side_effect=load), patch.object(metrics, "lpips_fn", Evaluator()):
            metrics.evaluate([str(self.root / "rendered")], output_root=str(target), device="cpu")
        self.assertEqual(loaded, ["a.png", "b.png"])
        result = json.loads((target / "results.json").read_text())["ours_40000"]
        self.assertEqual(result, {"PSNR": 30.0, "SSIM": 0.75, "LPIPS": 0.25, "GS_NUMS": 15.0})
        per_view = json.loads((target / "per_view.json").read_text())["ours_40000"]
        self.assertEqual(per_view["GS_NUMS"], {"a.png": 10, "b.png": 20})
        self.assertFalse((self.root / "rendered" / "results.json").exists())


if __name__ == "__main__":
    unittest.main()

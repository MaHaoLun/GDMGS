"""Nonauthor regressions for mesh/source binding before real GPU execution.

Only existing temporary mesh/depth records are read. There is deliberately no
checkpoint or dataset tree to scan, and the tests do not import a GPU runtime.
"""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "index_benchmark_binding_under_test", ROOT / "tools/gdmgs/benchmark_index.py")
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class MeshSourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "read_only_source_not_present" / "truck"
        self.mesh = self.root / "mesh.npz"
        self.depth_path = self.root / "depth_manifest.json"
        self.stats = [{"path": str(self.model / "point_cloud/iteration_40000/point_cloud.ply"),
                       "size": 1937, "mtime_ns": 1820000000000000000}]
        self.frames = [
            {"frame_index": i, "split": split, "uid": i,
             "image_name": f"view_{i:03d}", "R": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
             "T": [float(i), 0., 0.], "FoVx": 1., "FoVy": .8,
             "render_dimensions": [800, 600], "resolution_scale": 1.,
             "camera_token": f"experiment:truck:frame-{i}"}
            for i, split in enumerate(("train", "test", "train"))]
        bare = [{k: v for k, v in frame.items() if k != "camera_token"} for frame in self.frames]
        fusion = [dict(bare[0], fusion_index=0), dict(bare[2], fusion_index=1)]
        self.depth = {
            "status": "complete", "checkpoint_stats_unchanged": True,
            "model_path": str(self.model), "iteration": 40000,
            "run_token": "truck-depth-source", "checkpoint_stats": copy.deepcopy(self.stats),
            "evaluation_frames": bare, "fusion_frames": fusion,
            "completed": [{"frame_index": 0, "fusion_index": 0},
                          {"frame_index": 2, "fusion_index": 1}],
        }
        self.metadata = {
            "model_path": str(self.model), "iteration": 40000,
            "mesh_file": str(self.mesh), "mesh_token": "truck-mesh-candidate",
            "depth_manifest": str(self.depth_path), "depth_run_token": "truck-depth-source",
            "fusion_frames": copy.deepcopy(fusion),
        }

    def validate(self, *, metadata=None, depth=None):
        self.depth_path.write_text(json.dumps(self.depth if depth is None else depth))
        return BENCHMARK.validate_mesh_binding(
            self.mesh, self.metadata if metadata is None else metadata,
            self.model, 40000, self.frames, self.stats)

    def test_valid_existing_records_bind_without_checkpoint_or_dataset_access(self):
        self.assertFalse(self.model.exists())
        record = self.validate()
        self.assertEqual(record["model_path"], str(self.model.resolve()))
        self.assertEqual(record["iteration"], 40000)
        self.assertEqual(record["mesh_token"], "truck-mesh-candidate")
        self.assertIs(record["validated_depth_binding"], True)
        self.assertFalse(self.model.exists())

    def test_same_sized_wrong_scene_mesh_and_wrong_iteration_are_rejected(self):
        for field, value in (("model_path", str(self.root / "read_only_source_not_present" / "train")),
                             ("iteration", 30000)):
            with self.subTest(field=field):
                metadata = copy.deepcopy(self.metadata)
                metadata[field] = value
                with self.assertRaisesRegex(ValueError, "checkpoint or iteration"):
                    self.validate(metadata=metadata)

    def test_mesh_file_and_depth_run_cannot_be_substituted_under_valid_scene_name(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["mesh_file"] = str(self.root / "another_mesh.npz")
        with self.assertRaisesRegex(ValueError, "different artifact"):
            self.validate(metadata=metadata)
        depth = copy.deepcopy(self.depth)
        depth["run_token"] = "unrelated-earlier-depth-run"
        with self.assertRaisesRegex(ValueError, "depth source"):
            self.validate(depth=depth)

    def test_same_frame_count_does_not_hide_changed_camera_or_order(self):
        depth = copy.deepcopy(self.depth)
        depth["evaluation_frames"][1]["T"][0] += .25
        with self.assertRaisesRegex(ValueError, "calibration/order"):
            self.validate(depth=depth)
        depth = copy.deepcopy(self.depth)
        depth["evaluation_frames"] = depth["evaluation_frames"][::-1]
        with self.assertRaisesRegex(ValueError, "calibration/order"):
            self.validate(depth=depth)

    def test_claimed_complete_but_reduced_fusion_is_rejected(self):
        depth, metadata = copy.deepcopy(self.depth), copy.deepcopy(self.metadata)
        depth["fusion_frames"] = depth["fusion_frames"][:1]
        depth["completed"] = depth["completed"][:1]
        metadata["fusion_frames"] = metadata["fusion_frames"][:1]
        with self.assertRaisesRegex(ValueError, "complete frozen fusion"):
            self.validate(metadata=metadata, depth=depth)

    def test_missing_or_reordered_observations_fail_despite_complete_manifest(self):
        for completed in (self.depth["completed"][:1], self.depth["completed"][::-1]):
            with self.subTest(completed=completed):
                depth = copy.deepcopy(self.depth)
                depth["completed"] = copy.deepcopy(completed)
                with self.assertRaisesRegex(ValueError, "missing or reordered"):
                    self.validate(depth=depth)

    def test_failed_or_changed_source_export_cannot_bind(self):
        for field, value in (("status", "failed"), ("checkpoint_stats_unchanged", False)):
            with self.subTest(field=field):
                depth = copy.deepcopy(self.depth)
                depth[field] = value
                with self.assertRaisesRegex(ValueError, "depth source"):
                    self.validate(depth=depth)
        depth = copy.deepcopy(self.depth)
        depth["checkpoint_stats"][0]["mtime_ns"] += 1
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            self.validate(depth=depth)

    def test_formal_reduced_repeat_cli_fails_before_gpu_or_input_loading(self):
        command = [sys.executable, str(ROOT / "tools/gdmgs/benchmark_index.py"),
                   "--scene", "truck", "--model-path", str(self.model), "--mesh", str(self.mesh),
                   "--output", str(self.root / "outputs"), "--run-id", "review-fixture", "--repeats", "1"]
        result = subprocess.run(command, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 2)
        self.assertIn("require all three timing repetitions", result.stderr)
        self.assertFalse((self.root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()

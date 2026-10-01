"""CPU control tests; numerical qualification remains the remote full matrix."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
import unittest
from test_gpu_schedule_control import FakeRuntime as PriorFake, NAMESPACE as PRIOR

SOURCE = Path(__file__).resolve().parents[1] / "render_gpu_cpu_mesh_pipeline.py"
TREE = ast.parse(SOURCE.read_text())
RUNTIME = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "SceneRuntime")
RUN = next(n for n in RUNTIME.body if isinstance(n, ast.FunctionDef) and n.name == "run_mode")
DEFINITIONS = [n for n in TREE.body if
               (isinstance(n, ast.FunctionDef) and n.name in {"interval_union", "host_overlap_ms"})
               or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"PIPELINES", "MODES"} for t in n.targets))]
NAMESPACE = dict(PRIOR)
NAMESPACE.update(ThreadPoolExecutor=ThreadPoolExecutor)
exec(compile(ast.Module(body=DEFINITIONS + [RUN], type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)


class FakeRuntime(PriorFake):
    run_mode = NAMESPACE["run_mode"]
    def __init__(self, frames=32):
        super().__init__(frames)
        self.mesh_workers = 24
        self.mesh_threads = []
        self.gpu_threads = []
        self.actual_mesh_frames = []
    def _mesh(self, frame, timeline):
        self.mesh_threads.append(threading.current_thread().name)
        self.actual_mesh_frames.append(frame)
        timeline.add("mesh", "begin", frame=frame)
        timeline.add("mesh", "end", frame=frame)
        return frame, 1.0
    def select_pair_from_mesh(self, pair, mesh, timeline, submitted_ns=None):
        self.gpu_threads.append(threading.current_thread().name)
        self.asserted_mesh = tuple(mesh) == (2*pair, 2*pair+1)
        demand = super().select_pair_from_mesh(pair, mesh, timeline, submitted_ns)
        for record in demand.selection_records:
            record["mesh_workers"] = 24
        return demand
    def _render_cache_pair(self, pair, demand, cache, timeline, collect):
        self.gpu_threads.append(threading.current_thread().name)
        return super()._render_cache_pair(pair, demand, cache, timeline, collect)


class CPUMeshPipelineTests(unittest.TestCase):
    def test_full_window_bounded_mesh_cpu_only_and_gpu_calling_thread(self):
        for mode, (q, workers) in NAMESPACE["PIPELINES"].items():
            with self.subTest(mode=mode):
                runtime = FakeRuntime()
                workers = runtime.mesh_workers if workers is None else workers
                summary, outputs = runtime.run_mode(mode, collect=False)
                self.assertEqual(outputs, list(range(32)))
                self.assertEqual(sorted(runtime.actual_mesh_frames), list(range(32)))
                self.assertTrue(all(t.startswith("cpu-mesh-only") for t in runtime.mesh_threads))
                self.assertTrue(all(t == threading.current_thread().name for t in runtime.gpu_threads))
                self.assertEqual(summary["frame_count"], 32)
                self.assertEqual(summary["fallback_pairs"], [])
                pipeline = summary["mesh_pipeline"]
                self.assertEqual(pipeline["actual_mesh_workers"], workers)
                self.assertEqual(pipeline["native_query_threads"], 1)
                self.assertEqual(pipeline["max_pending_pairs"], q+1)
                for wait in pipeline["wait_records"]:
                    self.assertLessEqual(max(wait["pending_pairs"])-wait["pair"], q)
                    self.assertLessEqual(len(wait["pending_pairs"]), q+1)
                for record in summary["selection_records"]:
                    self.assertEqual(record["mesh_workers"], workers)
                    self.assertEqual(record["frozen_mesh_profile_workers"], 24)
                self.assertIsNone(runtime.pipeline_workers)
    def test_retained_window_uses_each_frozen_profile_and_rejects_short_window(self):
        mode = "gpu_cpu_mesh_pipeline_window_retained"
        for workers in (24, 32):
            runtime = FakeRuntime()
            runtime.mesh_workers = workers
            summary, outputs = runtime.run_mode(mode, collect=False)
            pipeline = summary["mesh_pipeline"]
            self.assertEqual(outputs, list(range(32)))
            self.assertEqual(pipeline["actual_mesh_workers"], workers)
            self.assertEqual(pipeline["max_pending_pairs"], 16)
            self.assertEqual(pipeline["wait_records"][0]["submitted_pairs"], list(range(16)))
            self.assertEqual(pipeline["worker_policy"], "frozen_retained_profile")
        with self.assertRaisesRegex(ValueError, "complete frozen"):
            FakeRuntime(2).run_mode(mode, collect=False)

    def test_failure_waits_and_cleans_pipeline_state(self):
        runtime = FakeRuntime()
        def failed(frame, timeline):
            raise ValueError("mesh failure")
        runtime._mesh = failed
        with self.assertRaisesRegex(ValueError, "mesh failure"):
            runtime.run_mode("gpu_cpu_mesh_pipeline_q1_pool2", collect=False)
        self.assertIsNone(runtime.pipeline_workers)
        self.assertEqual(runtime.gpu_threads, [])
    def test_overlap_uses_union_preventing_worker_double_count(self):
        records = []
        for stage, frame, begin, end in (("mesh", 0, 0, 10), ("mesh", 1, 2, 12),
                                         ("depth", 0, 3, 7), ("cache", 0, 8, 15)):
            for action, value in (("begin", begin), ("end", end)):
                records.append(dict(stage=stage, frame=frame, action=action,
                                    timestamp_ns=value*1000000, thread=str(frame)))
        self.assertEqual(NAMESPACE["host_overlap_ms"](records), 8.0)
    def test_small_development_window_still_has_one_complete_pair(self):
        result, outputs = FakeRuntime(2).run_mode("gpu_cpu_mesh_pipeline_q2_pool4", collect=True)
        self.assertEqual(outputs, [0, 1])
        self.assertEqual(result["timing_role"], "diagnostic_only")
        self.assertEqual(result["mesh_pipeline"]["max_pending_pairs"], 1)


if __name__ == "__main__":
    unittest.main()

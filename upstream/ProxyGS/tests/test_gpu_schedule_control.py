"""CPU-only control-flow tests for all seven complete-window scheduler modes.

Extract only the scheduler method so this test requires no CUDA/native imports.
The fake stages model completed pair publication; they do not claim GPU timing.
"""
import ast
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "render_step8_gpu_schedule.py"
TREE = ast.parse(SOURCE.read_text())
MODE_MAP = next(n for n in TREE.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "MODE_MAP" for t in n.targets))
RUNTIME = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "SceneRuntime")
RUN = next(n for n in RUNTIME.body if isinstance(n, ast.FunctionDef) and n.name == "run_mode")
TIMELINE = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "Timeline")
NAMESPACE = dict(threading=threading, time=time, ThreadPoolExecutor=ThreadPoolExecutor,
                 torch=SimpleNamespace(Tensor=type("Tensor", (), {}), cuda=SimpleNamespace(synchronize=lambda: None, memory_allocated=lambda: 0, reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0)),
                 percentile=lambda values, q: sorted(values)[len(values)//2])
exec(compile(ast.Module(body=[MODE_MAP, TIMELINE, RUN], type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)
NAMESPACE["MODES"] = tuple(NAMESPACE["MODE_MAP"])


class FakeRuntime:
    run_mode = NAMESPACE["run_mode"]
    def __init__(self, frames=32):
        self.args = SimpleNamespace(scene="fixture")
        self.views = list(range(frames))
        self.pending_checks = []
        self.gpu_oracle = []
        self.pair_submissions = []
        self.observed_leads = []
        self.cache_renders = []
        self.fresh_renders = []
    def new_cache(self):
        return SimpleNamespace(reset=lambda: None)
    def precompute_mesh_window(self, timeline):
        return [(f, 1.0) for f in self.views], 1.0
    def _finish_selection(self, frame, mesh, mesh_ms, timeline):
        self.assert_backend = self.cpu_baseline
        return [frame], {"frame": frame, "oracle_exact": True}
    def select_pair_from_mesh(self, pair, mesh, timeline, submitted_ns=None):
        self.pair_submissions.append(pair)
        results = [self._finish_selection(f, f, 1, timeline) for f in (2*pair, 2*pair+1)]
        return SimpleNamespace(selected_ids=[r[0] for r in results], selection_records=[r[1] for r in results])
    def _render_fresh(self, frame, ids, timeline, collect):
        self.fresh_renders.append(frame)
        return 1.0, frame, {"decoder_calls": 1}
    def _render_cache_pair(self, pair, demand, cache, timeline, collect):
        self.observed_leads.append(max(self.pair_submissions) - pair)
        frames = [pair*2, pair*2+1]
        self.cache_renders.extend(frames)
        return frames, [1.0, 1.0], [{"decoder_calls": 1}, {"decoder_calls": 0}]
    def verify_pending(self):
        return 1.0


class ScheduleControlTests(unittest.TestCase):
    def test_every_mode_preserves_full_order_and_labels(self):
        for mode in NAMESPACE["MODES"]:
            with self.subTest(mode=mode):
                runtime = FakeRuntime()
                summary, outputs = runtime.run_mode(mode, collect=False)
                self.assertEqual(summary["mode"], mode)
                self.assertEqual(summary["frame_count"], 32)
                self.assertEqual(outputs, list(range(32)))
                self.assertEqual([r["frame"] for r in summary["selection_records"]], list(range(32)))
                self.assertEqual(summary["timing_role"], "performance")
                self.assertEqual(runtime.cpu_baseline, mode.startswith("cpu_"))
                if "q2" in mode:
                    self.assertEqual(summary["schedule_lead_pairs"], 3 if mode.startswith("cpu_") else 2)
                if "fresh" not in mode:
                    self.assertEqual([r["pair"] for r in summary["deadline_records"]], list(range(16)))
    def test_two_frame_development_q2_has_no_nonexistent_second_pair(self):
        for mode in ("gpu_scheduled_cache_q2_mesh32", "cpu_scheduled_cache_q2_mesh32"):
            summary, outputs = FakeRuntime(frames=2).run_mode(mode, collect=True)
            self.assertEqual(outputs, [0, 1])
            self.assertEqual(summary["timing_role"], "diagnostic_only")
    def test_actual_submission_lead_is_two_gpu_three_historical_cpu(self):
        class ImmediateExecutor:
            def __init__(self, **kwargs):
                pass
            def submit(self, fn, *args):
                future = Future()
                future.set_result(fn(*args))
                return future
            def shutdown(self, **kwargs):
                pass
        original = NAMESPACE["ThreadPoolExecutor"]
        NAMESPACE["ThreadPoolExecutor"] = ImmediateExecutor
        try:
            for mode, bound in (("gpu_scheduled_cache_q2_mesh32", 2),
                                ("cpu_scheduled_cache_q2_mesh32", 3),
                                ("gpu_scheduled_cache_q1_mesh32", 1)):
                runtime = FakeRuntime()
                runtime.run_mode(mode, collect=False)
                self.assertEqual(max(runtime.observed_leads), bound)
                self.assertEqual(runtime.pair_submissions, list(range(16)))
        finally:
            NAMESPACE["ThreadPoolExecutor"] = original

    def test_unknown_mode_fails_closed(self):
        with self.assertRaises(ValueError):
            FakeRuntime().run_mode("old_unqualified_alias", collect=False)


if __name__ == "__main__":
    unittest.main()

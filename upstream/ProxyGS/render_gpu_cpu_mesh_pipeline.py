"""Additive bounded CPU Mesh pipeline; GPU selection and rendering stay ordered."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from render_step8_gpu_schedule import SceneRuntime as BaseRuntime, Timeline, parser as base_parser, percentile
from step4_runtime import atomic_json
from render_gdmgs_backend import _file_identity

PROTOCOL = "proxygs-gpu-cpu-mesh-pipeline-v2"
PIPELINES = {"gpu_cpu_mesh_pipeline_q1_pool2": (1, 2),
             "gpu_cpu_mesh_pipeline_q2_pool4": (2, 4),
             "gpu_cpu_mesh_pipeline_window_retained": (15, None)}
MODES = ("cpu_serial_fresh_mesh32", "gpu_serial_fresh_mesh32",
         "gpu_serial_cache_mesh32", *PIPELINES)


def interval_union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative timeline interval")
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def host_overlap_ms(records):
    """Intersect measured CPU task and GPU-stage host spans, not CUDA kernels."""
    active, mesh, gpu = {}, [], []
    for record in sorted(records, key=lambda r: r["timestamp_ns"]):
        stage = record["stage"]
        if stage not in {"mesh", "candidate", "depth", "anchor", "cache"}:
            continue
        key = (stage, record.get("frame"), record["thread"])
        if record["action"] == "begin":
            active[key] = record["timestamp_ns"]
        elif record["action"] == "end" and key in active:
            span = (active.pop(key), record["timestamp_ns"])
            (mesh if stage == "mesh" else gpu).append(span)
    return sum(max(0, min(a1, b1) - max(a0, b0))
               for a0, a1 in interval_union(mesh)
               for b0, b1 in interval_union(gpu)) / 1e6


class SceneRuntime(BaseRuntime):
    def __init__(self, args):
        super().__init__(args)
        self.pipeline_workers = None
        self.inputs["mesh_pipeline_runtime_source"] = _file_identity(Path(__file__))

    def warm_runtime(self):
        # Base run writes the common contract immediately before this hook.
        path = self.output / "contract.json"
        contract = json.loads(path.read_text())
        contract.update(protocol=PROTOCOL,
                        schedule_policy="bounded CPU Mesh futures only; GPU current-pair selection then cache render ordered",
                        warmup="first CPU mesh wait and current pair GPU selection included in wall",
                        fallback="wait for missing CPU mesh pair; no skipped frames or speculative GPU selection",
                        cpu_mesh_overlap_measurement="host interval intersection; no CUDA kernel overlap claim",
                        pipeline_modes={name: {"lookahead_pairs": q, "pool_workers": self.mesh_workers if w is None else w,
                                               "worker_policy": "frozen_retained_profile" if w is None else "fixed_pool",
                                               "future_camera_scope": "frozen_known_32_camera_window" if w is None else "bounded_frozen_window_lookahead",
                                               "native_query_threads": 1}
                                        for name, (q, w) in PIPELINES.items()})
        atomic_json(path, contract)
        return super().warm_runtime()

    def _mesh(self, frame, timeline):
        if self.pipeline_workers is None:
            return super()._mesh(frame, timeline)
        timeline.add("mesh", "begin", frame=frame, backend=self.mesh_backend,
                     workers=self.pipeline_workers, native_query_threads=1)
        started = time.perf_counter()
        result = self.mesh_index.query(self.domains[frame], backend=self.mesh_backend, threads=1)
        elapsed = (time.perf_counter() - started) * 1000.0
        if not np.array_equal(result.triangle_ids, self.oracle[frame][2]):
            raise RuntimeError(f"{self.views[frame].image_name}: pipeline Mesh IDs differ")
        timeline.add("mesh", "end", frame=frame, elapsed_ms=elapsed,
                     triangles=len(result.triangle_ids), workers=self.pipeline_workers,
                     native_query_threads=1)
        return result, elapsed

    def run_mode(self, mode, *, collect):
        if mode not in MODES:
            raise ValueError(f"unsupported mesh pipeline mode {mode}")
        if mode not in PIPELINES:
            self.pipeline_workers = None
            return super().run_mode(mode, collect=collect)
        lookahead, workers = PIPELINES[mode]
        if workers is None:
            if len(self.views) != 32:
                raise ValueError("retained-window pipeline requires the complete frozen 32-frame window")
            workers = self.mesh_workers
            if workers not in (24, 32):
                raise ValueError("retained-window worker count differs from frozen 24/32 profile")
        self.pipeline_workers = workers
        self.cpu_baseline = False
        self.pending_checks.clear()
        timeline, cache = Timeline(), self.new_cache()
        outputs, selection_records, frame_ms, cache_stats, waits = [], [], [], [], []
        allocation_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()
        pair_count = len(self.views) // 2
        pending, submitted = {}, []
        startup_selection_ready_ms = None
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cpu-mesh-only") as pool:
                def submit(pair, current):
                    if pair in pending or pair in submitted:
                        raise RuntimeError("duplicate mesh pair submission")
                    if pair > current + lookahead:
                        raise RuntimeError("mesh submission exceeds bounded lookahead")
                    pending[pair] = tuple(pool.submit(self._mesh, f, timeline)
                                          for f in (2 * pair, 2 * pair + 1))
                    submitted.append(pair)
                    timeline.add("mesh_pipeline", "submit", pair=pair, current_pair=current,
                                 pending_pairs=sorted(pending), pool_workers=workers)
                for pair in range(min(pair_count, lookahead + 1)):
                    submit(pair, 0)
                for pair in range(pair_count):
                    pending_before = sorted(pending)
                    timeline.add("mesh_wait", "begin", pair=pair)
                    started = time.perf_counter()
                    mesh_pair = [future.result() for future in pending.pop(pair)]
                    wait_ms = (time.perf_counter() - started) * 1000.0
                    timeline.add("mesh_wait", "end", pair=pair, elapsed_ms=wait_ms)
                    waits.append({"pair": pair, "wait_ms": wait_ms,
                                  "submitted_pairs": list(submitted), "pending_pairs": pending_before,
                                  "lookahead_pairs": lookahead, "pool_workers": workers})
                    # Only the calling thread performs CUDA work. This map supplies
                    # exactly the complete current pair to the inherited GPU selector.
                    mesh_values = {2 * pair + offset: value for offset, value in enumerate(mesh_pair)}
                    demand = self.select_pair_from_mesh(pair, mesh_values, timeline)
                    if pair == 0:
                        startup_selection_ready_ms = (time.perf_counter() - wall_start) * 1000.0
                    for record in demand.selection_records:
                        record["frozen_mesh_profile_workers"] = record["mesh_workers"]
                        record["mesh_workers"] = workers
                        record["mesh_native_query_threads"] = 1
                    selection_records.extend(demand.selection_records)
                    pair_outputs, pair_ms, pair_stats = self._render_cache_pair(pair, demand, cache, timeline, collect)
                    outputs.extend(pair_outputs)
                    frame_ms.extend(pair_ms)
                    cache_stats.extend(pair_stats)
                    next_pair = pair + lookahead + 1
                    if next_pair < pair_count:
                        submit(next_pair, pair + 1)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            peak_bytes = torch.cuda.max_memory_allocated()
            audit_bytes = sum(v.numel() * v.element_size()
                              for _, c, s, _ in self.pending_checks for v in (c, s))
            verification_ms = self.verify_pending()
            if (len(selection_records) != len(self.views) or len(frame_ms) != len(self.views)
                    or submitted != list(range(pair_count))):
                raise RuntimeError("mesh pipeline lost frozen frames or pairs")
            return {
                "mode": mode, "scene": self.args.scene, "frame_count": len(self.views),
                "anchor_backend": "gpu_fused", "wall_ms": wall_ms,
                "timing_role": "diagnostic_only" if collect else "performance",
                "gpu_execution": "calling thread/default stream; no GPU prefetch or kernel overlap claim",
                "verification_outside_wall_ms": verification_ms,
                "gpu_allocation_before_wall_bytes": allocation_before,
                "gpu_peak_allocated_bytes": peak_bytes, "audit_retained_id_bytes": audit_bytes,
                "gpu_oracle_resident_bytes": sum(t.numel() * t.element_size() for pair in self.gpu_oracle for t in pair),
                "frame_stage_sum_ms": float(sum(frame_ms)),
                "frame_ms_p50": percentile(frame_ms, 50), "frame_ms_p95": percentile(frame_ms, 95),
                "frame_ms_p99": percentile(frame_ms, 99),
                "selection_oracle_exact": all(r["oracle_exact"] for r in selection_records),
                "decoder_calls": sum(s.get("decoder_calls", 0) for s in cache_stats),
                "decoded_anchors": sum(s.get("decoded_anchors", 0) for s in cache_stats),
                "hit_anchors": sum(s.get("hit_anchors", 0) for s in cache_stats),
                "empty_hits": sum(s.get("empty_hits", 0) for s in cache_stats),
                "max_resident_rows": max((s.get("resident_rows", 0) for s in cache_stats), default=0),
                "max_resident_bytes": max((s.get("resident_bytes", 0) for s in cache_stats), default=0),
                "deadline_misses": 0, "deadline_exposed_wait_ms": 0.0, "fallback_pairs": [],
                "deadline_records": [{"pair": p, "ready": True, "late": False, "wait_ms": 0.0}
                                     for p in range(pair_count)],
                "selection_records": selection_records,
                "timeline": sorted(timeline.records, key=lambda r: r["timestamp_ns"]),
                "mesh_pipeline": {
                    "lookahead_pairs": lookahead, "pool_workers": workers,
                    "actual_mesh_workers": workers, "native_query_threads": 1,
                    "worker_policy": "frozen_retained_profile" if mode.endswith("window_retained") else "fixed_pool",
                    "future_camera_scope": "frozen_known_32_camera_window" if mode.endswith("window_retained") else "bounded_frozen_window_lookahead",
                    "submission_policy": "online_full_window_pairwise_consumption" if mode.endswith("window_retained") else "bounded_pair_lookahead",
                    "max_pending_pairs": max(len(w["pending_pairs"]) for w in waits),
                    "startup_mesh_wait_ms": waits[0]["wait_ms"],
                    "startup_pair_selection_ready_ms": startup_selection_ready_ms,
                    "total_mesh_wait_ms": sum(w["wait_ms"] for w in waits),
                    "wait_records": waits,
                    "cpu_mesh_gpu_stage_host_overlap_ms": host_overlap_ms(timeline.records),
                    "overlap_semantics": "host spans only; does not establish CUDA kernel overlap",
                },
            }, outputs
        finally:
            self.pipeline_workers = None


def parser():
    cli = base_parser()
    cli.description = __doc__
    action = next(a for a in cli._actions if a.dest == "modes")
    action.choices = MODES
    action.default = list(MODES)
    return cli


if __name__ == "__main__":
    args = parser().parse_args()
    if args.iteration != 40000 or (args.width, args.height) != (1600, 900):
        raise ValueError("frozen iteration/resolution changed")
    if args.performance_repeats < 3:
        raise ValueError("mesh pipeline requires at least three performance repeats")
    try:
        result = SceneRuntime(args).run()
    except Exception as exc:
        failure = args.output_root / args.scene / args.run_id
        failure.mkdir(parents=True, exist_ok=True)
        atomic_json(failure / "failure.json", {"error": repr(exc)})
        raise
    print(json.dumps({k: result[k] for k in ("status", "scene", "frame_count", "quality_pass", "performance_median_wall_ms")}, indent=2))

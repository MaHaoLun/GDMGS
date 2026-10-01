"""Step 8: real online selection plus pair2_prefix schedule validation.

Frozen Step 6 ID payloads are correctness oracles only.  Every measured mode
executes Mesh discovery, online proxy depth and the CPU Anchor query.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from argparse import Namespace
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anchor_query_runtime import query_record
from gaussian_renderer.gdmgs_gsplat_backend import render_gdmgs_backend
from gdmgs.anchor_index import AnchorPointIndex
from gdmgs.cache import CacheIdentity
from gdmgs.cache.temporal_bundle_cache_v3 import TemporalBundleCache
from gdmgs.mesh_index import MeshIndex
from gdmgs.query.nvdiffrast_compat import ensure_step8_nvdiffrast_plugin
from gdmgs.schedule import PairDemandLatch, PairIdentity
from online_proxy_depth import OnlineProxyDepthRasterizer
from render_gdmgs_backend import (
    _environment_record,
    _file_identity,
    _frozen_camera_names,
    _load_cfg,
    _new_model,
    _ordered_views,
)
from render_step6_windows import load_step5_ids, parser as step6_parser
from render_step7b_fidelity import decode_batch, image_metrics
from render_step7bc_confirmation import BUDGET, depth_metrics
from step4_runtime import DEPTH_MARGIN, atomic_json, camera_domain_from_view, file_identity


PROTOCOL = "proxygs-step8-online-pair-schedule-v1"
MODES = (
    "serial_fresh",
    "serial_fresh_mesh32",
    "serial_cache",
    "scheduled_cache",
    "serial_cache_mesh32",
    "scheduled_cache_q2_mesh32",
)
CAPACITY_ROWS = 6_826_846


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def percentile(values, q):
    return float(np.percentile(list(values), q)) if values else None


class Timeline:
    def __init__(self):
        self.origin_ns = time.perf_counter_ns()
        self.records = []
        self.lock = threading.Lock()

    def add(self, stage: str, action: str, **fields):
        item = {
            "timestamp_ns": time.perf_counter_ns() - self.origin_ns,
            "thread": threading.current_thread().name,
            "stage": stage,
            "action": action,
            **fields,
        }
        with self.lock:
            self.records.append(item)
        return item["timestamp_ns"]


class SceneRuntime:
    def __init__(self, args: Namespace):
        self.args = args
        native_dir = args.mesh_native_dir.resolve()
        native_files = list(native_dir.glob("GDMGS_mesh_native*.so"))
        if len(native_files) != 1:
            raise ValueError("Step 8 requires exactly one retained Mesh native extension")
        # MeshIndex imports lazily.  Override render_step6_windows.py's legacy
        # default before the first load/query so Barcelona uses Retained v2's
        # guarded fast predicates and density-aware output ordering.
        os.environ["GDMGS_NATIVE_DIR"] = str(native_dir)
        self.mesh_native_file = native_files[0]
        self.output = (args.output_root / args.scene / args.run_id).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "command.txt").write_text(" ".join(map(str, sys.argv)) + "\n")
        atomic_json(self.output / "status.json", {"state": "starting", "scene": args.scene})

        self.selected_contract = load(args.selected_cache_contract)
        if not (
            self.selected_contract.get("status") == "qualified"
            and self.selected_contract.get("selected_policy") == "pair2_prefix"
            and self.selected_contract.get("refresh_period") == 2
            and self.selected_contract.get("max_source_age") == 1
            and self.selected_contract.get("payload_dtype") == "float32"
            and self.selected_contract.get("capacity_rows") == CAPACITY_ROWS
            and self.selected_contract.get("lookahead")
            == "next selected request set must be ready BEFORE refresh decode"
        ):
            raise ValueError("Step 7 selected cache contract is not the frozen pair2_prefix result")
        confirmation = load(args.step7_confirmation)
        if not (
            confirmation.get("status") == "pass"
            and confirmation.get("scene_count") == 8
            and confirmation.get("frame_count") == 256
            and confirmation.get("selected_policy") == "pair2_prefix"
            and confirmation.get("failures") == []
        ):
            raise ValueError("Step 7 final confirmation is not passing")

        cfg = _load_cfg(args.model_path.resolve())
        if Path(cfg.source_path).resolve() != args.source_path.resolve():
            raise ValueError("model/source binding differs")
        cfg.data_device = "cpu"
        self.model = _new_model(cfg)
        from scene import Scene

        scene = Scene(
            cfg,
            self.model,
            load_iteration=args.iteration,
            shuffle=False,
            resolution_scales=cfg.resolution_scales,
        )
        self.model.eval()
        all_views = _ordered_views(scene, _frozen_camera_names(args.model_path.resolve()))
        if len(all_views) != args.expected_views:
            raise ValueError("full-scene camera denominator changed")

        selection = load(args.selection)
        if not (
            selection.get("status") == "frozen"
            and selection.get("selection_performed_once") is True
            and selection.get("window_count") == 8
            and selection.get("frame_count") == 256
        ):
            raise ValueError("Step 6 windows are not frozen")
        windows = [w for w in selection["windows"] if w["scene"] == args.scene]
        if len(windows) != 1:
            raise ValueError("scene must have exactly one frozen window")
        self.window = windows[0]
        start = int(self.window["start_index"])
        stop = int(self.window["end_index_inclusive"]) + 1
        self.views = all_views[start:stop]
        if len(self.views) != 32 or [v.image_name for v in self.views] != self.window["camera_ids"]:
            raise ValueError("camera order differs from frozen Step 6 window")
        if args.max_views is not None:
            if args.formal:
                raise ValueError("formal Step 8 cannot use max_views")
            if args.max_views < 2 or args.max_views % 2:
                raise ValueError("development max_views must be an even number >=2")
            self.views = self.views[: args.max_views]

        trace = load(args.cache_trace)
        scenes = [s for s in trace["scenes"] if s.get("scene") == args.scene]
        if len(scenes) != 1:
            raise ValueError("cache trace must contain exactly one scene record")
        self.trace = scenes[0]
        if self.trace["camera_ids"][: len(self.views)] != [v.image_name for v in self.views]:
            raise ValueError("cache trace camera order differs from online views")
        self.payloads = self.trace["id_payloads"][: len(self.views)]

        profile = load(args.mesh_window_profile)
        if not (
            profile.get("status") == "frozen_from_qualification"
            and profile.get("minimum_index_speedup_to_retain") == 1.2
            and profile.get("selection_changed_windows") is False
        ):
            raise ValueError("Retained v2 profile is not frozen")
        self.scene_profile = profile["scene_profiles"][args.scene]
        self.mesh_backend = self.scene_profile["backend"]
        self.mesh_workers = int(self.scene_profile["workers"])
        if self.mesh_backend not in {"brute_force", "optimized_bvh"}:
            raise ValueError("unsupported Retained v2 backend")

        self.mesh_index = MeshIndex.load(args.mesh_index.resolve(), mesh_token=args.mesh_token)
        anchor_source = file_identity(
            args.model_path.resolve() / "point_cloud/iteration_40000/point_cloud.ply"
        )
        anchor_source["iteration"] = 40000
        self.anchor_index = AnchorPointIndex.load(
            args.anchor_index.resolve(), scene=args.scene, source=anchor_source
        )
        ensure_step8_nvdiffrast_plugin()
        self.rasterizer = OnlineProxyDepthRasterizer(self.mesh_index, device="cuda:0")
        self.levels = self.model.get_level.detach().view(-1).long().contiguous()
        self.background = torch.tensor(
            [1.0, 1.0, 1.0] if cfg.white_background else [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device="cuda",
        )
        self.domains = [camera_domain_from_view(v) for v in self.views]
        self.oracle = []
        for payload in self.payloads:
            candidates, selected, triangles, anchor_count, triangle_count = load_step5_ids(
                Path(payload["path"])
            )
            if anchor_count != self.levels.numel() or triangle_count != len(self.mesh_index.triangles):
                raise ValueError("oracle row universe changed")
            self.oracle.append((candidates, selected, triangles))

        self.metric = None
        self.inputs = {
            "model_cfg": _file_identity(args.model_path / "cfg_args"),
            "point_cloud": _file_identity(args.model_path / "point_cloud/iteration_40000/point_cloud.ply"),
            "mesh_index": file_identity(args.mesh_index),
            "mesh_native": file_identity(self.mesh_native_file),
            "anchor_index": file_identity(args.anchor_index),
            "window_selection": file_identity(args.selection),
            "retained_profile": file_identity(args.mesh_window_profile),
            "cache_trace": _file_identity(args.cache_trace),
            "selected_cache_contract": _file_identity(args.selected_cache_contract),
            "step7_confirmation": _file_identity(args.step7_confirmation),
        }
        self.cache_identity = CacheIdentity(
            args.scene,
            str(args.model_path),
            "gdmgs-gsplat-v1-step8",
            str(args.model_path / "point_cloud/iteration_40000/point_cloud.ply"),
            str(args.cache_trace),
        )

    def new_cache(self):
        return TemporalBundleCache(
            identity=self.cache_identity,
            anchor_levels=self.levels,
            capacity_rows=CAPACITY_ROWS,
            n_offsets=self.model.n_offsets,
            max_age=1,
            fast_handoff=True,
            prefix_decode=True,
        )

    def _mesh(self, frame: int, timeline: Timeline):
        timeline.add("mesh", "begin", frame=frame, backend=self.mesh_backend, workers=self.mesh_workers)
        start = time.perf_counter()
        result = self.mesh_index.query(self.domains[frame], backend=self.mesh_backend, threads=1)
        elapsed = (time.perf_counter() - start) * 1000.0
        if not np.array_equal(result.triangle_ids, self.oracle[frame][2]):
            raise RuntimeError(f"{self.views[frame].image_name}: online Mesh IDs differ from oracle")
        timeline.add("mesh", "end", frame=frame, elapsed_ms=elapsed, triangles=len(result.triangle_ids))
        return result, elapsed

    def _finish_selection(self, frame: int, mesh_result, mesh_ms: float, timeline: Timeline):
        view = self.views[frame]
        # The frozen nvdiffrast context was qualified on the default stream.
        # Keep that stream contract; Step 8 overlaps CPU Mesh work and records
        # the resulting GPU serialization instead of moving the context to an
        # unqualified stream.
        with torch.no_grad():
            timeline.add("candidate", "begin", frame=frame)
            torch.cuda.synchronize()
            started = time.perf_counter()
            self.model.set_anchor_mask(view.camera_center, 40000, view.resolution_scale)
            candidates = (
                torch.nonzero(self.model._anchor_mask, as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            torch.cuda.synchronize()
            candidate_ms = (time.perf_counter() - started) * 1000.0
            timeline.add("candidate", "end", frame=frame, elapsed_ms=candidate_ms, count=len(candidates))
            if not np.array_equal(candidates, self.oracle[frame][0]):
                raise RuntimeError(f"{view.image_name}: online candidate IDs differ from oracle")

            timeline.add("depth", "begin", frame=frame)
            depth = self.rasterizer.render(
                mesh_result.triangle_ids, self.domains[frame], (self.args.width, self.args.height)
            )
            timeline.add("depth", "end", frame=frame, **depth.timings)

        world_view = np.ascontiguousarray(view.world_view_transform.detach().cpu().numpy(), dtype=np.float32)
        projection = np.ascontiguousarray(view.full_proj_transform.detach().cpu().numpy(), dtype=np.float32)
        timeline.add("anchor", "begin", frame=frame)
        anchor = self.anchor_index.query(
            candidates,
            depth.depth_cpu,
            world_view,
            projection,
            mode="tree",
            camera=view.image_name,
            margin=float(DEPTH_MARGIN),
            trusted_buffers=True,
            reuse_buffers=True,
        )
        selected = anchor.selected_anchor_ids
        timeline.add(
            "anchor",
            "end",
            frame=frame,
            elapsed_ms=anchor.timings["anchor_index_total_ms"],
            selected_count=len(selected),
        )
        if not np.array_equal(selected, self.oracle[frame][1]):
            raise RuntimeError(f"{view.image_name}: online selected IDs differ from oracle")
        return selected, {
            "frame": frame,
            "camera": view.image_name,
            "mesh_backend": self.mesh_backend,
            "mesh_workers": self.mesh_workers,
            "mesh_ms": mesh_ms,
            "candidate_ms": candidate_ms,
            "depth": depth.timings,
            "anchor": query_record(anchor),
            "selected_count": int(len(selected)),
            "selected_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
            "oracle_exact": True,
        }

    def select_frame(self, frame: int, timeline: Timeline):
        mesh, mesh_ms = self._mesh(frame, timeline)
        return self._finish_selection(frame, mesh, mesh_ms, timeline)

    def select_pair(self, pair: int, timeline: Timeline, submitted_ns: int | None = None):
        frames = (2 * pair, 2 * pair + 1)
        ident = PairIdentity(
            self.args.scene,
            pair,
            frames,
            (self.views[frames[0]].image_name, self.views[frames[1]].image_name),
        )
        latch = PairDemandLatch(ident, submitted_ns=submitted_ns)
        timeline.add("pair_selection", "begin", pair=pair, frames=list(frames))
        try:
            with ThreadPoolExecutor(max_workers=min(2, self.mesh_workers), thread_name_prefix="mesh") as pool:
                mesh_values = list(pool.map(lambda f: self._mesh(f, timeline), frames))
            results = [
                self._finish_selection(frame, mesh, mesh_ms, timeline)
                for frame, (mesh, mesh_ms) in zip(frames, mesh_values)
            ]
            demand = latch.publish(
                tuple(value[0] for value in results), tuple(value[1] for value in results)
            )
            timeline.add(
                "pair_selection",
                "ready",
                pair=pair,
                elapsed_ms=(demand.ready_ns - demand.submitted_ns) / 1e6,
            )
            return demand
        except Exception as exc:
            latch.fail(exc)
            timeline.add("pair_selection", "failed", pair=pair, error=repr(exc))
            raise

    def precompute_mesh_window(self, timeline: Timeline):
        """Execute the frozen Retained-v2 32-camera Mesh batch online."""
        timeline.add(
            "mesh_window_batch",
            "begin",
            frames=len(self.views),
            workers=self.mesh_workers,
            backend=self.mesh_backend,
        )
        started = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=min(self.mesh_workers, len(self.views)), thread_name_prefix="mesh-window"
        ) as pool:
            values = list(pool.map(lambda frame: self._mesh(frame, timeline), range(len(self.views))))
        elapsed = (time.perf_counter() - started) * 1000.0
        timeline.add("mesh_window_batch", "end", elapsed_ms=elapsed, frames=len(values))
        return values, elapsed

    def select_pair_from_mesh(
        self, pair: int, mesh_values, timeline: Timeline, submitted_ns: int | None = None
    ):
        frames = (2 * pair, 2 * pair + 1)
        ident = PairIdentity(
            self.args.scene,
            pair,
            frames,
            (self.views[frames[0]].image_name, self.views[frames[1]].image_name),
        )
        latch = PairDemandLatch(ident, submitted_ns=submitted_ns)
        timeline.add("pair_selection_mesh_ready", "begin", pair=pair, frames=list(frames))
        try:
            results = [
                self._finish_selection(frame, mesh, mesh_ms, timeline)
                for frame, (mesh, mesh_ms) in zip(frames, (mesh_values[f] for f in frames))
            ]
            demand = latch.publish(
                tuple(value[0] for value in results), tuple(value[1] for value in results)
            )
            timeline.add(
                "pair_selection_mesh_ready",
                "ready",
                pair=pair,
                elapsed_ms=(demand.ready_ns - demand.submitted_ns) / 1e6,
            )
            return demand
        except Exception as exc:
            latch.fail(exc)
            timeline.add("pair_selection_mesh_ready", "failed", pair=pair, error=repr(exc))
            raise

    def _render_fresh(self, frame, ids, timeline, collect):
        view = self.views[frame]
        timeline.add("fresh", "begin", frame=frame, selected_count=len(ids))
        torch.cuda.synchronize()
        start = time.perf_counter()
        ids_gpu = torch.from_numpy(ids.copy()).cuda()
        batch = decode_batch(view, self.model, ids_gpu, self.levels[ids_gpu])
        output = render_gdmgs_backend(view, batch, self.background, "RGB")
        image = torch.clamp(output["render"], 0.0, 1.0)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        timeline.add("fresh", "end", frame=frame, elapsed_ms=elapsed, rows=int(batch.xyz.shape[0]))
        diagnostics = None
        if collect:
            ed = render_gdmgs_backend(view, batch, self.background, "RGB+ED")
            diagnostics = {
                "image": image.detach().cpu(),
                "render_depth": ed["render_depth"].detach().cpu(),
                "render_alpha": ed["render_alpha"].detach().cpu(),
            }
        return elapsed, diagnostics, {
            "decoder_calls": 1,
            "decoded_anchors": int(len(ids)),
            "selected_anchors": int(len(ids)),
            "hit_anchors": 0,
            "empty_hits": 0,
            "source_age": 0,
            "resident_rows": 0,
            "resident_bytes": 0,
        }

    def _render_cache_pair(self, pair, demand, cache, timeline, collect):
        outputs, frame_ms, stats_records = [], [], []
        for offset, ids in enumerate(demand.selected_ids):
            frame = 2 * pair + offset
            view = self.views[frame]
            next_ids = (
                torch.from_numpy(demand.selected_ids[1].copy()).cuda() if offset == 0 else None
            )
            ids_gpu = torch.from_numpy(ids.copy()).cuda()
            timeline.add("cache", "begin", frame=frame, pair=pair, refresh=offset == 0)
            torch.cuda.synchronize()
            started = time.perf_counter()
            batch, stats = cache.resolve(
                frame_id=frame,
                anchor_ids=ids_gpu,
                next_ids=next_ids,
                decode=lambda request_ids, levels: decode_batch(view, self.model, request_ids, levels),
            )
            output = render_gdmgs_backend(view, batch, self.background, "RGB")
            image = torch.clamp(output["render"], 0.0, 1.0)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
            timeline.add("cache", "end", frame=frame, pair=pair, elapsed_ms=elapsed, **stats)
            diagnostic = None
            if collect:
                ed = render_gdmgs_backend(view, batch, self.background, "RGB+ED")
                diagnostic = {
                    "image": image.detach().cpu(),
                    "render_depth": ed["render_depth"].detach().cpu(),
                    "render_alpha": ed["render_alpha"].detach().cpu(),
                }
            outputs.append(diagnostic)
            frame_ms.append(elapsed)
            stats_records.append(stats)
        return outputs, frame_ms, stats_records

    def run_mode(self, mode: str, *, collect: bool):
        if mode not in MODES:
            raise ValueError(f"unsupported mode: {mode}")
        timeline = Timeline()
        cache = self.new_cache()
        outputs = []
        selection_records = []
        frame_ms = []
        cache_stats = []
        deadline_records = []
        wall_start = time.perf_counter()

        if mode in ("serial_fresh", "serial_fresh_mesh32"):
            mesh_values = None
            if mode == "serial_fresh_mesh32":
                mesh_values, _ = self.precompute_mesh_window(timeline)
            for frame in range(len(self.views)):
                ids, record = (
                    self.select_frame(frame, timeline)
                    if mesh_values is None
                    else self._finish_selection(
                        frame, mesh_values[frame][0], mesh_values[frame][1], timeline
                    )
                )
                elapsed, output, stats = self._render_fresh(frame, ids, timeline, collect)
                selection_records.append(record)
                frame_ms.append(elapsed)
                outputs.append(output)
                cache_stats.append(stats)
        elif mode in ("serial_cache", "serial_cache_mesh32"):
            mesh_values = None
            if mode == "serial_cache_mesh32":
                mesh_values, _ = self.precompute_mesh_window(timeline)
            for pair in range(len(self.views) // 2):
                demand = (
                    self.select_pair(pair, timeline)
                    if mesh_values is None
                    else self.select_pair_from_mesh(pair, mesh_values, timeline)
                )
                selection_records.extend(demand.selection_records)
                pair_outputs, pair_ms, pair_stats = self._render_cache_pair(
                    pair, demand, cache, timeline, collect
                )
                outputs.extend(pair_outputs)
                frame_ms.extend(pair_ms)
                cache_stats.extend(pair_stats)
                deadline_records.append({"pair": pair, "ready": True, "late": False, "wait_ms": 0.0})
        elif mode == "scheduled_cache":
            pair_count = len(self.views) // 2
            timeline.add("warmup", "begin", pair=0)
            current = self.select_pair(0, timeline)
            timeline.add("warmup", "end", pair=0)
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lookahead")
            next_future: Future | None = None
            try:
                for pair in range(pair_count):
                    late = False
                    wait_ms = 0.0
                    if pair:
                        deadline_ns = timeline.add("deadline", "check", pair=pair)
                        if next_future is None:
                            raise RuntimeError("scheduled pair is missing its lookahead future")
                        late = not next_future.done()
                        wait_start = time.perf_counter()
                        current = next_future.result()
                        wait_ms = (time.perf_counter() - wait_start) * 1000.0
                        timeline.add(
                            "deadline",
                            "resolved",
                            pair=pair,
                            late=late,
                            exposed_wait_ms=wait_ms,
                            deadline_ns=deadline_ns,
                        )
                    selection_records.extend(current.selection_records)
                    if pair + 1 < pair_count:
                        submitted_ns = time.perf_counter_ns()
                        timeline.add("lookahead", "submit", pair=pair + 1)
                        next_future = executor.submit(
                            self.select_pair, pair + 1, timeline, submitted_ns
                        )
                    else:
                        next_future = None

                    if late:
                        cache.reset()
                        pair_outputs, pair_ms, pair_stats = [], [], []
                        for offset, ids in enumerate(current.selected_ids):
                            frame = 2 * pair + offset
                            elapsed, output, stats = self._render_fresh(
                                frame, ids, timeline, collect
                            )
                            pair_outputs.append(output)
                            pair_ms.append(elapsed)
                            pair_stats.append({**stats, "fallback": "late_pair_lookahead"})
                    else:
                        pair_outputs, pair_ms, pair_stats = self._render_cache_pair(
                            pair, current, cache, timeline, collect
                        )
                    outputs.extend(pair_outputs)
                    frame_ms.extend(pair_ms)
                    cache_stats.extend(pair_stats)
                    deadline_records.append(
                        {"pair": pair, "ready": not late, "late": late, "wait_ms": wait_ms}
                    )
            finally:
                executor.shutdown(wait=True, cancel_futures=False)
        else:
            # Retained-v2 Mesh discovery is executed as the same 32-camera
            # batch qualified in Step 6.  Two complete pair demands are warmed
            # before rendering; thereafter each pair gets two pair intervals
            # of lookahead without extending cache source age beyond one.
            pair_count = len(self.views) // 2
            mesh_values, _ = self.precompute_mesh_window(timeline)
            timeline.add("warmup_q2", "begin", pairs=[0, 1])
            ready = {
                0: self.select_pair_from_mesh(0, mesh_values, timeline),
                1: self.select_pair_from_mesh(1, mesh_values, timeline),
            }
            timeline.add("warmup_q2", "end", pairs=[0, 1])
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lookahead-q2")
            futures = {}
            if pair_count > 2:
                submitted_ns = time.perf_counter_ns()
                futures[2] = executor.submit(
                    self.select_pair_from_mesh, 2, mesh_values, timeline, submitted_ns
                )
            try:
                for pair in range(pair_count):
                    late = False
                    wait_ms = 0.0
                    if pair in ready:
                        current = ready.pop(pair)
                    else:
                        future = futures.pop(pair)
                        deadline_ns = timeline.add("deadline_q2", "check", pair=pair)
                        late = not future.done()
                        wait_start = time.perf_counter()
                        current = future.result()
                        wait_ms = (time.perf_counter() - wait_start) * 1000.0
                        timeline.add(
                            "deadline_q2",
                            "resolved",
                            pair=pair,
                            late=late,
                            exposed_wait_ms=wait_ms,
                            deadline_ns=deadline_ns,
                        )
                    selection_records.extend(current.selection_records)
                    submit_pair = pair + 3
                    if submit_pair < pair_count and submit_pair not in futures:
                        submitted_ns = time.perf_counter_ns()
                        timeline.add("lookahead_q2", "submit", pair=submit_pair)
                        futures[submit_pair] = executor.submit(
                            self.select_pair_from_mesh,
                            submit_pair,
                            mesh_values,
                            timeline,
                            submitted_ns,
                        )
                    if late:
                        cache.reset()
                        pair_outputs, pair_ms, pair_stats = [], [], []
                        for offset, ids in enumerate(current.selected_ids):
                            frame = 2 * pair + offset
                            elapsed, output, stats = self._render_fresh(
                                frame, ids, timeline, collect
                            )
                            pair_outputs.append(output)
                            pair_ms.append(elapsed)
                            pair_stats.append({**stats, "fallback": "late_q2_lookahead"})
                    else:
                        pair_outputs, pair_ms, pair_stats = self._render_cache_pair(
                            pair, current, cache, timeline, collect
                        )
                    outputs.extend(pair_outputs)
                    frame_ms.extend(pair_ms)
                    cache_stats.extend(pair_stats)
                    deadline_records.append(
                        {"pair": pair, "ready": not late, "late": late, "wait_ms": wait_ms}
                    )
            finally:
                executor.shutdown(wait=True, cancel_futures=False)

        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        if len(selection_records) != len(self.views) or len(frame_ms) != len(self.views):
            raise RuntimeError("mode did not process the complete frozen denominator")
        summary = {
            "mode": mode,
            "scene": self.args.scene,
            "frame_count": len(self.views),
            "wall_ms": wall_ms,
            "frame_stage_sum_ms": float(sum(frame_ms)),
            "frame_ms_p50": percentile(frame_ms, 50),
            "frame_ms_p95": percentile(frame_ms, 95),
            "frame_ms_p99": percentile(frame_ms, 99),
            "selection_oracle_exact": all(r["oracle_exact"] for r in selection_records),
            "decoder_calls": int(sum(s.get("decoder_calls", 0) for s in cache_stats)),
            "decoded_anchors": int(sum(s.get("decoded_anchors", 0) for s in cache_stats)),
            "hit_anchors": int(sum(s.get("hit_anchors", 0) for s in cache_stats)),
            "empty_hits": int(sum(s.get("empty_hits", 0) for s in cache_stats)),
            "max_resident_rows": int(max((s.get("resident_rows", 0) for s in cache_stats), default=0)),
            "max_resident_bytes": int(max((s.get("resident_bytes", 0) for s in cache_stats), default=0)),
            "deadline_misses": int(sum(r["late"] for r in deadline_records)),
            "deadline_exposed_wait_ms": float(sum(r["wait_ms"] for r in deadline_records)),
            "fallback_pairs": [r["pair"] for r in deadline_records if r["late"]],
            "selection_records": selection_records,
            "deadline_records": deadline_records,
            "timeline": sorted(timeline.records, key=lambda r: r["timestamp_ns"]),
        }
        return summary, outputs

    def quality(self, candidate_outputs, fresh_outputs):
        if self.metric is None:
            import lpips

            self.metric = lpips.LPIPS(net="vgg").cuda().eval()
        records = []
        for frame, (candidate, reference) in enumerate(zip(candidate_outputs, fresh_outputs)):
            if candidate is None or reference is None:
                raise ValueError("quality outputs were not collected")
            cand_image = candidate["image"].cuda()
            ref_image = reference["image"].cuda()
            gt = self.views[frame].original_image.cuda().clamp(0.0, 1.0)
            metrics = image_metrics(cand_image, ref_image, gt, self.metric)
            depth = depth_metrics(
                {
                    "render_depth": candidate["render_depth"].cuda(),
                    "render_alpha": candidate["render_alpha"].cuda(),
                },
                {
                    "render_depth": reference["render_depth"].cuda(),
                    "render_alpha": reference["render_alpha"].cuda(),
                },
            )
            records.append({"frame": frame, "camera": self.views[frame].image_name, "metrics": metrics, "depth": depth})
        losses = {}
        for name in ("psnr", "ssim", "lpips"):
            values = np.asarray(
                [
                    (-r["metrics"]["reuse_minus_fresh_gt"][name] if name != "lpips"
                     else r["metrics"]["reuse_minus_fresh_gt"][name])
                    for r in records
                ]
            )
            losses[name] = {
                "mean_loss": float(values.mean()),
                "worst_loss": float(values.max()),
                "worst_frame": int(values.argmax()),
                "passed": bool(
                    np.isfinite(values).all()
                    and values.mean() <= BUDGET[f"{name}_mean"]
                    and values.max() <= BUDGET[f"{name}_worst"]
                ),
            }
        rel = np.asarray([r["depth"]["relative_mae"] for r in records])
        depth_report = {
            "relative_mae_mean": float(rel.mean()),
            "relative_mae_worst": float(rel.max()),
            "passed": bool(np.isfinite(rel).all() and rel.mean() <= 0.01 and rel.max() <= 0.05),
        }
        return {
            "rgb": losses,
            "depth": depth_report,
            "passed": all(v["passed"] for v in losses.values()) and depth_report["passed"],
            "records": records,
        }

    def warm_runtime(self):
        """Warm compiled kernels and allocators outside formal frame timing."""
        timeline = Timeline()
        demand = self.select_pair(0, timeline)
        self._render_fresh(0, demand.selected_ids[0], timeline, collect=False)
        self._render_cache_pair(0, demand, self.new_cache(), timeline, collect=False)
        torch.cuda.synchronize()

    def run(self):
        contract = {
            "protocol": PROTOCOL,
            "scene": self.args.scene,
            "formal": self.args.formal,
            "camera_ids": [v.image_name for v in self.views],
            "modes": list(self.args.modes),
            "retained_profile": self.scene_profile,
            "cache_policy": self.selected_contract,
            "quality_budget": BUDGET,
            "depth_budget": {"relative_mae_mean": 0.01, "relative_mae_worst": 0.05},
            "oracle_role": "correctness only; never supplied as candidate runtime selected IDs",
            "warmup": "S0/S1 online selection is inside scheduled wall time",
            "fallback": "late pair lookahead uses fresh current-pose decode for the complete pair",
            "environment": _environment_record(),
            "inputs": self.inputs,
        }
        atomic_json(self.output / "contract.json", contract)
        self.warm_runtime()
        atomic_json(self.output / "status.json", {"state": "qualification"})

        qualification = {}
        fresh_mode = (
            "serial_fresh_mesh32"
            if "serial_fresh_mesh32" in self.args.modes
            else "serial_fresh"
        )
        fresh_summary, fresh_outputs = self.run_mode(fresh_mode, collect=True)
        qualification[fresh_mode] = fresh_summary
        candidate_modes = [
            mode for mode in self.args.modes if not mode.startswith("serial_fresh")
        ]
        for mode in candidate_modes:
            summary, outputs = self.run_mode(mode, collect=True)
            summary["quality"] = self.quality(outputs, fresh_outputs)
            qualification[mode] = summary
        atomic_json(self.output / "qualification.json", qualification)

        atomic_json(self.output / "status.json", {"state": "performance"})
        performance = {mode: [] for mode in self.args.modes}
        for repeat in range(self.args.performance_repeats):
            order = list(self.args.modes)
            if repeat % 2:
                order.reverse()
            for mode in order:
                summary, _ = self.run_mode(mode, collect=False)
                performance[mode].append(summary)
                atomic_json(self.output / "performance.json", performance)

        medians = {
            mode: float(np.median([r["wall_ms"] for r in records]))
            for mode, records in performance.items()
        }
        final = {
            "status": "pass",
            "scene": self.args.scene,
            "frame_count": len(self.views),
            "qualification": qualification,
            "performance_median_wall_ms": medians,
            "speedups": {
                f"{fresh_mode}_to_{mode}": medians[fresh_mode] / medians[mode]
                for mode in candidate_modes
            },
            "quality_pass": all(
                qualification[m]["quality"]["passed"] for m in candidate_modes
            ),
            "selection_oracle_exact": all(
                qualification[m]["selection_oracle_exact"] for m in qualification
            ),
        }
        if not final["selection_oracle_exact"] or (self.args.formal and not final["quality_pass"]):
            final["status"] = "failed"
        final["development_quality_gate_applied"] = bool(self.args.formal)
        atomic_json(self.output / "summary.json", final)
        atomic_json(self.output / "status.json", {"state": "complete", "status": final["status"]})
        return final


def parser():
    result = step6_parser()
    result.description = __doc__
    result.add_argument("--cache-trace", type=Path, required=True)
    result.add_argument("--selected-cache-contract", type=Path, required=True)
    result.add_argument("--step7-confirmation", type=Path, required=True)
    result.add_argument("--mesh-native-dir", type=Path, required=True)
    result.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    result.add_argument("--performance-repeats", type=int, default=3)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if args.iteration != 40000 or args.width != 1600 or args.height != 900:
        raise ValueError("Step 8 freezes iteration=40000 and 1600x900")
    if args.performance_repeats < 1:
        raise ValueError("performance repeats must be positive")
    try:
        report = SceneRuntime(args).run()
    except Exception as exc:
        failure = args.output_root / args.scene / args.run_id
        failure.mkdir(parents=True, exist_ok=True)
        atomic_json(failure / "failure.json", {"error": repr(exc)})
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "scene": report["scene"],
                "frame_count": report["frame_count"],
                "quality_pass": report["quality_pass"],
                "selection_oracle_exact": report["selection_oracle_exact"],
                "performance_median_wall_ms": report["performance_median_wall_ms"],
                "speedups": report["speedups"],
            },
            indent=2,
            sort_keys=True,
        )
    )

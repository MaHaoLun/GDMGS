"""All-GPU Mesh/anchor fixed-pair2 qualification and paired ablation.

This is an additive experiment. The copied Step8 runtime and frozen inputs
remain versioned alongside this file. All measured modes use the same GPU
Mesh -> GPU depth -> GPU anchor selection producer and default CUDA stream.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import render_step8_gpu_schedule as base
from gaussian_renderer.gdmgs_gsplat_backend import render_gdmgs_backend
from gdmgs.mesh_index.gpu_index import GPUMeshIndex
from gdmgs.schedule.gpu_request_contract import GPUPairDemandLatch
from gdmgs.schedule import PairIdentity
from render_step7b_fidelity import decode_batch
from step7d_execution_optimizations import GPUSetPlanner, certify_pair, planned_refresh
from step7d_gpu_merge import bitmap_refresh_plan


MODES = (
    "cpu_fresh_reference", "gpu_fresh", "gpu_pair2_control", "gpu_pair2_B",
    "gpu_pair2_AB", "gpu_pair2_bitmap",
)
PERFORMANCE_MODES = ("gpu_fresh", "gpu_pair2_control", "gpu_pair2_B",
                     "gpu_pair2_AB", "gpu_pair2_bitmap")
base.MODES = MODES
base.MODE_MAP = {
    "cpu_fresh_reference": "serial_fresh",
    "gpu_fresh": "serial_fresh",
    "gpu_pair2_control": "serial_cache",
    "gpu_pair2_B": "serial_cache",
    "gpu_pair2_AB": "serial_cache",
    "gpu_pair2_bitmap": "serial_cache",
}
BATCH_FIELDS = ("xyz", "color", "opacity", "scaling", "rotation", "anchor_indices", "selection_mask")
META_FIELDS = ("request_anchor_ids", "request_level_ids", "counts", "offsets",
               "row_owner_ids", "row_owner_levels", "row_offset_slots")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def exact_batch(left, right):
    if left.sh_degree != right.sh_degree:
        return False, "sh_degree"
    for field in BATCH_FIELDS:
        if not torch.equal(getattr(left, field), getattr(right, field)):
            return False, field
    for field in META_FIELDS:
        if not torch.equal(getattr(left.bundle_metadata, field), getattr(right.bundle_metadata, field)):
            return False, "metadata." + field
    return True, None


class SceneRuntime(base.SceneRuntime):
    def __init__(self, args):
        super().__init__(args)
        self.gpu_mesh_index = GPUMeshIndex(self.mesh_index)
        self.gpu_mesh_oracle = [torch.as_tensor(triangles.copy(), device="cuda", dtype=torch.int64)
                                for _, _, triangles in self.oracle]
        self.pending_mesh_checks = []
        self.last_mesh_check_count = 0
        self.active_mode = None
        self.control_batches = []
        self.exact_checks = []
        self.gpu_index_source = Path(__file__).parent / "gdmgs/mesh_index/gpu_index.py"
        self.inputs["gpu_mesh_index_source"] = base._file_identity(self.gpu_index_source)
        self.inputs["gpu_mesh_cuda_source"] = base._file_identity(
            self.gpu_index_source.parent / "native/gpu_query.cu")
        self.inputs["gpu_depth_source"] = base._file_identity(Path(__file__).parent / "online_proxy_depth.py")
        self.inputs["step7d_AB_source"] = base._file_identity(
            Path(__file__).parent / "step7d_execution_optimizations.py")
        self.inputs["step7d_bitmap_source"] = base._file_identity(
            Path(__file__).parent / "step7d_gpu_merge.py")
        self.inputs["this_source"] = base._file_identity(Path(__file__))
        if len(self.views) != 32 and args.formal:
            raise ValueError("formal runs require the full 32-frame window")

    def _mesh(self, frame, timeline):
        if self.cpu_baseline:
            return base.SceneRuntime._mesh(self, frame, timeline)
        timeline.add("mesh", "begin", frame=frame, backend="gpu_bvh_leaf")
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = self.gpu_mesh_index.query(self.domains[frame], backend="bvh")
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        self.pending_mesh_checks.append((frame, result))
        timeline.add("mesh", "end", frame=frame, elapsed_ms=elapsed,
                     triangles=int(result.triangle_ids.numel()), backend="gpu_bvh_leaf")
        return result, elapsed

    def verify_pending(self):
        started = time.perf_counter()
        super().verify_pending()
        self.last_mesh_check_count = len(self.pending_mesh_checks)
        for frame, result in self.pending_mesh_checks:
            result.validate()
            if not torch.equal(result.triangle_ids, self.gpu_mesh_oracle[frame]):
                raise RuntimeError(f"GPU Mesh IDs differ at frame {frame}")
        self.pending_mesh_checks.clear()
        return (time.perf_counter() - started) * 1000.0

    def select_pair(self, pair, timeline, submitted_ns=None):
        """GPU Mesh calls run serially on the qualified default stream."""
        frames = (2 * pair, 2 * pair + 1)
        ident = PairIdentity(self.args.scene, pair, frames,
                             (self.views[frames[0]].image_name, self.views[frames[1]].image_name))
        latch = GPUPairDemandLatch(ident, submitted_ns=submitted_ns)
        timeline.add("pair_selection", "begin", pair=pair, frames=list(frames))
        try:
            results = [self.select_frame(frame, timeline) for frame in frames]
            demand = latch.publish(tuple(value[0] for value in results),
                                   tuple(value[1] for value in results))
            timeline.add("pair_selection", "ready", pair=pair,
                         elapsed_ms=(demand.ready_ns - demand.submitted_ns) / 1e6)
            return demand
        except Exception as exc:
            latch.fail(exc)
            timeline.add("pair_selection", "failed", pair=pair, error=repr(exc))
            raise

    @base.gpu_guard
    def _render_cache_pair(self, pair, demand, cache, timeline, collect):
        outputs, frame_ms, stats_records = [], [], []
        certificate = None
        plan = None
        if self.active_mode == "gpu_pair2_AB":
            plan = GPUSetPlanner(list(demand.selected_ids)).refresh_plan(2)
        elif self.active_mode == "gpu_pair2_bitmap":
            plan = bitmap_refresh_plan(demand.selected_ids[0], demand.selected_ids[1],
                                       self.gpu_anchor_index.anchor_count)
        for offset, ids in enumerate(demand.selected_ids):
            frame = 2 * pair + offset
            view = self.views[frame]
            timeline.add("cache", "begin", frame=frame, pair=pair, refresh=offset == 0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            if not (offset and certificate is not None):
                if self.model.dist2level == "progressive":
                    self.model.set_anchor_mask(view.camera_center, 40000, view.resolution_scale)
                decoder = lambda request_ids, levels: decode_batch(view, self.model, request_ids, levels)
                if plan is not None and offset == 0:
                    batch, stats = planned_refresh(cache, frame, plan, decoder)
                else:
                    batch, stats = cache.resolve(frame_id=frame, anchor_ids=ids,
                                                 next_ids=demand.selected_ids[1] if offset == 0 else None,
                                                 decode=decoder)
                if self.active_mode != "gpu_pair2_control" and offset == 0:
                    certificate = certify_pair(cache, frame, demand.selected_ids[1],
                                               int(demand.selected_ids[1].numel()), stats)
            else:
                batch, stats = certificate.consume(cache, frame, ids)
            output = render_gdmgs_backend(view, batch, self.background, "RGB")
            image = torch.clamp(output["render"], 0.0, 1.0)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            timeline.add("cache", "end", frame=frame, pair=pair, elapsed_ms=elapsed,
                         certified_hit=bool(offset and certificate is not None), **stats)
            if collect:
                if self.active_mode == "gpu_pair2_control":
                    self.control_batches.append(batch)
                else:
                    expected = self.control_batches[frame]
                    same, field = exact_batch(batch, expected)
                    self.exact_checks.append({"frame": frame, "passed": bool(same), "field": field})
                    if not same:
                        raise RuntimeError(f"batch mismatch at frame {frame}: {field}")
                ed = render_gdmgs_backend(view, batch, self.background, "RGB+ED")
                diagnostic = {"image": image.detach().cpu(),
                              "render_depth": ed["render_depth"].detach().cpu(),
                              "render_alpha": ed["render_alpha"].detach().cpu()}
            else:
                diagnostic = None
            outputs.append(diagnostic)
            frame_ms.append(elapsed)
            stats_records.append(stats)
        return outputs, frame_ms, stats_records

    def run_mode(self, mode, *, collect):
        self.active_mode = mode
        self.pending_mesh_checks.clear()
        if collect and mode == "gpu_pair2_control":
            self.control_batches = []
        if collect and mode not in ("cpu_fresh_reference", "gpu_fresh", "gpu_pair2_control"):
            self.exact_checks = []
        summary, outputs = base.SceneRuntime.run_mode(self, mode, collect=collect)
        summary["mesh_backend"] = "cpu_retained" if mode.startswith("cpu_") else "gpu_bvh_leaf"
        summary["mesh_oracle_exact"] = (mode.startswith("cpu_") or
                                        self.last_mesh_check_count == len(self.views))
        if not summary["mesh_oracle_exact"]:
            raise RuntimeError("GPU Mesh audit did not cover every frame")
        if collect and mode not in ("cpu_fresh_reference", "gpu_fresh", "gpu_pair2_control"):
            summary["payload_metadata_exact"] = len(self.exact_checks) == len(self.views) and all(
                check["passed"] for check in self.exact_checks)
            summary["payload_metadata_checks"] = list(self.exact_checks)
        return summary, outputs


def gpu_sample():
    index = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not index.isdigit():
        raise ValueError("select exactly one physical GPU index")
    uuid = subprocess.check_output(["nvidia-smi", "-i", index, "--query-gpu=uuid",
                                    "--format=csv,noheader"], text=True).strip()
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader"], text=True)
    pids = [int(parts[1]) for line in raw.splitlines()
            if len(parts := [part.strip() for part in line.split(",")]) >= 2 and parts[0] == uuid]
    return {"unix": time.time(), "gpu_index": int(index), "uuid": uuid,
            "pids": pids, "own_pid": os.getpid(), "foreign_pids": [p for p in pids if p != os.getpid()]}


def main():
    parser = base.parser()
    args = parser.parse_args()
    if args.iteration != 40000 or args.width != 1600 or args.height != 900:
        raise ValueError("frozen model/resolution changed")
    if args.performance_repeats < 3:
        raise ValueError("at least three repeats required")
    runtime = SceneRuntime(args)
    out = runtime.output
    contract = {"schema": "step7d_pair2_all_gpu_ablation_v1", "status": "running",
                "scene": args.scene, "camera_ids": [v.image_name for v in runtime.views],
                "mode_order": list(MODES), "performance_modes": list(PERFORMANCE_MODES),
                "performance_repeats": args.performance_repeats,
                "gpu_mesh_backend": "bvh_leaf_cluster", "gpu_anchor_backend": "fused",
                "cache_max_age": 1, "cache_capacity_rows": base.CAPACITY_ROWS,
                "stream": "default_serial", "inputs": runtime.inputs}
    atomic_json(out / "contract.json", contract)
    runtime.warm_runtime()
    qualification = {}
    reference_summary, reference = runtime.run_mode("cpu_fresh_reference", collect=True)
    qualification["cpu_fresh_reference"] = reference_summary
    fresh_summary, fresh = runtime.run_mode("gpu_fresh", collect=True)
    fresh_summary["fresh_exact"] = {
        "passed": len(fresh) == len(reference) and all(
            all(torch.equal(a[key], b[key]) for key in ("image", "render_depth", "render_alpha"))
            for a, b in zip(fresh, reference)),
        "frames": len(fresh),
    }
    if not fresh_summary["fresh_exact"]["passed"]:
        raise RuntimeError("GPU fresh differs from CPU fresh")
    qualification["gpu_fresh"] = fresh_summary
    for mode in ("gpu_pair2_control", "gpu_pair2_B", "gpu_pair2_AB", "gpu_pair2_bitmap"):
        summary, outputs = runtime.run_mode(mode, collect=True)
        summary["quality"] = runtime.quality(outputs, reference)
        if not summary["quality"]["passed"]:
            raise RuntimeError(mode + " failed frozen RGB/depth gates")
        if summary["decoder_calls"] != 16 or summary["frame_count"] != 32:
            raise RuntimeError(mode + " changed fixed pair2 work count")
        if mode != "gpu_pair2_control" and not summary["payload_metadata_exact"]:
            raise RuntimeError(mode + " payload/metadata differs")
        qualification[mode] = summary
    for mode in ("gpu_pair2_B", "gpu_pair2_AB", "gpu_pair2_bitmap"):
        for left, right in zip(qualification["gpu_pair2_control"]["quality"]["records"],
                               qualification[mode]["quality"]["records"]):
            if left["frame"] != right["frame"]:
                raise RuntimeError("quality record frame order differs")
    atomic_json(out / "qualification.json", qualification)
    del reference, fresh, runtime.control_batches
    runtime.control_batches = []
    torch.cuda.empty_cache()
    performance = {mode: [] for mode in PERFORMANCE_MODES}
    samples = []
    for repeat in range(args.performance_repeats):
        order = list(PERFORMANCE_MODES)
        rotation = repeat % len(order)
        order = order[rotation:] + order[:rotation]
        for mode in order:
            before = gpu_sample()
            if before["foreign_pids"]:
                raise RuntimeError("foreign process on measured GPU: " + str(before))
            summary, _ = runtime.run_mode(mode, collect=False)
            after = gpu_sample()
            if before["uuid"] != after["uuid"] or after["foreign_pids"]:
                raise RuntimeError("GPU identity/isolation changed")
            summary["gpu_sample_before"] = before
            summary["gpu_sample_after"] = after
            performance[mode].append(summary)
            samples.append({"repeat": repeat, "mode": mode, "before": before, "after": after})
            atomic_json(out / "performance.json", performance)
            print(args.scene, repeat + 1, mode, round(summary["wall_ms"], 3), flush=True)
    medians = {mode: float(np.median([record["wall_ms"] for record in records]))
               for mode, records in performance.items()}
    result = {"status": "pass", "scene": args.scene, "frames": len(runtime.views),
              "qualification": {mode: {"selection_oracle_exact": summary["selection_oracle_exact"],
                                       "mesh_oracle_exact": summary["mesh_oracle_exact"],
                                       "wall_ms_diagnostic": summary["wall_ms"],
                                       "decoder_calls": summary["decoder_calls"],
                                       "quality_pass": summary.get("quality", {}).get("passed"),
                                       "fresh_exact": summary.get("fresh_exact", {}).get("passed"),
                                       "payload_metadata_exact": summary.get("payload_metadata_exact")}
                                for mode, summary in qualification.items()},
              "median_wall_ms": medians,
              "pair2_speedups": {mode: medians["gpu_pair2_control"] / medians[mode]
                                  for mode in ("gpu_pair2_B", "gpu_pair2_AB", "gpu_pair2_bitmap")},
              "gpu_samples": samples}
    atomic_json(out / "summary.json", result)
    contract["status"] = "complete"
    atomic_json(out / "contract.json", contract)
    print(json.dumps({"scene": args.scene, "median_wall_ms": medians,
                      "pair2_speedups": result["pair2_speedups"]}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(repr(error), file=sys.stderr, flush=True)
        raise

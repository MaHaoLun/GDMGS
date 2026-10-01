"""All-GPU complete-window experiment for a two-frame sliding bundle cache."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import render_step8_gpu_schedule as base
import render_step7d_pair2_gpu as pair2
from gaussian_renderer.gdmgs_gsplat_backend import render_gdmgs_backend
from gdmgs.cache.sliding_window_cache import MODES as SLIDING_MODES, SlidingWindowCache
from render_step7b_fidelity import decode_batch


MODES = ("gpu_fresh", "pair2_control", "slide_lazy_age1",
         "slide_eager_age1", "slide_lazy_age2")
PERFORMANCE_MODES = MODES
base.MODES = ("cpu_fresh_reference", "gpu_fresh", "gpu_pair2_control",
              "pair2_control") + tuple(SLIDING_MODES)
base.MODE_MAP.update({name: "serial_cache" for name in SLIDING_MODES})


def run_sliding(runtime, mode, *, collect):
    if mode not in SLIDING_MODES:
        raise ValueError("unknown sliding mode")
    runtime.cpu_baseline = False
    runtime.pending_checks.clear()
    runtime.pending_mesh_checks.clear()
    runtime.last_mesh_check_count = 0
    timeline = base.Timeline()
    cache = SlidingWindowCache(
        identity=runtime.cache_identity,
        anchor_levels=runtime.levels,
        capacity_rows=base.CAPACITY_ROWS,
        n_offsets=runtime.model.n_offsets,
        mode=mode,
        audit=collect,
    )
    selected = {}
    outputs = []
    frame_ms = []
    cache_stats = []
    torch.cuda.reset_peak_memory_stats()

    def get_selected(frame):
        if frame not in selected:
            selected[frame] = runtime.select_frame(frame, timeline)
        return selected[frame]

    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for frame, view in enumerate(runtime.views):
        frame_start = time.perf_counter()
        ids, _ = get_selected(frame)
        next_ids = (get_selected(frame + 1)[0]
                    if SLIDING_MODES[mode].eager_next and frame + 1 < len(runtime.views)
                    else None)
        timeline.add("sliding_cache", "begin", frame=frame, mode=mode)
        with runtime.gpu_lock, torch.cuda.stream(torch.cuda.default_stream()), torch.no_grad():
            if runtime.model.dist2level == "progressive":
                runtime.model.set_anchor_mask(view.camera_center, 40000, view.resolution_scale)
            batch, stats = cache.resolve(
                frame_id=frame, anchor_ids=ids, next_ids=next_ids,
                decode=lambda request_ids, levels: decode_batch(
                    view, runtime.model, request_ids, levels),
            )
            rgb = render_gdmgs_backend(view, batch, runtime.background, "RGB")
            image = torch.clamp(rgb["render"], 0.0, 1.0)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - frame_start) * 1000
            timeline.add("sliding_cache", "end", frame=frame, mode=mode,
                         elapsed_ms=elapsed, **stats)
            if collect:
                ed = render_gdmgs_backend(view, batch, runtime.background, "RGB+ED")
                outputs.append({
                    "image": image.detach().cpu(),
                    "render_depth": ed["render_depth"].detach().cpu(),
                    "render_alpha": ed["render_alpha"].detach().cpu(),
                })
            else:
                outputs.append(None)
        frame_ms.append(elapsed)
        cache_stats.append(stats)
        if stats["source_age_max"] > SLIDING_MODES[mode].max_age:
            raise RuntimeError("sliding source age exceeded mode contract")
        if stats["resident_rows"] > base.CAPACITY_ROWS or stats["resident_generations"] > 2:
            raise RuntimeError("sliding cache exceeded bounded residency")
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    peak_bytes = torch.cuda.max_memory_allocated()
    verification_ms = runtime.verify_pending()
    selection_records = [selected[frame][1] for frame in range(len(runtime.views))]
    if len(selection_records) != 32 or runtime.last_mesh_check_count != 32:
        raise RuntimeError("sliding run did not verify the full 32-frame producer")
    if not all(record["oracle_exact"] for record in selection_records):
        raise RuntimeError("sliding anchor selection differs from frozen oracle")
    summary = {
        "mode": mode, "scene": runtime.args.scene, "frame_count": len(runtime.views),
        "wall_ms": wall_ms,
        "timing_role": "diagnostic_only" if collect else "performance",
        "gpu_execution": "default stream, serial GPU stages",
        "mesh_backend": "gpu_bvh_leaf", "anchor_backend": "gpu_fused",
        "selection_oracle_exact": True, "mesh_oracle_exact": True,
        "verification_outside_wall_ms": verification_ms,
        "decoder_calls": sum(s["decoder_calls"] for s in cache_stats),
        "decoded_anchors": sum(s["decoded_anchors"] for s in cache_stats),
        "hit_anchors": sum(s["hit_anchors"] for s in cache_stats),
        "hit_rows": sum(s["hit_rows"] for s in cache_stats),
        "hit_age1_anchors": sum(s["hit_age1_anchors"] for s in cache_stats),
        "hit_age2_anchors": sum(s["hit_age2_anchors"] for s in cache_stats),
        "hit_age1_rows": sum(s["hit_age1_rows"] for s in cache_stats),
        "hit_age2_rows": sum(s["hit_age2_rows"] for s in cache_stats),
        "miss_anchors": sum(s["miss_anchors"] for s in cache_stats),
        "prefetch_anchors": sum(s["prefetch_anchors"] for s in cache_stats),
        "evicted_rows": sum(s["evicted_rows"] for s in cache_stats),
        "evicted_generations": sum(s["evicted_generations"] for s in cache_stats),
        "max_source_age": max(s["source_age_max"] for s in cache_stats),
        "max_resident_rows": max(s["resident_rows"] for s in cache_stats),
        "max_resident_bytes": max(s["resident_bytes"] for s in cache_stats),
        "gpu_peak_allocated_bytes": peak_bytes,
        "first_frame_latency_ms": frame_ms[0],
        "full_frame_ms_p50": float(np.percentile(frame_ms, 50)),
        "full_frame_ms_p95": float(np.percentile(frame_ms, 95)),
        "full_frame_ms_p99": float(np.percentile(frame_ms, 99)),
        "frame_stats": cache_stats,
        "selection_records": selection_records,
        "timeline": sorted(timeline.records, key=lambda record: record["timestamp_ns"]),
    }
    return summary, outputs


def run_mode(runtime, mode, *, collect):
    if mode in SLIDING_MODES:
        return run_sliding(runtime, mode, collect=collect)
    baseline = "gpu_pair2_control" if mode == "pair2_control" else mode
    summary, outputs = runtime.run_mode(baseline, collect=collect)
    summary["mode"] = mode
    first_begin = next((x["timestamp_ns"] for x in summary["timeline"]
                        if x["stage"] == "mesh" and x["action"] == "begin"), None)
    first_stage = "cache" if mode == "pair2_control" else "fresh"
    first_end = next((x["timestamp_ns"] for x in summary["timeline"]
                      if x["stage"] == first_stage and x["action"] == "end"
                      and x.get("frame") == 0), None)
    if first_begin is not None and first_end is not None:
        summary["first_frame_latency_ms_diagnostic"] = (first_end - first_begin) / 1e6
    return summary, outputs


def main():
    args = base.parser().parse_args()
    if args.iteration != 40000 or args.width != 1600 or args.height != 900:
        raise ValueError("frozen model or render resolution changed")
    if args.performance_repeats < 3 or args.max_views is not None:
        raise ValueError("sliding formal/qualification requires full 32 frames and >=3 repeats")
    runtime = pair2.SceneRuntime(args)
    out = runtime.output
    contract = {
        "schema": "step7d_sliding_window2_all_gpu_v1", "status": "running",
        "scene": args.scene, "camera_ids": [view.image_name for view in runtime.views],
        "modes": list(MODES), "performance_repeats": args.performance_repeats,
        "capacity_rows": base.CAPACITY_ROWS, "payload_dtype": "float32",
        "max_resident_generations": 2,
        "default_max_source_age": 1, "age2_role": "quality-gated independent ablation",
        "producer": "GPU Mesh -> GPU depth -> GPU fused anchor index",
        "stream": "default_serial", "inputs": runtime.inputs,
        "sliding_source": base._file_identity(
            Path(__file__).parent / "gdmgs/cache/sliding_window_cache.py"),
        "runner_source": base._file_identity(Path(__file__)),
    }
    pair2.atomic_json(out / "contract.json", contract)
    runtime.warm_runtime()
    qualification = {}
    cpu_summary, cpu_outputs = runtime.run_mode("cpu_fresh_reference", collect=True)
    qualification["cpu_fresh_reference"] = cpu_summary
    fresh_summary, fresh_outputs = run_mode(runtime, "gpu_fresh", collect=True)
    fresh_summary["fresh_exact"] = {
        "passed": len(cpu_outputs) == len(fresh_outputs) == 32 and all(
            all(torch.equal(a[key], b[key]) for key in
                ("image", "render_depth", "render_alpha"))
            for a, b in zip(cpu_outputs, fresh_outputs)),
        "frames": 32,
    }
    if not fresh_summary["fresh_exact"]["passed"]:
        raise RuntimeError("GPU fresh output differs from CPU fresh")
    qualification["gpu_fresh"] = fresh_summary
    for mode in MODES[1:]:
        summary, outputs = run_mode(runtime, mode, collect=True)
        summary["quality"] = runtime.quality(outputs, fresh_outputs)
        if mode == "pair2_control" and not summary["quality"]["passed"]:
            raise RuntimeError("same-run pair2 quality baseline failed")
        if mode in SLIDING_MODES and not torch.equal(
                outputs[0]["image"], fresh_outputs[0]["image"]):
            raise RuntimeError(mode + " first frame must be current-pose exact")
        qualification[mode] = summary
        pair2.atomic_json(out / "qualification_partial.json", qualification)
        print(args.scene, "quality", mode, summary["quality"]["passed"],
              "decode_calls", summary["decoder_calls"], flush=True)
    pair2.atomic_json(out / "qualification.json", qualification)
    del cpu_outputs, fresh_outputs
    torch.cuda.empty_cache()
    performance = {mode: [] for mode in MODES}
    for repeat in range(args.performance_repeats):
        order = list(MODES)
        shift = repeat % len(order)
        order = order[shift:] + order[:shift]
        for mode in order:
            before = pair2.gpu_sample()
            if before["foreign_pids"]:
                raise RuntimeError("foreign process on measured GPU: " + str(before))
            summary, _ = run_mode(runtime, mode, collect=False)
            after = pair2.gpu_sample()
            if before["uuid"] != after["uuid"] or after["foreign_pids"]:
                raise RuntimeError("GPU identity/isolation changed")
            summary["gpu_sample_before"] = before
            summary["gpu_sample_after"] = after
            performance[mode].append(summary)
            pair2.atomic_json(out / "performance.json", performance)
            print(args.scene, "repeat", repeat + 1, mode,
                  round(summary["wall_ms"], 3), flush=True)
    medians = {mode: float(np.median([r["wall_ms"] for r in performance[mode]]))
               for mode in MODES}
    result = {
        "status": "complete", "scene": args.scene, "frames": len(runtime.views),
        "qualification": {mode: {
            "selection_oracle_exact": q["selection_oracle_exact"],
            "mesh_oracle_exact": q["mesh_oracle_exact"],
            "quality_pass": q.get("quality", {}).get("passed"),
            "fresh_exact": q.get("fresh_exact", {}).get("passed"),
            "decoder_calls": q["decoder_calls"],
            "max_source_age": q.get("max_source_age"),
        } for mode, q in qualification.items()},
        "median_wall_ms": medians,
    }
    pair2.atomic_json(out / "summary.json", result)
    contract["status"] = "complete"
    pair2.atomic_json(out / "contract.json", contract)
    print(json.dumps({"scene": args.scene, "median_wall_ms": medians}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(repr(error), file=sys.stderr, flush=True)
        raise

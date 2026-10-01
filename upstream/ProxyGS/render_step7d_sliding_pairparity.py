"""All-GPU, exact-call-count sliding pair cache ablation."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import render_step8_gpu_schedule as base
import render_step7d_pair2_gpu as pair2
from gaussian_renderer.gdmgs_gsplat_backend import render_gdmgs_backend
from gdmgs.cache.sliding_pair_parity_cache import SlidingPairParityCache
from render_step7b_fidelity import decode_batch


MODES = ("gpu_fresh", "pair2_control", "slide_pair_age1", "slide_pair_age2")
base.MODES = ("cpu_fresh_reference", "gpu_fresh", "gpu_pair2_control") + MODES[1:]
base.MODE_MAP.update({"pair2_control": "serial_cache",
                      "slide_pair_age1": "serial_cache",
                      "slide_pair_age2": "serial_cache"})


def run_parity(runtime, mode, *, collect):
    max_age = 1 if mode == "slide_pair_age1" else 2
    runtime.cpu_baseline = False
    runtime.pending_checks.clear()
    runtime.pending_mesh_checks.clear()
    runtime.last_mesh_check_count = 0
    cache = SlidingPairParityCache(
        identity=runtime.cache_identity, anchor_levels=runtime.levels,
        capacity_rows=base.CAPACITY_ROWS, n_offsets=runtime.model.n_offsets,
        max_age=max_age, audit=collect)
    timeline = base.Timeline()
    outputs, records, frame_stats, frame_ms, exact = [], [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for pair in range(len(runtime.views) // 2):
        demand = runtime.select_pair(pair, timeline)
        records.extend(demand.selection_records)
        for offset, ids in enumerate(demand.selected_ids):
            frame = 2 * pair + offset
            view = runtime.views[frame]
            timeline.add("parity_cache", "begin", frame=frame, pair=pair,
                         refresh=offset == 0, mode=mode)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with runtime.gpu_lock, torch.cuda.stream(torch.cuda.default_stream()), torch.no_grad():
                if runtime.model.dist2level == "progressive":
                    runtime.model.set_anchor_mask(view.camera_center, 40000, view.resolution_scale)
                decoder = lambda request_ids, levels: decode_batch(
                    view, runtime.model, request_ids, levels)
                if offset == 0:
                    batch, stats = cache.refresh(
                        frame_id=frame, anchor_ids=ids,
                        next_ids=demand.selected_ids[1], decode=decoder)
                else:
                    batch, stats = cache.consume(
                        frame_id=frame, anchor_ids=ids, decode=decoder)
                rgb = render_gdmgs_backend(view, batch, runtime.background, "RGB")
                image = torch.clamp(rgb["render"], 0.0, 1.0)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                timeline.add("parity_cache", "end", frame=frame, pair=pair,
                             elapsed_ms=elapsed, **stats)
                if collect:
                    if mode == "slide_pair_age1":
                        same, field = pair2.exact_batch(batch, runtime.control_batches[frame])
                        exact.append({"frame": frame, "passed": bool(same), "field": field})
                        if not same:
                            raise RuntimeError("age-one pair cache differs from pair2 at frame "
                                               + str(frame) + ": " + str(field))
                    ed = render_gdmgs_backend(view, batch, runtime.background, "RGB+ED")
                    outputs.append({
                        "image": image.detach().cpu(),
                        "render_depth": ed["render_depth"].detach().cpu(),
                        "render_alpha": ed["render_alpha"].detach().cpu(),
                    })
                else:
                    outputs.append(None)
            frame_stats.append(stats)
            frame_ms.append(elapsed)
            if stats["resident_rows"] > base.CAPACITY_ROWS:
                raise RuntimeError("pair cache resident-row cap exceeded")
            if stats["source_age_max"] > max_age:
                raise RuntimeError("pair cache source age exceeded")
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    peak_bytes = torch.cuda.max_memory_allocated()
    verification_ms = runtime.verify_pending()
    if len(records) != 32 or runtime.last_mesh_check_count != 32 \
            or not all(record["oracle_exact"] for record in records):
        raise RuntimeError("full GPU producer exactness/coverage failed")
    calls = sum(s["decoder_calls"] for s in frame_stats)
    if calls != 16:
        raise RuntimeError("sliding pair mode broke 16-call scene parity: " + str(calls))
    summary = {
        "mode": mode, "scene": runtime.args.scene, "frame_count": 32,
        "wall_ms": wall_ms,
        "timing_role": "diagnostic_only" if collect else "performance",
        "gpu_execution": "default stream, serialized GPU stages",
        "mesh_backend": "gpu_bvh_leaf", "anchor_backend": "gpu_fused",
        "selection_oracle_exact": True, "mesh_oracle_exact": True,
        "verification_outside_wall_ms": verification_ms,
        "decoder_calls": calls,
        "decoded_anchors": sum(s["decoded_anchors"] for s in frame_stats),
        "hit_anchors": sum(s["hit_anchors"] for s in frame_stats),
        "hit_age2_anchors": sum(s["hit_age2_anchors"] for s in frame_stats),
        "hit_age2_rows": sum(s["hit_age2_rows"] for s in frame_stats),
        "miss_anchors": sum(s["miss_anchors"] for s in frame_stats),
        "prefetch_anchors": sum(s["prefetch_anchors"] for s in frame_stats),
        "evicted_anchors": sum(s["evicted_anchors"] for s in frame_stats),
        "max_source_age": max(s["source_age_max"] for s in frame_stats),
        "max_resident_rows": max(s["resident_rows"] for s in frame_stats),
        "max_resident_bytes": max(s["resident_bytes"] for s in frame_stats),
        "gpu_peak_allocated_bytes": peak_bytes,
        "frame_ms_p95": float(np.percentile(frame_ms, 95)),
        "frame_stats": frame_stats,
        "selection_records": records,
        "timeline": sorted(timeline.records, key=lambda event: event["timestamp_ns"]),
    }
    if collect and mode == "slide_pair_age1":
        summary["payload_metadata_exact"] = len(exact) == 32 and all(x["passed"] for x in exact)
        summary["payload_metadata_checks"] = exact
    return summary, outputs


def run_mode(runtime, mode, *, collect):
    if mode.startswith("slide_pair_"):
        return run_parity(runtime, mode, collect=collect)
    baseline = "gpu_pair2_control" if mode == "pair2_control" else mode
    summary, outputs = runtime.run_mode(baseline, collect=collect)
    summary["mode"] = mode
    return summary, outputs


def main():
    args = base.parser().parse_args()
    if args.iteration != 40000 or args.width != 1600 or args.height != 900:
        raise ValueError("frozen model/render resolution changed")
    if args.performance_repeats < 4 or args.max_views is not None:
        raise ValueError("full 32 frames and >=4 balanced repeats required")
    runtime = pair2.SceneRuntime(args)
    out = runtime.output
    contract = {
        "schema": "step7d_sliding_pair_parity_v1", "status": "running",
        "scene": args.scene, "camera_ids": [view.image_name for view in runtime.views],
        "modes": list(MODES), "performance_repeats": args.performance_repeats,
        "capacity_rows": base.CAPACITY_ROWS, "payload_dtype": "float32",
        "decoder_calls_required_per_32": 16,
        "age_one_expected_exact_pair2": True,
        "age_two_role": "independent quality-gated cross-pair reuse",
        "producer": "GPU Mesh -> GPU depth -> GPU fused anchor index",
        "stream": "default_serial", "inputs": runtime.inputs,
        "pair_parity_source": base._file_identity(
            Path(__file__).parent / "gdmgs/cache/sliding_pair_parity_cache.py"),
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
        raise RuntimeError("GPU fresh differs from CPU fresh")
    qualification["gpu_fresh"] = fresh_summary
    control_summary, control_outputs = run_mode(runtime, "pair2_control", collect=True)
    control_summary["quality"] = runtime.quality(control_outputs, fresh_outputs)
    if not control_summary["quality"]["passed"] or control_summary["decoder_calls"] != 16:
        raise RuntimeError("same-run pair2 baseline failed")
    qualification["pair2_control"] = control_summary
    for mode in MODES[2:]:
        summary, outputs = run_mode(runtime, mode, collect=True)
        summary["quality"] = runtime.quality(outputs, fresh_outputs)
        if mode == "slide_pair_age1" and not summary["payload_metadata_exact"]:
            raise RuntimeError("age-one parity did not match pair2")
        qualification[mode] = summary
        pair2.atomic_json(out / "qualification_partial.json", qualification)
        print(args.scene, "quality", mode, summary["quality"]["passed"],
              "calls", summary["decoder_calls"], flush=True)
    pair2.atomic_json(out / "qualification.json", qualification)
    del cpu_outputs, fresh_outputs, control_outputs
    runtime.control_batches = []
    torch.cuda.empty_cache()
    performance = {mode: [] for mode in MODES}
    for repeat in range(args.performance_repeats):
        order = list(MODES)
        shift = repeat % len(order)
        order = order[shift:] + order[:shift]
        for mode in order:
            before = pair2.gpu_sample()
            if before["foreign_pids"]:
                raise RuntimeError("foreign same-GPU process: " + str(before))
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
            "payload_metadata_exact": q.get("payload_metadata_exact"),
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

"""Independent aggregate review for the Step 8 formal matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


SCENES = ("amsterdam", "barcelona", "bilbao", "chicago", "hollywood", "pompidou", "quebec", "rome")
MODES = (
    "serial_fresh_mesh32",
    "serial_cache_mesh32",
    "scheduled_cache_q2_mesh32",
)
CACHE_MODES = ("serial_cache_mesh32", "scheduled_cache_q2_mesh32")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs-root", type=Path, required=True)
    cli.add_argument("--run-id", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    failures, summaries, schedule_timing = [], {}, {}
    for scene in SCENES:
        path = args.runs_root / scene / args.run_id / "summary.json"
        if not path.is_file():
            failures.append(f"{scene}: missing summary")
            continue
        value = json.loads(path.read_text())
        summaries[scene] = value
        performance_path = args.runs_root / scene / args.run_id / "performance.json"
        if not performance_path.is_file():
            failures.append(f"{scene}: missing performance records")
        else:
            performance = json.loads(performance_path.read_text())
            records = performance.get("scheduled_cache_q2_mesh32", [])
            warmups, steadies, mesh_batches = [], [], []
            for record in records:
                warmup_events = [
                    event
                    for event in record.get("timeline", [])
                    if event.get("stage") == "warmup_q2" and event.get("action") == "end"
                ]
                mesh_events = [
                    event
                    for event in record.get("timeline", [])
                    if event.get("stage") == "mesh_window_batch" and event.get("action") == "end"
                ]
                if len(warmup_events) != 1 or len(mesh_events) != 1:
                    failures.append(f"{scene}: incomplete warm-up timeline")
                    continue
                warmup_ms = warmup_events[0]["timestamp_ns"] / 1e6
                warmups.append(warmup_ms)
                steadies.append(record["wall_ms"] - warmup_ms)
                mesh_batches.append(mesh_events[0]["elapsed_ms"])
            if warmups:
                schedule_timing[scene] = {
                    "warmup_ms_median": float(np.median(warmups)),
                    "steady_state_ms_median": float(np.median(steadies)),
                    "mesh_window_batch_ms_median": float(np.median(mesh_batches)),
                    "repeat_count": len(warmups),
                }
        if value.get("status") != "pass":
            failures.append(f"{scene}: status is not pass")
        if value.get("frame_count") != 32:
            failures.append(f"{scene}: frame denominator differs")
        if value.get("selection_oracle_exact") is not True:
            failures.append(f"{scene}: online selection differs from oracle")
        if value.get("quality_pass") is not True:
            failures.append(f"{scene}: Step 7 quality gate failed")

    total_wall = {mode: 0.0 for mode in MODES}
    deadline_misses = 0
    fallback_pairs = 0
    decoder_calls = {mode: 0 for mode in MODES}
    decoded_anchors = {mode: 0 for mode in MODES}
    quality_values = {
        mode: {metric: [] for metric in ("psnr", "ssim", "lpips")}
        for mode in CACHE_MODES
    }
    depth_values = {mode: [] for mode in CACHE_MODES}
    for summary in summaries.values():
        for mode in MODES:
            records = summary["performance_median_wall_ms"]
            total_wall[mode] += records[mode]
            qualification = summary["qualification"][mode]
            decoder_calls[mode] += qualification["decoder_calls"]
            decoded_anchors[mode] += qualification["decoded_anchors"]
        scheduled = summary["qualification"]["scheduled_cache_q2_mesh32"]
        deadline_misses += scheduled["deadline_misses"]
        fallback_pairs += len(scheduled["fallback_pairs"])
        for mode in CACHE_MODES:
            q = summary["qualification"][mode]["quality"]
            for record in q["records"]:
                delta = record["metrics"]["reuse_minus_fresh_gt"]
                quality_values[mode]["psnr"].append(-delta["psnr"])
                quality_values[mode]["ssim"].append(-delta["ssim"])
                quality_values[mode]["lpips"].append(delta["lpips"])
                depth_values[mode].append(record["depth"]["relative_mae"])

    aggregate_quality = {}
    for mode in CACHE_MODES:
        aggregate_quality[mode] = {
            metric: {
                "mean_loss": float(np.mean(values)),
                "worst_loss": float(np.max(values)),
            }
            for metric, values in quality_values[mode].items()
        } if quality_values[mode]["psnr"] else {}
        if depth_values[mode]:
            aggregate_quality[mode]["depth"] = {
                "relative_mae_mean": float(np.mean(depth_values[mode])),
                "relative_mae_worst": float(np.max(depth_values[mode])),
            }

    review = {
        "status": "pass" if len(summaries) == 8 and not failures else "failed",
        "scene_count": len(summaries),
        "frame_count": 32 * len(summaries),
        "failures": failures,
        "paired_wall_ms": total_wall,
        "speedups": {
            "fresh_to_serial_cache": total_wall["serial_fresh_mesh32"] / total_wall["serial_cache_mesh32"] if total_wall["serial_cache_mesh32"] else None,
            "fresh_to_scheduled_cache": total_wall["serial_fresh_mesh32"] / total_wall["scheduled_cache_q2_mesh32"] if total_wall["scheduled_cache_q2_mesh32"] else None,
            "serial_cache_to_scheduled_cache": total_wall["serial_cache_mesh32"] / total_wall["scheduled_cache_q2_mesh32"] if total_wall["scheduled_cache_q2_mesh32"] else None,
        },
        "deadline_misses": deadline_misses,
        "fallback_pairs": fallback_pairs,
        "decoder_calls": decoder_calls,
        "decoded_anchors": decoded_anchors,
        "quality": aggregate_quality,
        "schedule_timing": {
            "per_scene": schedule_timing,
            "warmup_ms_sum_of_scene_medians": float(
                sum(value["warmup_ms_median"] for value in schedule_timing.values())
            ),
            "steady_state_ms_sum_of_scene_medians": float(
                sum(value["steady_state_ms_median"] for value in schedule_timing.values())
            ),
            "mesh_window_batch_ms_sum_of_scene_medians": float(
                sum(value["mesh_window_batch_ms_median"] for value in schedule_timing.values())
            ),
        },
        "scenes": summaries,
    }
    atomic_json(args.output, review)
    print(json.dumps({key: review[key] for key in ("status", "scene_count", "frame_count", "failures", "paired_wall_ms", "speedups", "deadline_misses", "fallback_pairs")}, indent=2))
    raise SystemExit(0 if review["status"] == "pass" else 1)


if __name__ == "__main__":
    main()

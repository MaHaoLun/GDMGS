"""Aggregate the complete Step 7B unconditional cross-pose matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCENES = (
    "amsterdam",
    "barcelona",
    "bilbao",
    "chicago",
    "hollywood",
    "pompidou",
    "quebec",
    "rome",
)
LAGS = (1, 2, 4, 8)
RUN_ID = "formal_step7b_unconditional_v1_20260916"
STEP6_DIAG = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step6_bvh_diagnosis_20260915")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def stats(values: Iterable[float]) -> dict:
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "minimum": None, "maximum": None}
    return {
        "count": len(items),
        "mean": mean(items),
        "p50": float(np.percentile(items, 50)),
        "p95": float(np.percentile(items, 95)),
        "minimum": min(items),
        "maximum": max(items),
    }


def pair_key(scene: str, pair: dict) -> dict:
    return {
        "scene": scene,
        "mode": pair["mode"],
        "lag": pair["lag"],
        "source_index": pair["source_index"],
        "target_index": pair["target_index"],
        "source_camera": pair["source_camera"],
        "target_camera": pair["target_camera"],
    }


def main(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    run_root = root / "runs" / "formal"
    summaries = {}
    records = {}
    formal_pairs = []
    adversarial_pairs = []
    for scene in SCENES:
        directory = run_root / scene / RUN_ID
        status = load(directory / "status.json")
        summary = load(directory / "summary.json")
        per_view = load(directory / "per_view.json")
        if not (
            status.get("state") == "complete"
            and status.get("completed_views") == 32
            and summary.get("state") == "complete"
            and summary.get("view_count") == 32
            and summary.get("formal_pair_count") == 113
            and len(per_view) == 32
        ):
            raise ValueError(f"{scene}: incomplete Step 7B formal run")
        if summary.get("same_pose_failure_count") != 0:
            raise ValueError(f"{scene}: same-pose cache regression failed")
        summaries[scene] = summary
        records[scene] = per_view
        for frame in per_view:
            for lag in LAGS:
                pair = frame["cross_pose"][f"lag{lag}"]
                if pair.get("available") is True:
                    formal_pairs.append((scene, pair))
            if frame.get("adversarial") is not None:
                adversarial_pairs.append((scene, frame["adversarial"]))

    if len(formal_pairs) != 904 or len(adversarial_pairs) != 8:
        raise ValueError("Step 7B matrix denominator must be 904 formal plus 8 adversarial pairs")
    failed_pairs = [(scene, pair) for scene, pair in formal_pairs if pair["fidelity"]["pass"] is False]
    if len(failed_pairs) != 852:
        raise ValueError(f"expected persisted matrix to contain 852 failures, found {len(failed_pairs)}")

    thresholds = next(iter(summaries.values()))["thresholds"]
    if any(summary["thresholds"] != thresholds for summary in summaries.values()):
        raise ValueError("scene thresholds differ")
    gate_names = tuple(next(iter(formal_pairs))[1]["fidelity"]["gates"].keys())
    gate_failures = {
        name: sum(pair["fidelity"]["gates"][name] is False for _, pair in formal_pairs)
        for name in gate_names
    }

    def aggregate_group(group):
        return {
            "pair_count": len(group),
            "pass_count": sum(pair["fidelity"]["pass"] is True for _, pair in group),
            "failure_count": sum(pair["fidelity"]["pass"] is False for _, pair in group),
            "direct": {
                metric: stats(pair["fidelity"]["direct"][metric] for _, pair in group)
                for metric in ("psnr", "ssim", "lpips", "rgb_mae", "rgb_p99_abs", "rgb_max_abs")
            },
            "reuse_minus_fresh_gt": {
                metric: stats(
                    pair["fidelity"]["reuse_minus_fresh_gt"][metric]
                    for _, pair in group
                )
                for metric in ("psnr", "ssim", "lpips")
            },
            "hit_rate": stats(
                pair["resolution"]["hit_anchors"]
                / (pair["resolution"]["hit_anchors"] + pair["resolution"]["miss_anchors"])
                for _, pair in group
            ),
            "pose_translation": stats(pair["pose_delta"]["translation"] for _, pair in group),
            "pose_rotation_degrees": stats(
                pair["pose_delta"]["rotation_degrees"] for _, pair in group
            ),
            "length_mismatch_rate": stats(
                pair["request_diagnostics"]["length_mismatch_anchors"]
                / max(1, pair["request_diagnostics"]["hit_anchors"])
                for _, pair in group
            ),
        }

    per_lag = {
        f"lag{lag}": aggregate_group(
            [(scene, pair) for scene, pair in formal_pairs if pair["lag"] == lag]
        )
        for lag in LAGS
    }
    per_scene = {
        scene: aggregate_group(
            [(name, pair) for name, pair in formal_pairs if name == scene]
        )
        for scene in SCENES
    }

    worst = {
        "direct_psnr": min(formal_pairs, key=lambda item: item[1]["fidelity"]["direct"]["psnr"]),
        "direct_ssim": min(formal_pairs, key=lambda item: item[1]["fidelity"]["direct"]["ssim"]),
        "direct_lpips": max(formal_pairs, key=lambda item: item[1]["fidelity"]["direct"]["lpips"]),
        "rgb_max_abs": max(formal_pairs, key=lambda item: item[1]["fidelity"]["direct"]["rgb_max_abs"]),
        "gt_psnr_delta": min(
            formal_pairs,
            key=lambda item: item[1]["fidelity"]["reuse_minus_fresh_gt"]["psnr"],
        ),
        "gt_ssim_delta": min(
            formal_pairs,
            key=lambda item: item[1]["fidelity"]["reuse_minus_fresh_gt"]["ssim"],
        ),
        "gt_lpips_delta": max(
            formal_pairs,
            key=lambda item: item[1]["fidelity"]["reuse_minus_fresh_gt"]["lpips"],
        ),
    }
    worst_records = {}
    for name, (scene, pair) in worst.items():
        worst_records[name] = {
            **pair_key(scene, pair),
            "direct": pair["fidelity"]["direct"],
            "reuse_minus_fresh_gt": pair["fidelity"]["reuse_minus_fresh_gt"],
            "pose_delta": pair["pose_delta"],
            "request_diagnostics": pair["request_diagnostics"],
        }

    fidelity_report = {
        "schema": "proxygs_step7b_cross_pose_fidelity_contract_v1",
        "status": "complete_with_fidelity_failure",
        "scope": "8 frozen speed-selected windows; unconditional complete-bundle residency",
        "scene_count": 8,
        "frame_count": 256,
        "formal_pair_count": 904,
        "failed_formal_pair_count": 852,
        "passed_formal_pair_count": 52,
        "same_pose_failure_count": 0,
        "thresholds_frozen_before_formal_output": thresholds,
        "gate_failure_counts": gate_failures,
        "per_lag": per_lag,
        "per_scene": per_scene,
        "worst_formal_pairs": worst_records,
        "adversarial": aggregate_group(adversarial_pairs),
        "window_reselection": False,
        "deleted_pairs": 0,
        "completion_boundary": (
            "fidelity calibration is complete; failure does not authorize a pose/age gate"
        ),
    }
    atomic_json(root / "review" / "cross_pose_fidelity_contract.json", fidelity_report)

    reuse_contract = {
        "schema": "proxygs_step7b_reuse_eligibility_contract_v1",
        "status": "blocked_under_adr_0006",
        "normative_hit_validity": "unconditional complete-bundle residency",
        "pose_age_attribute_gate": "forbidden without a new user-approved ADR",
        "diagnostic_bins_change_hits": False,
        "formal_pair_count": 904,
        "failed_pair_count": 852,
        "failing_scenes": list(SCENES),
        "payload_dtype": "float32",
        "formal_capacity_rows": 6_826_846,
        "step7c_authorized": False,
        "fallback": "fresh current-pose decode; do not claim cache reuse",
        "reason": (
            "unconditional cross-pose residency violates the preregistered C2 gate in "
            "every scene and at every tested lag"
        ),
    }
    atomic_json(root / "review" / "reuse_eligibility_contract.json", reuse_contract)

    decoder_points = []
    hit_points = []
    capacity_accumulator = defaultdict(
        lambda: {"oracle_hits": 0, "available_overlap": 0, "rows_used": 0, "pairs": 0}
    )
    for scene, pair in formal_pairs:
        common = pair_key(scene, pair)
        decoder_points.append(
            {
                **common,
                "miss_anchors": pair["resolution"]["miss_anchors"],
                "fresh_rows": pair["resolution"]["fresh_rows"],
                "miss_decode_and_regroup_ms": pair["resolution"]["timings_ms"][
                    "miss_decode_and_regroup"
                ],
            }
        )
        hit_points.append(
            {
                **common,
                "hit_anchors": pair["resolution"]["hit_anchors"],
                "hit_rows": pair["resolution"]["hit_rows"],
                "lookup_ms": pair["resolution"]["timings_ms"]["lookup"],
                "gather_and_sealed_assembly_ms": pair["resolution"]["timings_ms"][
                    "gather_and_sealed_assembly"
                ],
                "resolution_total_ms": pair["resolution"]["timings_ms"]["total"],
            }
        )
        for point in pair["capacity_hit_ceiling"]:
            aggregate = capacity_accumulator[int(point["capacity_rows"])]
            aggregate["oracle_hits"] += int(point["oracle_hit_anchor_ceiling"])
            aggregate["available_overlap"] += int(point["available_overlap_anchors"])
            aggregate["rows_used"] += int(point["rows_used"])
            aggregate["pairs"] += 1

    decoder_table = {
        "schema": "proxygs_step7b_decoder_batch_cost_table_v1",
        "status": "complete",
        "hardware_bound": identity(
            run_root / "bilbao" / RUN_ID / "run_contract.json"
        ),
        "point_count": len(decoder_points),
        "miss_anchor_range": [
            min(item["miss_anchors"] for item in decoder_points),
            max(item["miss_anchors"] for item in decoder_points),
        ],
        "fresh_row_range": [
            min(item["fresh_rows"] for item in decoder_points),
            max(item["fresh_rows"] for item in decoder_points),
        ],
        "timing_ms": stats(item["miss_decode_and_regroup_ms"] for item in decoder_points),
        "points": decoder_points,
    }
    atomic_json(root / "review" / "decoder_batch_cost_table.json", decoder_table)

    hit_table = {
        "schema": "proxygs_step7b_hit_gather_cost_table_v1",
        "status": "complete",
        "point_count": len(hit_points),
        "hit_anchor_range": [
            min(item["hit_anchors"] for item in hit_points),
            max(item["hit_anchors"] for item in hit_points),
        ],
        "hit_row_range": [
            min(item["hit_rows"] for item in hit_points),
            max(item["hit_rows"] for item in hit_points),
        ],
        "lookup_ms": stats(item["lookup_ms"] for item in hit_points),
        "gather_and_sealed_assembly_ms": stats(
            item["gather_and_sealed_assembly_ms"] for item in hit_points
        ),
        "resolution_total_ms": stats(item["resolution_total_ms"] for item in hit_points),
        "points": hit_points,
    }
    atomic_json(root / "review" / "hit_gather_cost_table.json", hit_table)

    write_points = []
    for scene in SCENES:
        for frame in records[scene]:
            build = frame["generation_build"]
            write_points.append(
                {
                    "scene": scene,
                    "target_index": frame["target_index"],
                    "target_camera": frame["target_camera"],
                    "selected_anchors": frame["selected_anchor_count"],
                    "nonempty_descriptors": build["descriptors"],
                    "rows": build["rows"],
                    "build_ms": build["ms"],
                    "raw_memory_bytes": build["raw_memory_bytes"],
                    "scratch_bytes": build["scratch_bytes"],
                }
            )
    write_table = {
        "schema": "proxygs_step7b_write_cost_table_v1",
        "status": "complete",
        "point_count": len(write_points),
        "build_ms": stats(item["build_ms"] for item in write_points),
        "rows": stats(item["rows"] for item in write_points),
        "scratch_bytes": stats(item["scratch_bytes"] for item in write_points),
        "points": write_points,
    }
    atomic_json(root / "review" / "write_cost_table.json", write_table)

    capacity_sweep = []
    for capacity, values in sorted(capacity_accumulator.items()):
        capacity_sweep.append(
            {
                "capacity_rows": capacity,
                "pair_count": values["pairs"],
                "oracle_hit_anchor_ceiling": values["oracle_hits"],
                "available_overlap_anchors": values["available_overlap"],
                "oracle_hit_ceiling_ratio": values["oracle_hits"]
                / max(1, values["available_overlap"]),
                "rows_used": values["rows_used"],
            }
        )
    memory_capacity = {
        "schema": "proxygs_step7b_memory_capacity_sweep_v1",
        "status": "complete",
        "formal_capacity_rows": 6_826_846,
        "formal_capacity_role": "fit every complete source-frame generation; no eviction confound",
        "capacity_sweep_role": "unit-value oracle hit ceiling only; not an admission policy",
        "sweep": capacity_sweep,
        "maximum_generation_raw_bytes": max(
            item["raw_memory_bytes"]["total"] for item in write_points
        ),
        "maximum_build_scratch_bytes": max(item["scratch_bytes"] for item in write_points),
        "maximum_observed_resolution_or_render_peak_bytes": max(
            pair["resolution"]["memory"]["render_peak_bytes"] for _, pair in formal_pairs
        ),
    }
    atomic_json(root / "review" / "memory_capacity_sweep.json", memory_capacity)

    fragmentation = {
        "schema": "proxygs_step7b_fragmentation_compaction_diagnostic_v1",
        "status": "complete_with_scope_boundary",
        "step7b_generation_layout": "contiguous packed rebuild",
        "internal_hole_rows": 0,
        "internal_fragmentation_ratio": 0.0,
        "repack_cost_source": "write_cost_table generation-build measurements",
        "mutable_allocator_and_compaction": "not implemented; belongs to Step 7C",
        "step7c_authorized": False,
        "reason": "reuse fidelity is blocked before allocator/residency implementation",
    }
    atomic_json(root / "review" / "fragmentation_compaction_diagnostic.json", fragmentation)

    upstream_records = {}
    for scene in SCENES:
        path = (
            STEP6_DIAG
            / "runs"
            / "windows"
            / scene
            / "formal_step6_retained_window_v2_20260915"
            / "per_view.json"
        )
        values = load(path)
        upstream_records[scene] = {item["camera"]: item for item in values}
    full_path_modes = {}
    for lag in LAGS:
        fresh_sum = 0.0
        reuse_sum = 0.0
        upstream_sum = 0.0
        pair_count = 0
        for scene, pair in formal_pairs:
            if pair["lag"] != lag:
                continue
            target_frame = records[scene][pair["target_index"]]
            upstream = upstream_records[scene][pair["target_camera"]]
            step6_render = upstream["renders"]["g2_j3_window"]
            upstream_without_cache_stage = (
                upstream["frame_total_ms"]["g2_j3_window"]
                - upstream["selected_ids_h2d_ms"]["g2_j3_window"]
                - step6_render["decode_seconds"] * 1000.0
                - step6_render["render_seconds_mean"] * 1000.0
                - upstream["image_write_ms"]["g2_j3_window"]
            )
            fresh_stage = (
                target_frame["request_setup_ms"]
                + target_frame["fresh"]["decode_ms"]
                + target_frame["fresh"]["render_mean_ms"]
            )
            reuse_stage = (
                target_frame["request_setup_ms"]
                + pair["resolution"]["timings_ms"]["total"]
                + pair["render_mean_ms"]
            )
            upstream_sum += upstream_without_cache_stage
            fresh_sum += upstream_without_cache_stage + fresh_stage
            reuse_sum += upstream_without_cache_stage + reuse_stage
            pair_count += 1
        full_path_modes[f"lag{lag}"] = {
            "pair_count": pair_count,
            "upstream_frozen_retained_v2_sum_ms": upstream_sum,
            "composed_fresh_sum_ms": fresh_sum,
            "composed_reuse_sum_ms": reuse_sum,
            "composed_mechanical_speedup": fresh_sum / reuse_sum,
        }
    pipeline = {
        "schema": "proxygs_step7b_retained_pipeline_cost_breakdown_v1",
        "status": "complete_composed_same_protocol",
        "upstream": "frozen Step 6 Retained-v2 plus J3-Window per-frame records",
        "cache_stage": "Step 7B same-process F0/reuse paired records",
        "modes": full_path_modes,
        "claim_boundary": (
            "component composition, not a newly observed single-run wall clock; no overlap or schedule claim"
        ),
        "upstream_files": {
            scene: identity(
                STEP6_DIAG
                / "runs"
                / "windows"
                / scene
                / "formal_step6_retained_window_v2_20260915"
                / "per_view.json"
            )
            for scene in SCENES
        },
    }
    atomic_json(root / "review" / "retained_pipeline_cost_breakdown.json", pipeline)

    pose_values = [pair["pose_delta"]["translation"] for _, pair in formal_pairs]
    rotation_values = [pair["pose_delta"]["rotation_degrees"] for _, pair in formal_pairs]
    pose_edges = list(map(float, np.percentile(pose_values, [0, 25, 50, 75, 100])))
    rotation_edges = list(map(float, np.percentile(rotation_values, [0, 25, 50, 75, 100])))

    def bin_report(field: str, edges: list[float]):
        output = []
        for index in range(4):
            lower, upper = edges[index], edges[index + 1]
            group = [
                (scene, pair)
                for scene, pair in formal_pairs
                if (
                    lower <= pair["pose_delta"][field] <= upper
                    if index == 3
                    else lower <= pair["pose_delta"][field] < upper
                )
            ]
            output.append(
                {
                    "lower_inclusive": lower,
                    "upper_inclusive": upper if index == 3 else None,
                    "upper_exclusive": upper if index != 3 else None,
                    **aggregate_group(group),
                }
            )
        return output

    diagnostics = {
        "schema": "proxygs_step7b_diagnostic_bins_v1",
        "status": "complete_non_gating",
        "changes_hit_validity": False,
        "by_lag": per_lag,
        "by_translation_quartile": bin_report("translation", pose_edges),
        "by_rotation_quartile": bin_report("rotation_degrees", rotation_edges),
        "interpretation": (
            "bins explain unconditional-residency failure; they are not a pose/age eligibility policy"
        ),
    }
    atomic_json(root / "review" / "diagnostic_bins.json", diagnostics)

    matrix = {
        "schema": "proxygs_step7b_formal_matrix_v1",
        "status": "complete",
        "scene_count": 8,
        "frame_count": 256,
        "formal_pair_count": 904,
        "adversarial_pair_count": 8,
        "same_pose_failure_count": 0,
        "formal_fidelity_failure_count": 852,
        "run_id": RUN_ID,
        "scenes": {
            scene: identity(run_root / scene / RUN_ID / "summary.json")
            for scene in SCENES
        },
        "command_ledger": identity(run_root / "command_ledger.csv"),
    }
    atomic_json(root / "review" / "formal_matrix.json", matrix)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/ssddata/lun/gdmgs_artifacts/proxygs_step7b_cross_pose_20260916"
        ),
    )
    return result


if __name__ == "__main__":
    main(parser().parse_args())

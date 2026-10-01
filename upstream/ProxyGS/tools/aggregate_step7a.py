"""Aggregate the complete Step 7A formal matrix into reviewable artifacts."""

from __future__ import annotations

import argparse
import json
import os
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
RUN_ID = "formal_step7a_cache_core_v4_20260916"


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


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def main(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    run_root = root / "runs" / "formal"
    summaries = {}
    records = {}
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
            and len(per_view) == 32
        ):
            raise ValueError(f"{scene}: formal run is incomplete")
        if [item["camera"] for item in per_view] != summary["camera_ids"]:
            raise ValueError(f"{scene}: per-view camera order differs from summary")
        if any(item.get("pass") is not True for item in per_view):
            raise ValueError(f"{scene}: per-view mechanical gate failed")
        summaries[scene] = summary
        records[scene] = per_view

    flat = [item for scene in SCENES for item in records[scene]]
    if len(flat) != 256:
        raise ValueError("formal matrix must contain exactly 256 frames")
    if len({(scene, item["camera"]) for scene in SCENES for item in records[scene]}) != 256:
        raise ValueError("formal matrix contains duplicate scene/camera identities")

    input_paths = {
        "step6_final_review": Path(
            "/ssddata/lun/gdmgs_artifacts/proxygs_step6_high_overlap_cpu_index_20260915/review/final_step6_review.json"
        ),
        "step6_cache_trace": Path(
            "/ssddata/lun/gdmgs_artifacts/proxygs_step6_high_overlap_cpu_index_20260915/review/cache_trace_manifest.json"
        ),
        "step6_j3_window_contract": Path(
            "/ssddata/lun/gdmgs_artifacts/proxygs_step6_high_overlap_cpu_index_20260915/manifests/j3_window_contract.json"
        ),
        "step6_bvh_review": Path(
            "/ssddata/lun/gdmgs_artifacts/proxygs_step6_bvh_diagnosis_20260915/review/final_bvh_diagnosis_review.json"
        ),
        "retained_profile": Path(
            "/ssddata/lun/gdmgs_artifacts/proxygs_step6_bvh_diagnosis_20260915/qualification/retained_profile_conservative.json"
        ),
        "retained_validation": Path(
            "/ssddata/lun/gdmgs_artifacts/proxygs_step6_bvh_diagnosis_20260915/formal/profile_validation_conservative.json"
        ),
    }
    step6_trace = load(input_paths["step6_cache_trace"])
    input_binding = {
        "schema": "proxygs_step7a_input_binding_v1",
        "status": "bound",
        "identity_policy": "path-size-mtime; no hashes or checksums",
        "inputs": {name: identity(path) for name, path in input_paths.items()},
        "window_count": 8,
        "frame_count": 256,
        "transition_count": 248,
        "windows": [
            {
                "scene": item["scene"],
                "start_index": item["start_index"],
                "end_index_inclusive": item["end_index_inclusive"],
                "camera_ids": item["camera_ids"],
            }
            for item in step6_trace["scenes"]
        ],
        "formal_run_id": RUN_ID,
        "formal_scene_summaries": {
            scene: identity(run_root / scene / RUN_ID / "summary.json")
            for scene in SCENES
        },
        "failed_attempts_retained": [
            {
                "scene": "bilbao",
                "run_id": "formal_step7a_cache_core_v1_20260916",
                "reason": (
                    "runner incorrectly required literal all-hit behavior; zero-row bundles "
                    "must remain misses and are not admitted"
                ),
                "status": identity(
                    run_root
                    / "bilbao"
                    / "formal_step7a_cache_core_v1_20260916"
                    / "status.json"
                ),
            }
        ],
        "diagnostic_runs_retained": [
            {
                "run_id": "formal_step7a_cache_core_v2_20260916",
                "scope": (
                    "complete 8x32 correctness matrix; request ID H2D and level gather were "
                    "outside the paired cache-stage timing, so v2 is not the final timing source"
                ),
            },
            {
                "run_id": "formal_step7a_cache_core_v3_20260916",
                "scope": (
                    "complete 8x32 exactness and final paired request-handoff timing; phase-specific "
                    "generation-build, resolution, and renderer scratch peaks were added in v4"
                ),
            },
        ],
        "window_reselection": False,
        "reduced_count": False,
    }
    atomic_json(root / "manifests" / "step6_cache_input_binding.json", input_binding)

    cache_contract = {
        "schema": "proxygs_full_bundle_cache_contract_v1",
        "status": "implemented",
        "key": ["level_id", "final_ply_anchor_row_id"],
        "n_offsets": 10,
        "bundle_rows": {"minimum_fresh": 0, "minimum_resident": 1, "maximum": 10},
        "payload": {
            "dtype": "float32",
            "fields": {
                "xyz": 3,
                "color": 3,
                "opacity": 1,
                "scaling": 3,
                "rotation": 4,
            },
            "bytes_per_row": 56,
        },
        "row_metadata": ["owner_anchor_id", "owner_level_id", "offset_slot"],
        "directory": [
            "packed_composite_key",
            "level_id",
            "anchor_id",
            "row_offset",
            "row_length",
        ],
        "lookup": "one GPU torch.searchsorted over sorted unique composite keys",
        "miss_decode": "one ordered batched ProxyGS decoder invocation",
        "assembly": "request-order sealed payload with complete hit/fresh segments",
        "zero_row_admission": False,
        "partial_admission": False,
        "partial_eviction": False,
        "cross_pose_validity": "not decided by Step 7A",
    }
    atomic_json(root / "manifests" / "full_bundle_cache_contract.json", cache_contract)

    lifecycle = {
        "schema": "proxygs_step7a_cache_lifecycle_contract_v1",
        "status": "implemented",
        "identity_fields": ["scene", "model", "backend", "anchor_table", "trace"],
        "identity_change": "complete reset before lookup",
        "generation_ids": "nonnegative and monotonically increasing",
        "current_resolution": "read-only snapshot",
        "publication": "validated atomic assignment between frames only",
        "publication_during_resolution": "fail-closed",
        "sealed_output_aliases_cache_rows": False,
        "scene_window_boundary": "reset",
        "future_residency_writer": "not implemented in Step 7A",
    }
    atomic_json(root / "manifests" / "cache_lifecycle_contract.json", lifecycle)

    rows = [int(item["decoded_row_count"]) for item in flat]
    descriptors = [int(item["nonempty_bundle_count"]) for item in flat]
    selected = [int(item["selected_anchor_count"]) for item in flat]
    histogram = {str(index): 0 for index in range(11)}
    for item in flat:
        for key, value in item["bundle_length_histogram"].items():
            histogram[key] += int(value)
    capacity_points = []
    for q in (25, 50, 75, 90, 95, 99, 100):
        capacity = max(10, int(np.ceil(np.percentile(rows, q))))
        capacity_points.append(
            {
                "label": f"p{q}",
                "capacity_rows": capacity,
                "float32_payload_bytes": capacity * 56,
                "frames_fitting": sum(value <= capacity for value in rows),
                "frame_count": 256,
                "per_scene_frames_fitting": {
                    scene: sum(
                        int(item["decoded_row_count"]) <= capacity
                        for item in records[scene]
                    )
                    for scene in SCENES
                },
            }
        )
    deduplicated = []
    seen = set()
    for point in capacity_points:
        if point["capacity_rows"] not in seen:
            deduplicated.append(point)
            seen.add(point["capacity_rows"])
    capacity_report = {
        "schema": "proxygs_step7a_bundle_size_capacity_report_v1",
        "status": "complete",
        "scope": "8 frozen windows, 256 frames, same-pose fresh decoder output",
        "n_offsets": 10,
        "row_payload_bytes": 56,
        "bundle_length_histogram": histogram,
        "selected_anchor_total": sum(selected),
        "resident_nonempty_bundle_total": sum(descriptors),
        "zero_row_bundle_total": histogram["0"],
        "decoded_row_total": sum(rows),
        "rows_per_frame": {
            "minimum": min(rows),
            "p25": percentile(rows, 25),
            "p50": percentile(rows, 50),
            "p75": percentile(rows, 75),
            "p90": percentile(rows, 90),
            "p95": percentile(rows, 95),
            "p99": percentile(rows, 99),
            "maximum": max(rows),
        },
        "capacity_sweep": deduplicated,
        "maximum_generation_raw_bytes": max(
            item["same_pose_seeded"]["generation"]["raw_memory_bytes"]["total"]
            for item in flat
        ),
        "maximum_observed_torch_peak_allocated_bytes": max(
            item["same_pose_seeded"]["peak_allocated_bytes"] for item in flat
        ),
        "scratch_memory_bytes": {
            "generation_build_max": max(
                item["same_pose_seeded"]["generation_build_memory"]["scratch_bytes"]
                for item in flat
            ),
            "resolution_assembly_max": max(
                sample["scratch_bytes"]
                for item in flat
                for sample in item["same_pose_seeded"]["resolution_memory_samples"]
            ),
            "renderer_max": max(
                item["same_pose_seeded"]["render_memory"]["scratch_bytes"]
                for item in flat
            ),
            "measurement": "torch.cuda memory_allocated before/after and max_memory_allocated peak",
        },
        "fragmentation": {
            "internal_hole_rows": 0,
            "ratio": 0.0,
            "reason": "Step 7A rebuilds each sealed generation into contiguous packed segments",
            "mutable_allocator_external_fragmentation": "not applicable until Step 7C",
        },
        "logical_capacity_note": (
            "row capacity does not include model, sealed output, gsplat working memory, "
            "directory, row metadata, or future transition scratch"
        ),
    }
    atomic_json(root / "review" / "bundle_size_capacity_report.json", capacity_report)

    exactness = {
        "schema": "proxygs_step7a_same_pose_exactness_report_v1",
        "status": "pass",
        "scene_count": 8,
        "frame_count": 256,
        "modes": ["C0-fresh", "C1-same-pose-seeded", "C2-controlled-mixed"],
        "same_pose_payload_exact": all(summary["same_pose_payload_exact"] for summary in summaries.values()),
        "same_pose_metadata_exact": all(summary["same_pose_metadata_exact"] for summary in summaries.values()),
        "same_pose_render_exact": all(summary["same_pose_render_exact"] for summary in summaries.values()),
        "controlled_mixed_payload_exact": all(summary["controlled_mixed_payload_exact"] for summary in summaries.values()),
        "controlled_mixed_metadata_exact": all(summary["controlled_mixed_metadata_exact"] for summary in summaries.values()),
        "controlled_mixed_render_exact": all(summary["controlled_mixed_render_exact"] for summary in summaries.values()),
        "zero_row_not_admitted": all(summary["zero_row_not_admitted"] for summary in summaries.values()),
        "maximum_render_abs_delta": max(
            max(
                item["same_pose_seeded"]["render_max_abs_delta"],
                item["controlled_mixed"]["render_max_abs_delta"],
            )
            for item in flat
        ),
        "failures": [],
        "per_scene": {
            scene: {
                "frames": summaries[scene]["view_count"],
                "same_pose_payload_exact": summaries[scene]["same_pose_payload_exact"],
                "same_pose_metadata_exact": summaries[scene]["same_pose_metadata_exact"],
                "same_pose_render_exact": summaries[scene]["same_pose_render_exact"],
                "controlled_mixed_payload_exact": summaries[scene]["controlled_mixed_payload_exact"],
                "controlled_mixed_metadata_exact": summaries[scene]["controlled_mixed_metadata_exact"],
                "controlled_mixed_render_exact": summaries[scene]["controlled_mixed_render_exact"],
            }
            for scene in SCENES
        },
    }
    if not all(
        value is True
        for key, value in exactness.items()
        if key.endswith("_exact") or key == "zero_row_not_admitted"
    ) or exactness["maximum_render_abs_delta"] != 0.0:
        raise ValueError("formal exactness aggregate failed")
    atomic_json(root / "review" / "same_pose_exactness_report.json", exactness)

    phase_names = (
        "key_build",
        "lookup",
        "miss_decode_and_regroup",
        "prefix_sum",
        "gather_and_sealed_assembly",
        "total",
    )
    microbench = {
        "schema": "proxygs_step7a_cache_resolution_microbench_v1",
        "status": "complete",
        "scope": "component timings with explicit synchronization; not end-to-end FPS",
        "scene_count": 8,
        "frame_count": 256,
        "same_pose_seeded_ms": {
            phase: {
                "mean": mean(
                    sample[phase]
                    for item in flat
                    for sample in item["same_pose_seeded"]["resolution_samples_ms"]
                ),
                "p50": percentile(
                    [
                        sample[phase]
                        for item in flat
                        for sample in item["same_pose_seeded"]["resolution_samples_ms"]
                    ],
                    50,
                ),
                "p95": percentile(
                    [
                        sample[phase]
                        for item in flat
                        for sample in item["same_pose_seeded"]["resolution_samples_ms"]
                    ],
                    95,
                ),
            }
            for phase in phase_names
        },
        "controlled_mixed_ms": {
            phase: {
                "mean": mean(item["controlled_mixed"]["resolution_ms"][phase] for item in flat),
                "p50": percentile(
                    [item["controlled_mixed"]["resolution_ms"][phase] for item in flat],
                    50,
                ),
                "p95": percentile(
                    [item["controlled_mixed"]["resolution_ms"][phase] for item in flat],
                    95,
                ),
            }
            for phase in phase_names
        },
        "generation_build_ms": {
            "mean": mean(item["same_pose_seeded"]["generation_build_ms"] for item in flat),
            "p50": percentile(
                [item["same_pose_seeded"]["generation_build_ms"] for item in flat], 50
            ),
            "p95": percentile(
                [item["same_pose_seeded"]["generation_build_ms"] for item in flat], 95
            ),
        },
        "fresh_decode_ms": {
            "mean": mean(item["fresh"]["decode_ms"] for item in flat),
            "p50": percentile([item["fresh"]["decode_ms"] for item in flat], 50),
            "p95": percentile([item["fresh"]["decode_ms"] for item in flat], 95),
        },
        "request_setup_ms": {
            "mean": mean(item["request_setup_ms"] for item in flat),
            "p50": percentile([item["request_setup_ms"] for item in flat], 50),
            "p95": percentile([item["request_setup_ms"] for item in flat], 95),
        },
        "paired_cache_stage_frame": {
            "scope": "selected-ID H2D plus level gather plus decode/cache resolution plus renderer",
            "fresh_sum_ms": sum(item["fresh"]["cache_stage_frame_ms"] for item in flat),
            "same_pose_seeded_sum_ms": sum(
                mean(item["same_pose_seeded"]["cache_stage_frame_samples_ms"])
                for item in flat
            ),
        },
    }
    paired = microbench["paired_cache_stage_frame"]
    paired["mechanical_speedup"] = (
        paired["fresh_sum_ms"] / paired["same_pose_seeded_sum_ms"]
    )
    paired["claim_boundary"] = (
        "trace-replay cache-stage comparison only; CPU index, online depth, Future Residency, "
        "and asynchronous schedule are not timed here"
    )
    atomic_json(root / "review" / "cache_resolution_microbench.json", microbench)

    matrix = {
        "schema": "proxygs_step7a_formal_matrix_v1",
        "status": "complete",
        "scene_count": 8,
        "frame_count": 256,
        "run_id": RUN_ID,
        "scenes": {
            scene: identity(run_root / scene / RUN_ID / "summary.json")
            for scene in SCENES
        },
        "command_ledger": identity(run_root / "command_ledger.csv"),
        "failures": [],
    }
    atomic_json(root / "review" / "formal_matrix.json", matrix)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/ssddata/lun/gdmgs_artifacts/proxygs_step7a_full_bundle_cache_20260916"
        ),
    )
    return result


if __name__ == "__main__":
    main(parser().parse_args())

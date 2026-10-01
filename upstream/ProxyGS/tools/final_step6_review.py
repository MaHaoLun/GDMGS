#!/usr/bin/env python3
"""Independently review the complete Step 6 full-view and window matrices."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step6_high_overlap_cpu_index_20260915")
FULL_RUN_ID = "formal_step6_j3_full_v1_20260915"
WINDOW_RUN_ID = "formal_step6_j3_window_v1_20260915"
EXPECTED = {
    "amsterdam": 161,
    "barcelona": 160,
    "bilbao": 129,
    "chicago": 160,
    "hollywood": 125,
    "pompidou": 161,
    "quebec": 160,
    "rome": 158,
}


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    failures: list[str] = []
    audit_path = ROOT / "review" / "step6_input_audit.json"
    qualification_path = (
        ROOT / "runs" / "qualification" / "amsterdam"
        / "formal_complete_scene_qualification_j3_full_20260915" / "summary.json"
    )
    selection_path = ROOT / "review" / "speed_selected_high_overlap_windows.json"
    ranking_path = ROOT / "review" / "all_candidate_window_ranking.json"
    j3_review_path = ROOT / "review" / "j3_full_review.json"
    prerequisites = [
        audit_path,
        qualification_path,
        selection_path,
        ranking_path,
        j3_review_path,
        ROOT / "command_ledger.csv",
        ROOT / "window_command_ledger.csv",
        ROOT / "manifests" / "environment_manifest.json",
        ROOT / "manifests" / "window_selection_policy.json",
        ROOT / "manifests" / "optimized_mesh_index_contract.json",
        ROOT / "manifests" / "optimized_anchor_index_contract.json",
        ROOT / "manifests" / "j3_full_contract.json",
        ROOT / "manifests" / "joint_index_execution_contract.json",
        ROOT / "manifests" / "window_mesh_batch_tuning_manifest.json",
        ROOT / "review" / "index_load_memory_report.json",
    ]
    for path in prerequisites:
        if not path.is_file():
            failures.append(f"missing prerequisite: {path}")
    if failures:
        raise RuntimeError("; ".join(failures))
    for ledger_path in (ROOT / "command_ledger.csv", ROOT / "window_command_ledger.csv"):
        with ledger_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 8 or any(not Path(row["log"]).is_file() for row in rows):
            failures.append(f"command ledger is incomplete or has missing logs: {ledger_path}")
    audit = load(audit_path)
    qualification = load(qualification_path)
    selection = load(selection_path)
    ranking = load(ranking_path)
    j3_review = load(j3_review_path)
    window_tuning = load(ROOT / "manifests" / "window_mesh_batch_tuning_manifest.json")
    load_memory = load(ROOT / "review" / "index_load_memory_report.json")
    if not (
        audit.get("status") == "pass"
        and audit.get("view_count") == 1214
        and qualification.get("state") == "complete"
        and qualification.get("qualification") is True
        and qualification.get("view_count") == 161
        and all(qualification.get("parity", {}).values())
        and qualification.get("timing_ms", {}).get("mesh_speedup", 0) > 1.0
        and qualification.get("timing_ms", {}).get("anchor_speedup", 0) > 1.0
        and selection.get("status") == "frozen"
        and selection.get("window_count") == 8
        and selection.get("frame_count") == 256
        and selection.get("transition_count") == 248
        and selection.get("cache_signals_used") is False
        and ranking.get("status") == "pass"
        and j3_review.get("status") == "pass"
        and j3_review.get("view_count") == 1214
        and window_tuning.get("status") == "frozen_before_window_matrix"
        and window_tuning.get("selection_changed") is False
        and load_memory.get("status") == "pass"
        and load_memory.get("scene_count") == 8
    ):
        failures.append("aggregate prerequisite or qualification contract failed")

    winners = {record["scene"]: record for record in selection.get("windows", [])}
    full_scene_reviews = []
    window_scene_reviews = []
    cache_scenes = []
    full_view_count = 0
    window_frame_count = 0
    full_totals = {key: 0.0 for key in ("j0_mesh", "j1_mesh", "j0_anchor", "j2_anchor")}
    full_frame_totals = {key: 0.0 for key in ("j0", "j1", "j2", "j3_full")}
    window_totals = {key: 0.0 for key in ("j0_mesh", "j1_mesh", "j0_anchor", "j2_anchor")}
    frame_totals = {key: 0.0 for key in ("g1_ref", "g1_brute", "g1_bvh", "g2_j0", "g2_j3_window")}
    distributions: dict[str, list[dict[str, Any]]] = {
        key: []
        for key in (
            "j0_mesh",
            "j1_mesh",
            "j0_anchor",
            "j2_anchor",
            "j0_index",
            "j3_index",
            "j0_frame",
            "j3_frame",
        )
    }

    for scene, expected in EXPECTED.items():
        full_run = ROOT / "runs" / "formal" / scene / FULL_RUN_ID
        window_run = ROOT / "runs" / "windows" / scene / WINDOW_RUN_ID
        full_summary_path = full_run / "summary.json"
        full_records_path = full_run / "per_view.json"
        window_summary_path = window_run / "summary.json"
        window_records_path = window_run / "per_view.json"
        required = [full_summary_path, full_records_path, window_summary_path, window_records_path]
        if any(not path.is_file() for path in required):
            failures.append(f"{scene}: formal completion files missing")
            continue
        full_summary = load(full_summary_path)
        full_records = load(full_records_path)
        window_summary = load(window_summary_path)
        window_records = load(window_records_path)
        winner = winners.get(scene)
        full_checks = {
            "state": full_summary.get("state") == "complete",
            "formal": full_summary.get("formal") is True,
            "not_qualification": full_summary.get("qualification") is False,
            "view_count": full_summary.get("view_count") == expected == len(full_records),
            "parity": all(full_summary.get("parity", {}).values()),
            "unique_cameras": len({record["camera"] for record in full_records}) == expected,
            "mesh_positive": full_summary.get("timing_ms", {}).get("mesh_speedup", 0) > 1.0,
            "anchor_positive": full_summary.get("timing_ms", {}).get("anchor_speedup", 0) > 1.0,
            "joint_positive": full_summary.get("timing_ms", {}).get("joint_index_speedup", 0) > 1.0,
        }
        window_checks = {
            "state": window_summary.get("state") == "complete",
            "formal": window_summary.get("formal") is True,
            "view_count": window_summary.get("view_count") == 32 == len(window_records),
            "parity": all(window_summary.get("parity", {}).values()),
            "winner_bound": winner is not None and window_summary.get("window", {}).get("start_index") == winner.get("start_index"),
            "camera_ids": winner is not None and [record["camera"] for record in window_records] == winner.get("camera_ids"),
            "source_indices": winner is not None and [record["index"] for record in window_records] == list(range(winner["start_index"], winner["end_index_inclusive"] + 1)),
            "batch_workers": all(
                record["mesh"]["j3_window"]["workers"] == window_tuning["selected_workers"]
                for record in window_records
            ),
        }
        if not all(full_checks.values()):
            failures.append(f"{scene}: full-view checks failed: {full_checks}")
        if not all(window_checks.values()):
            failures.append(f"{scene}: window checks failed: {window_checks}")
        full_view_count += len(full_records)
        window_frame_count += len(window_records)
        for record in full_records:
            values = {
                "j0_mesh": record["mesh"]["j0"]["mean_ms"],
                "j1_mesh": record["mesh"]["j1"]["mean_ms"],
                "j0_anchor": record["anchor"]["j0"]["mean_ms"],
                "j2_anchor": record["anchor"]["j2"]["mean_ms"],
            }
            values["j0_index"] = values["j0_mesh"] + values["j0_anchor"]
            values["j3_index"] = values["j1_mesh"] + values["j2_anchor"]
            values["j0_frame"] = record["frame_total_ms"]["j0"]
            values["j3_frame"] = record["frame_total_ms"]["j3_full"]
            for key in full_totals:
                full_totals[key] += values[key]
            for key in full_frame_totals:
                full_frame_totals[key] += record["frame_total_ms"][key]
            for key, value in values.items():
                distributions[key].append(
                    {"scene": scene, "camera": record["camera"], "value_ms": value}
                )
        payloads = []
        for record in window_records:
            parity = record.get("parity", {})
            boolean_parity_keys = (
                "step5_payload_reversible",
                "g1_brute_bvh_mesh_ids",
                "g1_brute_bvh_depth",
                "g1_ref_bvh_depth",
                "g1_dense_ids",
                "j2_anchor_ids",
                "j2_ranges_expand",
                "j0_vs_step5_ids",
                "decoded_ownership",
                "render_exact",
                "metrics_exact",
            )
            if not all(bool(parity.get(key)) for key in boolean_parity_keys) or float(
                parity.get("render_max_abs_delta", float("inf"))
            ) != 0.0:
                failures.append(f"{scene}/{record.get('camera')}: window parity failed")
            window_totals["j0_mesh"] += record["mesh"]["j0"]["mean_ms"]
            window_totals["j1_mesh"] += record["mesh"]["j3_window"]["mean_ms"]
            window_totals["j0_anchor"] += record["anchor"]["j0"]["mean_ms"]
            window_totals["j2_anchor"] += record["anchor"]["j2"]["mean_ms"]
            for mode in frame_totals:
                frame_totals[mode] += record["frame_total_ms"][mode]
            payload = Path(record["id_payload"]["path"])
            expected_identity = {key: record["id_payload"][key] for key in ("path", "bytes", "mtime_ns")}
            if not payload.is_file() or identity(payload) != expected_identity:
                failures.append(f"{scene}/{record.get('camera')}: window ID payload drifted")
            payloads.append(record["id_payload"])
        full_scene_reviews.append(
            {
                "scene": scene,
                "view_count": len(full_records),
                "checks": full_checks,
                "summary": identity(full_summary_path),
                "timing_ms": full_summary["timing_ms"],
            }
        )
        window_scene_reviews.append(
            {
                "scene": scene,
                "frame_count": len(window_records),
                "checks": window_checks,
                "summary": identity(window_summary_path),
                "timing_ms": window_summary["timing_ms"],
            }
        )
        cache_scenes.append(
            {
                "scene": scene,
                "start_index": winner["start_index"] if winner else None,
                "end_index_inclusive": winner["end_index_inclusive"] if winner else None,
                "camera_ids": winner["camera_ids"] if winner else [],
                "id_payloads": payloads,
                "per_view": identity(window_records_path),
            }
        )

    full_speedups = {
        "mesh": full_totals["j0_mesh"] / full_totals["j1_mesh"],
        "anchor": full_totals["j0_anchor"] / full_totals["j2_anchor"],
        "joint": (full_totals["j0_mesh"] + full_totals["j0_anchor"])
        / (full_totals["j1_mesh"] + full_totals["j2_anchor"]),
    }
    window_speedups = {
        "mesh": window_totals["j0_mesh"] / window_totals["j1_mesh"],
        "anchor": window_totals["j0_anchor"] / window_totals["j2_anchor"],
        "joint": (window_totals["j0_mesh"] + window_totals["j0_anchor"])
        / (window_totals["j1_mesh"] + window_totals["j2_anchor"]),
    }
    distribution_review = {}
    for key, records in distributions.items():
        values = np.asarray([record["value_ms"] for record in records], dtype=np.float64)
        worst = max(records, key=lambda record: record["value_ms"])
        distribution_review[key] = {
            "mean_ms": float(values.mean()),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "worst": worst,
        }
    if full_view_count != 1214 or window_frame_count != 256 or len(cache_scenes) != 8:
        failures.append("final full/window denominator drifted")
    if not all(value > 1.0 for value in full_speedups.values()):
        failures.append(f"full-view optimized profile is not faster: {full_speedups}")
    if not all(value > 1.0 for value in window_speedups.values()):
        failures.append(f"window optimized profile is not faster: {window_speedups}")

    cache_trace = {
        "schema": "proxygs_step6_cache_trace_manifest_v1",
        "status": "pass" if not failures else "failed",
        "label": "speed-selected high-reuse continuous-view workload",
        "window_selection": identity(selection_path),
        "window_count": len(cache_scenes),
        "frame_count": window_frame_count,
        "transition_count": window_frame_count - len(cache_scenes),
        "j3_window_profile": "optimized Mesh plus optimized Anchor; unrestricted per-frame discovery",
        "scenes": cache_scenes,
    }
    cache_path = ROOT / "review" / "cache_trace_manifest.json"
    atomic_json(cache_path, cache_trace)
    j3_window_contract = {
        "schema": "proxygs_step6_j3_window_contract_v1",
        "status": "frozen" if not failures else "failed",
        "protocol_id": "proxygs-step6-j3-window-v1",
        "window_count": 8,
        "frame_count": window_frame_count,
        "transition_count": window_frame_count - 8,
        "profile": "J3-Full Anchor optimization plus fixed-window batched Mesh query",
        "window_only_candidate_result": "retained 32-camera fixed worker pool; no temporal result reuse or restrictive hint",
        "temporal_discovery": "unrestricted; each frame executes a complete query",
        "implementation": identity(
            ROOT / "runtime" / "Proxy-GS-eac937e8" / "render_step6_windows.py"
        ),
        "full_view_speedups": full_speedups,
        "window_speedups": window_speedups,
        "cache_trace": identity(cache_path),
    }
    j3_window_path = ROOT / "manifests" / "j3_window_contract.json"
    atomic_json(j3_window_path, j3_window_contract)
    review = {
        "schema": "proxygs_step6_final_review_v1",
        "status": "pass" if not failures else "failed",
        "completion_claim": (
            "J3-Full matched J0 on all 8 scenes and 1,214 views; the frozen algorithm "
            "selected one 32-frame speed-selected high-reuse window per scene; J3-Window "
            "and the complete G1 Ref/Brute/BVH chains matched on all 256 frozen frames."
        ),
        "scene_count": len(full_scene_reviews),
        "full_view_count": full_view_count,
        "window_count": len(window_scene_reviews),
        "window_frame_count": window_frame_count,
        "window_transition_count": window_frame_count - len(window_scene_reviews),
        "correctness_failure_count": len(failures),
        "failures": failures,
        "reduced_count": False,
        "silent_cap": False,
        "normal_path_fallback": False,
        "cache_signals_used_for_selection": False,
        "window_reselection_after_j3_window": False,
        "full_view_elapsed_sums_ms": full_totals,
        "full_view_speedups": full_speedups,
        "full_view_frame_sums_ms": full_frame_totals,
        "full_view_frame_speedup_j0_to_j3": full_frame_totals["j0"] / full_frame_totals["j3_full"],
        "full_view_timing_distribution": distribution_review,
        "window_elapsed_sums_ms": window_totals,
        "window_speedups": window_speedups,
        "same_protocol_window_frame_sums_ms": frame_totals,
        "same_protocol_window_frame_speedups": {
            "g2_j0_to_j3_window": frame_totals["g2_j0"] / frame_totals["g2_j3_window"],
            "g1_brute_to_bvh": frame_totals["g1_brute"] / frame_totals["g1_bvh"],
            "g1_ref_to_brute": frame_totals["g1_ref"] / frame_totals["g1_brute"],
        },
        "full_scene_reviews": full_scene_reviews,
        "window_scene_reviews": window_scene_reviews,
        "speed_selected_windows": identity(selection_path),
        "cache_trace": identity(cache_path),
        "j3_window_contract": identity(j3_window_path),
        "index_load_memory_report": identity(ROOT / "review" / "index_load_memory_report.json"),
        "scope": "speed-selected high-reuse continuous-view workload; not an unbiased full-scene trajectory result",
    }
    review_path = ROOT / "review" / "final_step6_review.json"
    atomic_json(review_path, review)
    if failures:
        raise RuntimeError(f"Step 6 final review failed: {failures}")
    print(json.dumps(review, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

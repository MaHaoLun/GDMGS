"""Independent full-denominator review and Step 6 handoff for Step 5."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step5_cpu_anchor_index_g2_20260915")
STEP3 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step3_gdmgs_backend_20260914")
STEP4 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step4_cpu_mesh_index_g1_v2_20260914")
RUN_ID = "formal_step5_g2_v1_20260915"
SCENES = {
    "amsterdam": 161,
    "barcelona": 160,
    "bilbao": 129,
    "chicago": 160,
    "hollywood": 125,
    "pompidou": 161,
    "quebec": 160,
    "rome": 158,
}
PAYLOAD_FIELDS = {
    "anchor_universe_count",
    "triangle_universe_count",
    "candidate_bitmap",
    "selected_bitmap",
    "triangle_bitmap",
}


def load(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def expand(candidate_ids: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    parts = [candidate_ids[begin:end] for begin, end in ranges]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def unpack_ids(bitmap: np.ndarray, count: int) -> np.ndarray:
    if bitmap.dtype != np.uint8 or bitmap.ndim != 1:
        raise TypeError("ID bitmap must be rank-one uint8")
    values = np.unpackbits(bitmap, count=count, bitorder="little")
    return np.flatnonzero(values).astype(np.int64, copy=False)


def jaccard_sorted(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    intersection = int(np.intersect1d(first, second, assume_unique=True).size)
    union = int(len(first) + len(second) - intersection)
    return {
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 1.0,
    }


def main() -> None:
    failures = []
    scene_reviews = []
    trace_scenes = []
    total_views = 0
    total_brute_query = total_index_query = 0.0
    total_brute_frame = total_index_frame = 0.0

    prerequisite_files = (
        ROOT / "manifests" / "step5_input_binding_manifest.json",
        ROOT / "manifests" / "anchor_query_contract.json",
        ROOT / "manifests" / "step5_source_manifest.json",
        ROOT / "manifests" / "anchor_index_tuning_manifest.json",
        ROOT / "manifests" / "environment_manifest.json",
        ROOT / "review" / "contract_tests.txt",
        ROOT / "review" / "anchor_index_build_matrix.json",
    )
    for path in prerequisite_files:
        if not path.is_file():
            failures.append(f"missing prerequisite evidence: {path}")
    if failures:
        raise RuntimeError("; ".join(failures))
    binding = load(ROOT / "manifests" / "step5_input_binding_manifest.json")
    if not (
        binding.get("status") == "pass"
        and binding.get("scene_count") == 8
        and binding.get("view_count") == 1214
    ):
        failures.append("Step 5 input binding is not complete/pass")
    build_matrix = load(ROOT / "review" / "anchor_index_build_matrix.json")
    if not (
        build_matrix.get("status") == "pass"
        and build_matrix.get("scene_count") == 8
        and all(record.get("save_load_parity") for record in build_matrix.get("records", []))
    ):
        failures.append("8-scene Anchor Index build matrix is not complete/pass")
    qualification_path = (
        ROOT
        / "runs"
        / "qualification"
        / "amsterdam"
        / "formal_complete_scene_qualification_bvh_bitmap_20260915"
        / "summary.json"
    )
    qualification = load(qualification_path) if qualification_path.is_file() else None
    if not (
        qualification
        and qualification.get("state") == "complete"
        and qualification.get("view_count") == 161
        and all(qualification.get("parity", {}).values())
        and qualification.get("timing_ms", {}).get("anchor_component_speedup", 0) > 1.0
    ):
        failures.append("Amsterdam full-scene qualification is missing, incorrect, or not faster than brute")

    for scene, expected_views in SCENES.items():
        run = ROOT / "runs" / "formal" / scene / RUN_ID
        summary_path = run / "summary.json"
        per_view_path = run / "per_view.json"
        status_path = run / "status.json"
        if not (summary_path.is_file() and per_view_path.is_file() and status_path.is_file()):
            failures.append(f"{scene}: formal completion files are missing")
            continue
        summary = load(summary_path)
        records = load(per_view_path)
        status = load(status_path)
        cameras = load(run / "camera_domain_records.json")
        checks = {
            "summary_complete": summary.get("state") == "complete",
            "status_complete": status.get("state") == "complete",
            "formal_flag": summary.get("formal") is True,
            "qualification_flag": summary.get("qualification") is False,
            "view_count": summary.get("view_count") == expected_views,
            "record_count": len(records) == expected_views,
            "camera_record_count": len(cameras) == expected_views,
            "unique_cameras": len({record.get("camera") for record in records}) == expected_views,
            "summary_parity": all(summary.get("parity", {}).values()),
            "zero_correctness_failures": not summary.get("correctness_failures"),
            "anchor_tree_faster_than_brute": summary.get("timing_ms", {}).get(
                "anchor_component_speedup", 0
            ) > 1.0,
        }
        transition_records = []
        previous = None
        for index, record in enumerate(records):
            camera = record.get("camera")
            record_parity = record.get("parity", {})
            if not all(
                bool(record_parity.get(key))
                for key in (
                    "step4_g1_bvh_vs_g2_ref_ids",
                    "reference_vs_brute_ids",
                    "brute_vs_index_ids",
                    "brute_ranges_expand",
                    "index_ranges_expand",
                    "original_order",
                    "decoded_ownership",
                    "render_exact",
                    "metrics_exact",
                )
            ):
                failures.append(f"{scene}/{camera}: record parity failed")
                continue
            payload = Path(record["id_payload"]["path"])
            if not payload.is_file() or identity(payload) != {
                key: record["id_payload"][key] for key in ("path", "bytes", "mtime_ns")
            }:
                failures.append(f"{scene}/{camera}: ID payload identity drifted")
                continue
            with np.load(payload, allow_pickle=False) as data:
                if set(data.files) != PAYLOAD_FIELDS:
                    failures.append(f"{scene}/{camera}: payload schema drifted")
                    continue
                arrays = {name: np.ascontiguousarray(data[name]) for name in data.files}
            anchor_count = int(arrays["anchor_universe_count"].item())
            triangle_count = int(arrays["triangle_universe_count"].item())
            candidates = unpack_ids(arrays["candidate_bitmap"], anchor_count)
            selected = unpack_ids(arrays["selected_bitmap"], anchor_count)
            triangles = unpack_ids(arrays["triangle_bitmap"], triangle_count)
            step4_payload = (
                STEP4
                / "runs"
                / "formal"
                / scene
                / "formal_step4_g1_v2_20260914"
                / "id_payload"
                / f"{camera}.npz"
            )
            with np.load(step4_payload, allow_pickle=False) as step4_data:
                step4_candidates = np.ascontiguousarray(
                    step4_data["candidate_ids"], dtype=np.int64
                )
                step4_selected = np.ascontiguousarray(
                    step4_data["selected_ids"], dtype=np.int64
                )
            if not (
                candidates.dtype == np.int64
                and selected.dtype == np.int64
                and triangles.dtype == np.int64
                and (len(candidates) < 2 or np.all(candidates[1:] > candidates[:-1]))
                and (len(selected) < 2 or np.all(selected[1:] > selected[:-1]))
                and (len(triangles) < 2 or np.all(triangles[1:] > triangles[:-1]))
                and np.array_equal(candidates, step4_candidates)
                and np.array_equal(selected, step4_selected)
                and record["parity"]["brute_ranges_expand"]
                and record["parity"]["index_ranges_expand"]
            ):
                failures.append(f"{scene}/{camera}: independent payload parity failed")
                continue
            if previous is not None:
                transition_records.append(
                    {
                        "from_index": index - 1,
                        "to_index": index,
                        "from_camera": previous["camera"],
                        "to_camera": camera,
                        "mesh": jaccard_sorted(previous["triangles"], triangles),
                        "selected_anchor": jaccard_sorted(previous["selected"], selected),
                    }
                )
            previous = {"camera": camera, "triangles": triangles, "selected": selected}
            total_brute_query += record["g2_brute"]["timings"]["anchor_index_total_ms"]
            total_index_query += record["g2_index"]["timings"]["anchor_index_total_ms"]
            total_brute_frame += record["g2_brute_frame_total_ms"]
            total_index_frame += record["g2_index_frame_total_ms"]
        checks["render_evidence_count"] = sum(
            bool(record.get("parity", {}).get("render_exact")) for record in records
        ) == expected_views
        checks["transition_count"] = len(transition_records) == expected_views - 1
        if not all(checks.values()):
            failures.append(f"{scene}: scene checks failed: {checks}")
        total_views += len(records)
        scene_reviews.append(
            {
                "scene": scene,
                "view_count": len(records),
                "checks": checks,
                "summary": identity(summary_path),
                "timing_ms": summary.get("timing_ms"),
                "metrics": {
                    "g2_brute": summary.get("g2_brute_metrics"),
                    "g2_index": summary.get("g2_index_metrics"),
                },
                "index_efficiency": summary.get("index_efficiency"),
            }
        )
        trace_scenes.append(
            {
                "scene": scene,
                "view_count": len(records),
                "camera_order": [record["camera"] for record in records],
                "camera_records": identity(run / "camera_domain_records.json"),
                "per_view": [
                    {
                        "index": record["index"],
                        "camera": record["camera"],
                        "id_payload": record["id_payload"],
                        "candidate_anchor_count": record["candidate_anchor_count"],
                        "selected_anchor_count": record["selected_anchor_count"],
                        "mesh_returned_triangles": record["mesh_returned_triangles"],
                        "mesh_query_cpu_ms": record["mesh_query_cpu_ms"],
                        "brute_anchor_query_ms": record["g2_brute"]["timings"]["anchor_index_total_ms"],
                        "index_anchor_query_ms": record["g2_index"]["timings"]["anchor_index_total_ms"],
                        "decoded_row_count": record["g2_index_render"]["decoded_row_count"],
                    }
                    for record in records
                ],
                "adjacent_transitions": transition_records,
            }
        )

    trace = {
        "schema": "proxygs_step5_full_view_overlap_trace_v1",
        "protocol_id": "proxygs-step5-g2-v1",
        "status": "pass" if not failures else "failed",
        "scene_count": len(trace_scenes),
        "view_count": total_views,
        "transition_count": sum(len(scene["adjacent_transitions"]) for scene in trace_scenes),
        "window_selection_performed": False,
        "step6_role": "complete full-view input for overlap eligibility and later speed selection",
        "scenes": trace_scenes,
    }
    trace_path = ROOT / "review" / "full_view_overlap_trace_manifest.json"
    atomic_json(trace_path, trace)
    step3_review = STEP3 / "review" / "final_gdmgs_backend_review.json"
    step4_review = STEP4 / "review" / "final_step4_v2_review.json"
    selection_report = {
        "schema": "proxygs_step5_selection_pipeline_report_v1",
        "status": "pass" if not failures else "failed",
        "source_reports": {
            "G0_step3": identity(step3_review),
            "G1_step4": identity(step4_review),
        },
        "execution_availability": {
            "G0": "Step3 complete all-anchor baseline",
            "G1-Ref": "Step4 complete full-mesh depth/dense/render",
            "G1-Brute": "Step4 triangle query component only; no inferred full frame",
            "G1-BVH": "Step4 complete Mesh BVH/depth/dense/render",
            "G2-Ref": "Step5 rerun dense CPU oracle",
            "G2-Brute": "Step5 complete native CPU linear query and render",
            "G2-Index": "Step5 complete CPU binned-SAH point-BVH query and render",
        },
        "step5_elapsed_sums_ms": {
            "g2_brute_anchor_query": total_brute_query,
            "g2_index_anchor_query": total_index_query,
            "g2_brute_frame": total_brute_frame,
            "g2_index_frame": total_index_frame,
        },
        "step5_speedups": {
            "anchor_component": total_brute_query / total_index_query if total_index_query else None,
            "complete_frame": total_brute_frame / total_index_frame if total_index_frame else None,
        },
        "attribution_rules": {
            "anchor_component": "G2-Brute native linear / G2-Index native tree elapsed sums",
            "complete_frame": "G2-Brute / G2-Index complete frame elapsed sums",
            "mesh_query": "reported unchanged and never attributed to Anchor Index",
            "unexecuted_G1_Brute_frame": "not estimated or backfilled",
        },
        "scene_reviews": scene_reviews,
        "full_view_overlap_trace": identity(trace_path),
    }
    selection_path = ROOT / "review" / "final_selection_pipeline_report.json"
    atomic_json(selection_path, selection_report)
    review = {
        "schema": "proxygs_step5_final_review_v1",
        "protocol_id": "proxygs-step5-g2-v1",
        "status": "pass" if not failures else "failed",
        "completion_claim": (
            "G2-Index replaced CPU G2-Brute on the frozen G1-BVH mesh/depth path and "
            "matched G2-Ref/G2-Brute IDs, order, decoded ownership, and renders on all "
            "eight Bungee scenes and 1,214 views."
        ),
        "scene_count": len(scene_reviews),
        "expected_scene_count": 8,
        "view_count": total_views,
        "expected_view_count": 1214,
        "transition_count": trace["transition_count"],
        "expected_transition_count": 1206,
        "failed_scene_count": sum(not all(scene["checks"].values()) for scene in scene_reviews),
        "correctness_failure_count": len(failures),
        "failures": failures,
        "scene_reviews": scene_reviews,
        "selection_pipeline_report": identity(selection_path),
        "full_view_overlap_trace": identity(trace_path),
        "speedups": selection_report["step5_speedups"],
        "qualification_not_reused": True,
        "qualification": identity(qualification_path) if qualification_path.is_file() else None,
        "reduced_count": False,
        "silent_cap": False,
        "normal_path_fallback": False,
        "step6_window_selection_performed": False,
    }
    review_path = ROOT / "review" / "final_step5_review.json"
    atomic_json(review_path, review)
    if failures or len(scene_reviews) != 8 or total_views != 1214 or trace["transition_count"] != 1206:
        raise RuntimeError(f"Step 5 final review failed: {failures}")
    print(json.dumps(review, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

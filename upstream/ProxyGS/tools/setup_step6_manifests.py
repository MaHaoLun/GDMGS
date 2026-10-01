#!/usr/bin/env python3
"""Audit the frozen Step 5 handoff and preregister Step 6 contracts."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step6_high_overlap_cpu_index_20260915")
STEP5 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step5_cpu_anchor_index_g2_20260915")
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
PAYLOAD_FIELDS = {
    "anchor_universe_count",
    "triangle_universe_count",
    "candidate_bitmap",
    "selected_bitmap",
    "triangle_bitmap",
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


def unpack(bitmap: np.ndarray, count: int) -> np.ndarray:
    return np.flatnonzero(
        np.unpackbits(np.ascontiguousarray(bitmap, dtype=np.uint8), count=count, bitorder="little")
    ).astype(np.int64, copy=False)


def main() -> None:
    review_path = STEP5 / "review" / "final_step5_review.json"
    trace_path = STEP5 / "review" / "full_view_overlap_trace_manifest.json"
    selection_path = STEP5 / "review" / "final_selection_pipeline_report.json"
    review = load(review_path)
    trace = load(trace_path)
    failures: list[str] = []
    if not (
        review.get("status") == "pass"
        and review.get("scene_count") == 8
        and review.get("view_count") == 1214
        and review.get("transition_count") == 1206
        and review.get("correctness_failure_count") == 0
        and review.get("reduced_count") is False
        and review.get("silent_cap") is False
        and review.get("normal_path_fallback") is False
    ):
        failures.append("Step 5 final review is not the complete passing handoff")
    if not (
        trace.get("status") == "pass"
        and trace.get("scene_count") == 8
        and trace.get("view_count") == 1214
        and trace.get("transition_count") == 1206
        and trace.get("window_selection_performed") is False
    ):
        failures.append("Step 5 full-view trace is incomplete or already selected")

    scene_records = []
    total_views = 0
    total_transitions = 0
    for scene in trace.get("scenes", []):
        name = scene.get("scene")
        expected = EXPECTED.get(name)
        if expected is None or scene.get("view_count") != expected:
            failures.append(f"{name}: frozen view count drifted")
            continue
        camera_order = scene.get("camera_order", [])
        per_view = scene.get("per_view", [])
        transitions = scene.get("adjacent_transitions", [])
        if len(camera_order) != expected or len(per_view) != expected or len(set(camera_order)) != expected:
            failures.append(f"{name}: camera order is incomplete or non-unique")
            continue
        if len(transitions) != expected - 1:
            failures.append(f"{name}: transition count drifted")
            continue
        row_bounds = None
        payload_records = []
        for index, record in enumerate(per_view):
            camera = camera_order[index]
            if record.get("index") != index or record.get("camera") != camera:
                failures.append(f"{name}/{camera}: camera identity/order drifted")
                continue
            payload = Path(record["id_payload"]["path"])
            if identity(payload) != {
                key: record["id_payload"][key] for key in ("path", "bytes", "mtime_ns")
            }:
                failures.append(f"{name}/{camera}: payload identity drifted")
                continue
            with np.load(payload, allow_pickle=False) as data:
                if set(data.files) != PAYLOAD_FIELDS:
                    failures.append(f"{name}/{camera}: payload schema drifted")
                    continue
                anchor_count = int(data["anchor_universe_count"].item())
                triangle_count = int(data["triangle_universe_count"].item())
                candidates = unpack(data["candidate_bitmap"], anchor_count)
                selected = unpack(data["selected_bitmap"], anchor_count)
                triangles = unpack(data["triangle_bitmap"], triangle_count)
            current_bounds = (anchor_count, triangle_count)
            if row_bounds is None:
                row_bounds = current_bounds
            elif row_bounds != current_bounds:
                failures.append(f"{name}/{camera}: row upper bounds changed within scene")
            if not (
                len(candidates) == record["id_payload"]["candidate_count"]
                and len(selected) == record["id_payload"]["selected_count"]
                and len(triangles) == record["id_payload"]["triangle_count"]
                and (len(candidates) < 2 or np.all(candidates[1:] > candidates[:-1]))
                and (len(selected) < 2 or np.all(selected[1:] > selected[:-1]))
                and (len(triangles) < 2 or np.all(triangles[1:] > triangles[:-1]))
            ):
                failures.append(f"{name}/{camera}: bitmap lossless decode/count/order failed")
            payload_records.append(identity(payload))
        total_views += len(per_view)
        total_transitions += len(transitions)
        scene_records.append(
            {
                "scene": name,
                "view_count": len(per_view),
                "transition_count": len(transitions),
                "camera_order": camera_order,
                "row_upper_bounds": {
                    "anchor": row_bounds[0] if row_bounds else None,
                    "triangle": row_bounds[1] if row_bounds else None,
                },
                "payload_count": len(payload_records),
                "camera_records": scene["camera_records"],
            }
        )
    if total_views != 1214 or total_transitions != 1206 or len(scene_records) != 8:
        failures.append("aggregate frozen denominator drifted")

    runtime = ROOT / "runtime" / "Proxy-GS-eac937e8"
    sources = {
        "mesh_native": identity(runtime / "gdmgs" / "mesh_index" / "native" / "mesh_native.cpp"),
        "mesh_python": identity(runtime / "gdmgs" / "mesh_index" / "index.py"),
        "anchor_native": identity(runtime / "gdmgs" / "anchor_index" / "native" / "anchor_point_native.cpp"),
        "anchor_python": identity(runtime / "gdmgs" / "anchor_index" / "point_index.py"),
        "full_runner": identity(runtime / "render_step6_full.py"),
    }
    audit = {
        "schema": "proxygs_step6_input_audit_v1",
        "status": "pass" if not failures else "failed",
        "scene_count": len(scene_records),
        "view_count": total_views,
        "transition_count": total_transitions,
        "bitmap_encoding": "numpy_packbits_original_row_bitmap",
        "bitorder": "little",
        "camera_order": "Step5 frozen per-scene order; no wrap or shuffle",
        "source_reports": {
            "step5_final_review": identity(review_path),
            "step5_selection_pipeline": identity(selection_path),
            "step5_full_view_trace": identity(trace_path),
        },
        "scenes": scene_records,
        "failures": failures,
    }
    atomic_json(ROOT / "review" / "step6_input_audit.json", audit)

    environment = {
        "schema": "proxygs_step6_environment_v1",
        "status": "frozen_before_qualification",
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "cpu_threads": {"mesh_optimized": 16, "anchor": 1},
        "numa_policy": "single process, OS placement; no cross-process formal concurrency",
        "warmup": 1,
        "index_repeat": 3,
        "render_repeat": 1,
        "source_identity_policy": "path, bytes, and mtime_ns; no checksum",
    }
    atomic_json(ROOT / "manifests" / "environment_manifest.json", environment)
    atomic_json(
        ROOT / "manifests" / "window_selection_policy.json",
        {
            "schema": "proxygs_step6_window_selection_policy_v1",
            "status": "frozen_before_j3_full",
            "window_length": 32,
            "enumeration": "all contiguous windows per scene; no wrap; no shuffle",
            "overlap_dtype": "float64",
            "transition_aggregation": "numpy median over 31 adjacent transitions",
            "joint_overlap": "harmonic mean of median anchor and median mesh Jaccard; zero if denominator is zero",
            "eligibility_quantile": 0.75,
            "quantile_algorithm": "numpy.quantile method=linear over all candidate JointOverlap values",
            "boundary": "inclusive: JointOverlap >= quantile threshold",
            "empty_policy": "hard failure; inclusive maximum guarantees nonempty finite inputs",
            "ranking": [
                "maximum paired-sum J0/J3-Full CPU index speedup",
                "minimum J3-Full CPU index elapsed sum",
                "maximum JointOverlap",
                "maximum minimum per-transition joint overlap",
                "minimum start index",
                "lexicographically minimum start camera ID",
            ],
            "prohibited_signals": ["cache hit", "cache fidelity", "cache latency", "Future Residency", "Schedule"],
        },
    )
    atomic_json(
        ROOT / "manifests" / "optimized_mesh_index_contract.json",
        {
            "schema": "proxygs_step6_optimized_mesh_index_v1",
            "status": "frozen_before_qualification",
            "oracle": "Step4 G1-BVH original-row triangle IDs",
            "implementation": "parallel triangle tests with double sign prefilter and frozen long-double ambiguity fallback",
            "threads": 16,
            "discovery": "unrestricted full BVH traversal; temporal state cannot filter candidates",
            "output_order": "sorted unique original triangle row order",
            "sources": {key: value for key, value in sources.items() if key.startswith("mesh_")},
        },
    )
    atomic_json(
        ROOT / "manifests" / "optimized_anchor_index_contract.json",
        {
            "schema": "proxygs_step6_optimized_anchor_index_v1",
            "status": "frozen_before_qualification",
            "oracle": "Step5 G2-Index and G2-Brute original-row selected IDs",
            "tree": "unchanged Step5 binned-SAH leaf_capacity=4096 max_depth=32",
            "optimization": "skip duplicate native scans only after external bitmap/depth audit; reuse terminal/selected/range buffers",
            "predicate": "unchanged float32 center-pixel depth + float32 0.3",
            "output_order": "single source-order compaction",
            "sources": {key: value for key, value in sources.items() if key.startswith("anchor_")},
        },
    )
    atomic_json(
        ROOT / "manifests" / "j3_full_contract.json",
        {
            "schema": "proxygs_step6_j3_full_contract_v1",
            "status": "frozen_before_qualification",
            "protocol_id": "proxygs-step6-j3-full-v1",
            "modes": ["J0", "J1-Mesh", "J2-Anchor", "J3-Full"],
            "scene_count": 8,
            "view_count": 1214,
            "camera_order": "Step5 frozen order",
            "online_depth": "one identical indexed-online depth per paired mode set",
            "render_backend": "gdmgs-gsplat-v1",
            "paired_order": "baseline/optimized order rotates by camera and repeat",
            "index_repeat": 3,
            "render_repeat": 1,
            "normal_path_fallback": False,
            "reduced_count": False,
            "sources": sources,
        },
    )
    atomic_json(
        ROOT / "manifests" / "joint_index_execution_contract.json",
        {
            "schema": "proxygs_step6_joint_execution_v1",
            "status": "frozen_before_qualification",
            "shared": ["camera/domain identity", "online depth producer event", "persistent index objects", "buffer lifecycle"],
            "separate_payloads": ["triangle original rows", "anchor original rows"],
            "temporal_hint_policy": "disabled in J3-Full v1; no stale or restrictive hint path exists",
            "fallback": "none",
        },
    )
    if failures:
        raise RuntimeError("; ".join(failures))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

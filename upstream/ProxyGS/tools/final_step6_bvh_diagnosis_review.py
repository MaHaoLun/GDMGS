#!/usr/bin/env python3
"""Final review for the Step 6 BVH diagnosis and retained window profile."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step6_bvh_diagnosis_20260915")
RUN_ID = "formal_step6_retained_window_v2_20260915"
SCENES = (
    "amsterdam", "barcelona", "bilbao", "chicago",
    "hollywood", "pompidou", "quebec", "rome",
)


def load(path: Path):
    return json.loads(path.read_text())


def identity(path: Path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    failures = []
    query_formal_path = ROOT / "formal" / "profile_validation_conservative.json"
    query_formal = load(query_formal_path)
    profile_path = ROOT / "qualification" / "retained_profile_conservative.json"
    profile = load(profile_path)
    if not (
        query_formal.get("status") == "pass"
        and query_formal.get("frame_count") == 256
        and query_formal.get("window_count") == 8
        and query_formal.get("profile_reselected") is False
        and query_formal.get("speedups", {}).get("brute_w32_to_retained", 0) > 1.0
        and not query_formal.get("failures")
    ):
        failures.append("independent retained-profile query validation did not pass")
    if not (
        profile.get("status") == "frozen_from_qualification"
        and profile.get("minimum_index_speedup_to_retain") == 1.2
        and profile.get("selection_changed_windows") is False
        and profile.get("frame_count") == 256
    ):
        failures.append("conservative retained profile is not frozen")

    totals = {
        "g1_ref": 0.0,
        "g1_brute": 0.0,
        "g1_bvh": 0.0,
        "g1_retained": 0.0,
        "g2_j0": 0.0,
        "g2_j3_window": 0.0,
        "legacy_bvh_mesh": 0.0,
        "retained_mesh": 0.0,
    }
    scene_reviews = []
    frame_count = 0
    for scene in SCENES:
        run = ROOT / "runs" / "windows" / scene / RUN_ID
        summary_path = run / "summary.json"
        records_path = run / "per_view.json"
        contract_path = run / "run_contract.json"
        if any(not path.is_file() for path in (summary_path, records_path, contract_path)):
            failures.append(f"{scene}: full-chain completion files missing")
            continue
        summary = load(summary_path)
        records = load(records_path)
        contract = load(contract_path)
        expected_profile = profile["scene_profiles"][scene]
        checks = {
            "state": summary.get("state") == "complete",
            "protocol": summary.get("protocol_id") == "proxygs-step6-j3-window-v2",
            "frame_count": summary.get("view_count") == len(records) == 32,
            "parity": all(summary.get("parity", {}).values()),
            "no_correctness_failures": not summary.get("correctness_failures"),
            "profile_backend": contract.get("window_mesh_batch", {}).get("backend") == expected_profile["backend"],
            "profile_workers": contract.get("window_mesh_batch", {}).get("workers") == expected_profile["workers"],
            "g1_retained_present": "g1_retained" in summary.get("timing_ms", {}).get("frame_sums", {}),
            "unique_cameras": len({record["camera"] for record in records}) == 32,
        }
        if not all(checks.values()):
            failures.append(f"{scene}: full-chain checks failed: {checks}")
        frame_count += len(records)
        frame_sums = summary["timing_ms"]["frame_sums"]
        for key in ("g1_ref", "g1_brute", "g1_bvh", "g1_retained", "g2_j0", "g2_j3_window"):
            totals[key] += frame_sums[key]
        totals["legacy_bvh_mesh"] += summary["timing_ms"]["j0_mesh_sum"]
        totals["retained_mesh"] += summary["timing_ms"]["j3_window_mesh_sum"]
        scene_reviews.append(
            {
                "scene": scene,
                "checks": checks,
                "profile": expected_profile,
                "summary": identity(summary_path),
                "timing_ms": summary["timing_ms"],
            }
        )
    if frame_count != 256 or len(scene_reviews) != 8:
        failures.append("formal full-chain denominator is not 8 windows / 256 frames")
    speedups = {
        "independent_query_brute_w32_to_retained": query_formal.get("speedups", {}).get("brute_w32_to_retained"),
        "independent_query_legacy_bvh_w32_to_retained": query_formal.get("speedups", {}).get("legacy_bvh_w32_to_retained"),
        "full_chain_g1_brute_to_retained": totals["g1_brute"] / totals["g1_retained"],
        "full_chain_g1_bvh_to_retained": totals["g1_bvh"] / totals["g1_retained"],
        "full_chain_g2_j0_to_j3_window": totals["g2_j0"] / totals["g2_j3_window"],
        "legacy_bvh_mesh_to_retained": totals["legacy_bvh_mesh"] / totals["retained_mesh"],
    } if frame_count else {}
    if frame_count and not (
        speedups["full_chain_g1_brute_to_retained"] > 1.0
        and speedups["full_chain_g1_bvh_to_retained"] > 1.0
        and speedups["legacy_bvh_mesh_to_retained"] > 1.0
    ):
        failures.append(f"retained full-chain profile is not faster: {speedups}")
    review = {
        "schema": "proxygs_step6_bvh_diagnosis_final_review_v1",
        "status": "pass" if not failures else "failed",
        "completion_claim": (
            "Diagnosed the old BVH regression, retained a conservative exact window profile, "
            "and validated it on all 8 frozen windows / 256 frames without reselection."
        ),
        "window_count": len(scene_reviews),
        "frame_count": frame_count,
        "correctness_failure_count": len(failures),
        "failures": failures,
        "reduced_count": False,
        "window_reselection": False,
        "index_build_load_in_query_timing": False,
        "query_formal": identity(query_formal_path),
        "retained_profile": identity(profile_path),
        "totals_ms": totals,
        "speedups": speedups,
        "scene_reviews": scene_reviews,
    }
    atomic_json(ROOT / "review" / "final_bvh_diagnosis_review.json", review)
    print(json.dumps(review, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(failures)


if __name__ == "__main__":
    main()

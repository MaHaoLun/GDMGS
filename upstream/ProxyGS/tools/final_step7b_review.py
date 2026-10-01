"""Independent persisted-artifact audit for Step 7B.

The reviewer does not import the cache core, formal runner, or aggregate tool.
It validates JSON records, source text, request-evidence NPZ headers, and test
logs, then distinguishes experiment completion from reuse-policy acceptance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

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


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def identity_matches(record: dict) -> bool:
    path = Path(record["path"])
    if not path.is_file():
        return False
    current = identity(path)
    return current["bytes"] == record["bytes"] and current["mtime_ns"] == record["mtime_ns"]


def npz_headers(path: Path) -> dict[str, dict]:
    output = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            output[name[:-4]] = {
                "shape": list(shape),
                "fortran_order": bool(fortran),
                "dtype": str(dtype),
            }
    return output


def main(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    runtime = root / "runtime" / "Proxy-GS-eac937e8"
    review_root = root / "review"
    required = {
        "adr": root / "manifests" / "adr_0013.md",
        "fidelity": review_root / "cross_pose_fidelity_contract.json",
        "reuse": review_root / "reuse_eligibility_contract.json",
        "decoder_cost": review_root / "decoder_batch_cost_table.json",
        "hit_cost": review_root / "hit_gather_cost_table.json",
        "write_cost": review_root / "write_cost_table.json",
        "capacity": review_root / "memory_capacity_sweep.json",
        "fragmentation": review_root / "fragmentation_compaction_diagnostic.json",
        "pipeline": review_root / "retained_pipeline_cost_breakdown.json",
        "bins": review_root / "diagnostic_bins.json",
        "matrix": review_root / "formal_matrix.json",
        "tests": root / "tests" / "all_contract_tests.txt",
        "runner": runtime / "render_step7b_fidelity.py",
        "core": runtime / "gdmgs" / "cache" / "full_bundle_cache.py",
    }
    checks = {f"required:{name}": path.is_file() for name, path in required.items()}
    if not all(checks.values()):
        raise FileNotFoundError([name for name, passed in checks.items() if not passed])

    fidelity = load(required["fidelity"])
    reuse = load(required["reuse"])
    decoder = load(required["decoder_cost"])
    hit = load(required["hit_cost"])
    write = load(required["write_cost"])
    capacity = load(required["capacity"])
    fragmentation = load(required["fragmentation"])
    pipeline = load(required["pipeline"])
    bins = load(required["bins"])
    matrix = load(required["matrix"])
    checks.update(
        {
            "fidelity_denominator": (
                fidelity.get("status") == "complete_with_fidelity_failure"
                and fidelity.get("scene_count") == 8
                and fidelity.get("frame_count") == 256
                and fidelity.get("formal_pair_count") == 904
                and fidelity.get("failed_formal_pair_count") == 852
                and fidelity.get("passed_formal_pair_count") == 52
                and fidelity.get("same_pose_failure_count") == 0
                and fidelity.get("deleted_pairs") == 0
                and fidelity.get("window_reselection") is False
            ),
            "lag_denominators": (
                fidelity["per_lag"]["lag1"]["pair_count"] == 248
                and fidelity["per_lag"]["lag2"]["pair_count"] == 240
                and fidelity["per_lag"]["lag4"]["pair_count"] == 224
                and fidelity["per_lag"]["lag8"]["pair_count"] == 192
                and fidelity["per_lag"]["lag1"]["failure_count"] == 196
                and fidelity["per_lag"]["lag2"]["failure_count"] == 240
                and fidelity["per_lag"]["lag4"]["failure_count"] == 224
                and fidelity["per_lag"]["lag8"]["failure_count"] == 192
            ),
            "reuse_blocker": (
                reuse.get("status") == "blocked_under_adr_0006"
                and reuse.get("normative_hit_validity")
                == "unconditional complete-bundle residency"
                and reuse.get("diagnostic_bins_change_hits") is False
                and reuse.get("step7c_authorized") is False
                and reuse.get("formal_pair_count") == 904
                and reuse.get("failed_pair_count") == 852
            ),
            "cost_tables": (
                decoder.get("status") == "complete"
                and decoder.get("point_count") == 904
                and hit.get("status") == "complete"
                and hit.get("point_count") == 904
                and write.get("status") == "complete"
                and write.get("point_count") == 256
            ),
            "capacity": (
                capacity.get("status") == "complete"
                and capacity.get("formal_capacity_rows") == 6_826_846
                and len(capacity.get("sweep", [])) >= 5
                and capacity["sweep"][-1]["oracle_hit_ceiling_ratio"] == 1.0
            ),
            "fragmentation_scope": (
                fragmentation.get("internal_hole_rows") == 0
                and fragmentation.get("internal_fragmentation_ratio") == 0.0
                and fragmentation.get("step7c_authorized") is False
                and "Step 7C" in fragmentation.get("mutable_allocator_and_compaction", "")
            ),
            "pipeline_claim_boundary": (
                pipeline.get("status") == "complete_composed_same_protocol"
                and "not a newly observed single-run wall clock" in pipeline.get("claim_boundary", "")
                and all(
                    pipeline["modes"][f"lag{lag}"]["pair_count"] == 32 * 8 - lag * 8
                    for lag in LAGS
                )
            ),
            "diagnostic_bins_non_gating": (
                bins.get("status") == "complete_non_gating"
                and bins.get("changes_hit_validity") is False
            ),
            "formal_matrix": (
                matrix.get("status") == "complete"
                and matrix.get("scene_count") == 8
                and matrix.get("frame_count") == 256
                and matrix.get("formal_pair_count") == 904
                and matrix.get("adversarial_pair_count") == 8
                and matrix.get("same_pose_failure_count") == 0
                and matrix.get("formal_fidelity_failure_count") == 852
                and matrix.get("run_id") == RUN_ID
            ),
        }
    )

    adr_text = required["adr"].read_text()
    runner_text = required["runner"].read_text()
    core_text = required["core"].read_text()
    checks["adr_conflict_resolved"] = (
        "unconditional complete-bundle residency" in adr_text
        and "blocked_under_adr_0006" in adr_text
        and "new user-approved" in adr_text
        and "ADR" in adr_text
    )
    checks["no_pose_gate_implementation"] = (
        "diagnostic_bins_do_not_gate_hits" in runner_text
        and "pose_threshold" not in runner_text
        and "age_threshold" not in runner_text
        and "pose_distance" not in core_text
    )
    checks["inputs_current"] = True
    scene_reviews = {}
    total_frames = 0
    total_pairs = 0
    total_failures = 0
    total_adversarial = 0
    evidence_files = 0
    evidence_headers_valid = True
    expected_npz_fields = {
        "anchor_ids",
        "level_ids",
        "source_lengths",
        "target_lengths",
        "matched_offset_slots",
        "source_only_slots",
        "target_only_slots",
        "attribute_max_abs",
        "attribute_mean_abs",
    }
    for scene in SCENES:
        directory = root / "runs" / "formal" / scene / RUN_ID
        status = load(directory / "status.json")
        contract = load(directory / "run_contract.json")
        summary = load(directory / "summary.json")
        per_view = load(directory / "per_view.json")
        total_frames += len(per_view)
        total_pairs += summary["formal_pair_count"]
        total_failures += summary["failed_formal_pair_count"]
        total_adversarial += sum(frame.get("adversarial") is not None for frame in per_view)
        input_unchanged = contract["inputs"] == summary["input_identities_after"]
        checks["inputs_current"] = checks["inputs_current"] and input_unchanged and all(
            identity_matches(value) for value in contract["inputs"].values()
        )
        pair_count = 0
        failure_count = 0
        same_pose_ok = True
        scene_evidence = 0
        for frame in per_view:
            same = frame["same_pose_regression"]
            same_pose_ok = (
                same_pose_ok
                and same["payload_metadata_exact"] is True
                and same["render_exact"] is True
                and same["render_max_abs_delta"] == 0.0
            )
            available_pairs = []
            for lag in LAGS:
                pair = frame["cross_pose"][f"lag{lag}"]
                if pair.get("available") is True:
                    available_pairs.append(pair)
            if frame.get("adversarial") is not None:
                available_pairs.append(frame["adversarial"])
            for pair in available_pairs:
                if pair["mode"] != "adversarial_first_last":
                    pair_count += 1
                    failure_count += pair["fidelity"]["pass"] is False
                evidence = pair["request_diagnostics"]["request_evidence"]
                if not identity_matches(evidence):
                    evidence_headers_valid = False
                    continue
                headers = npz_headers(Path(evidence["path"]))
                hit_count = pair["request_diagnostics"]["hit_anchors"]
                if set(headers) != expected_npz_fields:
                    evidence_headers_valid = False
                for name in expected_npz_fields - {"attribute_max_abs", "attribute_mean_abs"}:
                    evidence_headers_valid = evidence_headers_valid and headers[name]["shape"] == [hit_count]
                for name in ("attribute_max_abs", "attribute_mean_abs"):
                    evidence_headers_valid = evidence_headers_valid and headers[name]["shape"] == [hit_count, 5]
                scene_evidence += 1
                evidence_files += 1
        scene_pass = (
            status.get("state") == "complete"
            and status.get("completed_views") == 32
            and status.get("formal_pairs") == 113
            and status.get("same_pose_failures") == 0
            and contract.get("formal") is True
            and contract.get("camera_count") == 32
            and contract.get("lags") == [1, 2, 4, 8]
            and contract.get("hit_validity")
            == "unconditional-complete-bundle-residency-per-ADR-0006-and-0013"
            and contract.get("diagnostic_bins_do_not_gate_hits") is True
            and contract.get("capacity_rows") == 6_826_846
            and contract.get("future_residency") is False
            and contract.get("schedule") is False
            and summary.get("state") == "complete"
            and summary.get("view_count") == 32
            and summary.get("formal_pair_count") == 113
            and summary.get("same_pose_failure_count") == 0
            and summary.get("reuse_policy_result") == "blocked_under_adr_0006"
            and len(per_view) == 32
            and pair_count == 113
            and failure_count == summary["failed_formal_pair_count"]
            and same_pose_ok
            and input_unchanged
            and scene_evidence == 114
        )
        checks[f"scene:{scene}"] = scene_pass
        scene_reviews[scene] = {
            "pass": scene_pass,
            "frame_count": len(per_view),
            "formal_pair_count": pair_count,
            "formal_failure_count": failure_count,
            "same_pose_exact": same_pose_ok,
            "request_evidence_files": scene_evidence,
            "input_identities_unchanged": input_unchanged,
            "summary": identity(directory / "summary.json"),
        }
    checks["global_denominators"] = (
        total_frames == 256
        and total_pairs == 904
        and total_failures == 852
        and total_adversarial == 8
    )
    checks["request_evidence_complete"] = evidence_files == 912 and evidence_headers_valid

    test_text = required["tests"].read_text()
    match = re.search(r"(\d+) passed", test_text)
    checks["all_contract_tests"] = bool(match and int(match.group(1)) == 57)
    tests = {
        "schema": "proxygs_step7b_tests_v1",
        "status": "pass" if checks["all_contract_tests"] else "fail",
        "passed": int(match.group(1)) if match else 0,
        "command": "python -m pytest -q tests",
        "environment": {
            "GDMGS_NATIVE_DIR": (
                "/ssddata/lun/gdmgs_artifacts/"
                "proxygs_step6_bvh_diagnosis_20260915/native_fast"
            ),
            "TORCH_EXTENSIONS_DIR": (
                "/ssddata/lun/gdmgs_artifacts/"
                "proxygs_step4_cpu_mesh_index_g1_v2_20260914/torch_extensions"
            ),
        },
        "log": identity(required["tests"]),
        "coverage": [
            "frozen C2 thresholds and p100 capacity",
            "finite exact-image metric handling",
            "owner/offset-aligned hit-request diagnostics",
            "capacity hit-ceiling oracle",
            "Step 7A full-bundle core regressions",
            "Step 3 decoder/raster and Step 4/5/6 index regressions",
        ],
    }
    atomic_json(review_root / "step7b_tests.json", tests)

    admission = {
        "schema": "proxygs_step7b_admission_decision_v1",
        "status": "no_cross_pose_admission_authorized",
        "reason": "852 of 904 formal unconditional-residency pairs fail the preregistered C2 gate",
        "capacity_rows_calibrated": 6_826_846,
        "capacity_is_not_adoption": True,
        "current_frame_behavior": "fresh current-pose decode",
        "cache_update_strategy": "do not construct a reusable cross-pose next generation",
        "pose_age_gate": "not authorized; requires a new user-approved ADR",
        "step7c_authorized": False,
    }
    atomic_json(review_root / "admission_decision.json", admission)

    failures = sorted(name for name, passed in checks.items() if not passed)
    review = {
        "schema": "proxygs_step7b_final_review_v1",
        "status": "pass_calibration_complete" if not failures else "fail",
        "reuse_policy_status": "blocked_under_adr_0006",
        "reviewer": "independent persisted-artifact validator; no cache/runner imports",
        "scene_count": 8,
        "frame_count": total_frames,
        "formal_pair_count": total_pairs,
        "formal_fidelity_failure_count": total_failures,
        "same_pose_failure_count": 0,
        "request_evidence_file_count": evidence_files,
        "window_reselection": False,
        "deleted_pairs": 0,
        "step7c_authorized": False,
        "checks": checks,
        "scene_reviews": scene_reviews,
        "failures": failures,
        "artifacts": {name: identity(path) for name, path in required.items()},
        "completion_claim": (
            "Step 7B unconditional cross-pose fidelity and cost calibration is complete on "
            "the frozen 8-window, 256-frame workload. Same-pose cache mechanics remain exact, "
            "but unconditional residency fails C2 in 852/904 formal pairs. Under ADR 0006/0013, "
            "cross-pose admission is blocked, fresh current-pose decode is the fallback, and "
            "Step 7C is not authorized. No pose/age gate or end-to-end schedule claim is made."
        ),
    }
    atomic_json(review_root / "final_step7b_review.json", review)
    print(json.dumps(review, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


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

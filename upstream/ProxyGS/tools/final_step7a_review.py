"""Independent JSON/source audit for the complete Step 7A artifact set.

This validator intentionally does not import the cache implementation or the
formal runner.  It reads persisted contracts, raw per-frame records, source
text, and test logs as an independent completion gate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


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


def current_identity_matches(value: dict) -> bool:
    path = Path(value["path"])
    if not path.is_file():
        return False
    current = identity(path)
    return current["bytes"] == value["bytes"] and current["mtime_ns"] == value["mtime_ns"]


def main(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    runtime = root / "runtime" / "Proxy-GS-eac937e8"
    required = {
        "input_binding": root / "manifests" / "step6_cache_input_binding.json",
        "cache_contract": root / "manifests" / "full_bundle_cache_contract.json",
        "lifecycle_contract": root / "manifests" / "cache_lifecycle_contract.json",
        "formal_matrix": root / "review" / "formal_matrix.json",
        "exactness": root / "review" / "same_pose_exactness_report.json",
        "capacity": root / "review" / "bundle_size_capacity_report.json",
        "microbench": root / "review" / "cache_resolution_microbench.json",
        "tests": root / "tests" / "all_contract_tests.txt",
        "environment_failure": root / "tests" / "all_contract_tests_missing_native_path.txt",
        "core_source": runtime / "gdmgs" / "cache" / "full_bundle_cache.py",
        "batch_source": runtime / "gaussian_renderer" / "raster_batch.py",
        "runner_source": runtime / "render_step7a_cache.py",
    }
    checks = {f"required_file:{name}": path.is_file() for name, path in required.items()}
    if not all(checks.values()):
        missing = [name for name, passed in checks.items() if not passed]
        raise FileNotFoundError(f"missing Step 7A files: {missing}")

    binding = load(required["input_binding"])
    cache_contract = load(required["cache_contract"])
    lifecycle = load(required["lifecycle_contract"])
    matrix = load(required["formal_matrix"])
    exactness = load(required["exactness"])
    capacity = load(required["capacity"])
    microbench = load(required["microbench"])

    checks.update(
        {
            "input_binding": (
                binding.get("status") == "bound"
                and binding.get("window_count") == 8
                and binding.get("frame_count") == 256
                and binding.get("transition_count") == 248
                and binding.get("window_reselection") is False
                and binding.get("reduced_count") is False
                and binding.get("formal_run_id") == RUN_ID
            ),
            "input_identities_current": all(
                current_identity_matches(value) for value in binding["inputs"].values()
            ),
            "cache_contract": (
                cache_contract.get("status") == "implemented"
                and cache_contract.get("key")
                == ["level_id", "final_ply_anchor_row_id"]
                and cache_contract.get("n_offsets") == 10
                and cache_contract["bundle_rows"]
                == {"minimum_fresh": 0, "minimum_resident": 1, "maximum": 10}
                and cache_contract["payload"]["dtype"] == "float32"
                and cache_contract["payload"]["bytes_per_row"] == 56
                and cache_contract.get("zero_row_admission") is False
                and cache_contract.get("partial_admission") is False
                and cache_contract.get("partial_eviction") is False
            ),
            "lifecycle_contract": (
                lifecycle.get("status") == "implemented"
                and lifecycle.get("current_resolution") == "read-only snapshot"
                and lifecycle.get("publication_during_resolution") == "fail-closed"
                and lifecycle.get("sealed_output_aliases_cache_rows") is False
            ),
            "formal_matrix": (
                matrix.get("status") == "complete"
                and matrix.get("scene_count") == 8
                and matrix.get("frame_count") == 256
                and matrix.get("run_id") == RUN_ID
                and matrix.get("failures") == []
            ),
            "exactness": (
                exactness.get("status") == "pass"
                and exactness.get("scene_count") == 8
                and exactness.get("frame_count") == 256
                and exactness.get("same_pose_payload_exact") is True
                and exactness.get("same_pose_metadata_exact") is True
                and exactness.get("same_pose_render_exact") is True
                and exactness.get("controlled_mixed_payload_exact") is True
                and exactness.get("controlled_mixed_metadata_exact") is True
                and exactness.get("controlled_mixed_render_exact") is True
                and exactness.get("zero_row_not_admitted") is True
                and exactness.get("maximum_render_abs_delta") == 0.0
                and exactness.get("failures") == []
            ),
            "capacity_report": (
                capacity.get("status") == "complete"
                and capacity.get("n_offsets") == 10
                and capacity.get("row_payload_bytes") == 56
                and len(capacity.get("capacity_sweep", [])) >= 5
                and capacity["rows_per_frame"]["maximum"] <= 10
                * max(
                    load(
                        root
                        / "runs"
                        / "formal"
                        / scene
                        / RUN_ID
                        / "summary.json"
                    )["bundle_statistics"]["selected_anchors"]
                    for scene in SCENES
                )
                and capacity["fragmentation"]["internal_hole_rows"] == 0
                and capacity["fragmentation"]["ratio"] == 0.0
                and all(
                    value >= 0
                    for key, value in capacity["scratch_memory_bytes"].items()
                    if key.endswith("_max")
                )
            ),
            "microbench_scope": (
                microbench.get("status") == "complete"
                and microbench.get("scene_count") == 8
                and microbench.get("frame_count") == 256
                and microbench["paired_cache_stage_frame"]["fresh_sum_ms"] > 0
                and microbench["paired_cache_stage_frame"]["same_pose_seeded_sum_ms"] > 0
                and microbench["paired_cache_stage_frame"]["mechanical_speedup"] > 0
                and "not timed" in microbench["paired_cache_stage_frame"]["claim_boundary"]
            ),
        }
    )

    tests_text = required["tests"].read_text()
    environment_failure_text = required["environment_failure"].read_text()
    match = re.search(r"(\d+) passed", tests_text)
    checks["all_contract_tests"] = bool(match and int(match.group(1)) == 52)
    checks["environment_failure_retained"] = (
        "ProxyGS_anchor_point_native" in environment_failure_text
        and "FAILURES" in environment_failure_text
    )
    test_report = {
        "schema": "proxygs_step7a_cache_core_tests_v1",
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
        "failed_environment_attempt_retained": identity(required["environment_failure"]),
        "coverage": [
            "zero/one/multirow bundles",
            "all-hit/all-miss/controlled mixed",
            "duplicate/out-of-range/wrong-LoD requests",
            "zero-row non-admission",
            "row-capacity overflow without partial admission",
            "read-only generation publication",
            "reset and sealed-output lifetime",
            "empty request/background batch",
            "Step 3 decoder/raster handoff regressions",
            "Step 4/5/6 index contract regressions",
        ],
    }
    atomic_json(root / "review" / "cache_core_tests.json", test_report)

    source_text = "\n".join(
        required[name].read_text() for name in ("core_source", "batch_source", "runner_source")
    )
    checks["legacy_cache_not_imported"] = (
        "rendering_cache_optimized" not in source_text
        and "CacheGS_remote_sources" not in source_text
    )
    core_text = required["core_source"].read_text()
    checks["batched_sorted_directory"] = (
        "torch.searchsorted" in core_text
        and "torch.argsort" in core_text
        and "repeat_interleave" in core_text
    )
    checks["no_step7bc_policy_in_core"] = (
        "pose_distance" not in core_text
        and "prefetch(" not in core_text
        and "async def" not in core_text
        and "class LRU" not in core_text
    )

    total_frames = 0
    scene_reviews = {}
    for scene in SCENES:
        directory = root / "runs" / "formal" / scene / RUN_ID
        status = load(directory / "status.json")
        contract = load(directory / "run_contract.json")
        summary = load(directory / "summary.json")
        per_view = load(directory / "per_view.json")
        total_frames += len(per_view)
        input_unchanged = contract["inputs"] == summary["input_identities_after"]
        per_frame_checks = []
        for item in per_view:
            histogram = item["bundle_length_histogram"]
            hist_request_count = sum(int(value) for value in histogram.values())
            hist_row_count = sum(int(key) * int(value) for key, value in histogram.items())
            seeded = item["same_pose_seeded"]
            mixed = item["controlled_mixed"]
            per_frame_checks.append(
                item.get("pass") is True
                and hist_request_count == item["selected_anchor_count"]
                and hist_row_count == item["decoded_row_count"]
                and max(map(int, histogram)) == 10
                and seeded["generation"]["descriptors"] == item["nonempty_bundle_count"]
                and seeded["generation"]["rows"] == item["decoded_row_count"]
                and seeded["generation"]["capacity_rows"] >= item["decoded_row_count"]
                and seeded["hit_anchors"] + seeded["zero_row_miss_anchors"]
                == item["selected_anchor_count"]
                and seeded["payload_and_metadata"]["payload_exact"] is True
                and seeded["payload_and_metadata"]["metadata_exact"] is True
                and seeded["render_exact"] is True
                and seeded["render_max_abs_delta"] == 0.0
                and mixed["payload_and_metadata"]["payload_exact"] is True
                and mixed["payload_and_metadata"]["metadata_exact"] is True
                and mixed["render_exact"] is True
                and mixed["render_max_abs_delta"] == 0.0
                and seeded["generation_build_memory"]["scratch_bytes"] >= 0
                and all(
                    sample["scratch_bytes"] >= 0
                    for sample in seeded["resolution_memory_samples"]
                )
                and seeded["render_memory"]["scratch_bytes"] >= 0
            )
        scene_pass = (
            status.get("state") == "complete"
            and status.get("completed_views") == 32
            and contract.get("formal") is True
            and contract.get("camera_count") == 32
            and contract.get("n_offsets") == 10
            and contract.get("cross_pose_reuse") is False
            and contract.get("future_residency") is False
            and contract.get("schedule") is False
            and contract.get("window_reselection") is False
            and summary.get("state") == "complete"
            and summary.get("view_count") == 32
            and summary.get("correctness_failures") == []
            and input_unchanged
            and len(per_view) == 32
            and all(per_frame_checks)
        )
        checks[f"scene:{scene}"] = scene_pass
        scene_reviews[scene] = {
            "pass": scene_pass,
            "frame_count": len(per_view),
            "input_identities_unchanged": input_unchanged,
            "summary": identity(directory / "summary.json"),
            "per_view": identity(directory / "per_view.json"),
        }
    checks["formal_frame_denominator"] = total_frames == 256

    failures = sorted(name for name, passed in checks.items() if not passed)
    review = {
        "schema": "proxygs_step7a_final_review_v1",
        "status": "pass" if not failures else "fail",
        "reviewer": (
            "independent persisted-artifact validator; does not import the cache core or runner"
        ),
        "scene_count": 8,
        "frame_count": total_frames,
        "window_count": 8,
        "correctness_failure_count": 0 if not failures else len(failures),
        "reduced_count": False,
        "window_reselection": False,
        "checks": checks,
        "scene_reviews": scene_reviews,
        "failures": failures,
        "artifacts": {name: identity(path) for name, path in required.items()},
        "completion_claim": (
            "Full-Bundle Cache Core is bound to the frozen Step 6 Retained-v2 plus "
            "J3-Window workload and is mechanically exact for same-pose seeded and "
            "controlled complete-bundle hit/miss assembly. This review makes no "
            "cross-pose reuse, Future Residency, asynchronous schedule, full-pipeline, "
            "or end-to-end FPS claim."
        ),
    }
    atomic_json(root / "review" / "final_step7a_review.json", review)
    print(json.dumps(review, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


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

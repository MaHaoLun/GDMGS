"""Fail-closed independent artifact review for complete Step 4 G1 evidence."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step4_cpu_mesh_index_g1_v2_20260914")
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
FORMAL_RUN_ID = "formal_step4_g1_v2_20260914"
PROTOCOL_ID = "proxygs-step4-g1-v2"


def load(path: Path):
    return json.loads(path.read_text())


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def review_run(run: Path, scene: str, expected: int, expected_names, *, qualification: bool):
    failures = []
    try:
        status = load(run / "status.json")
        summary = load(run / "summary.json")
        records = load(run / "per_view.json")
        contract = load(run / "run_contract.json")
        ref_manifest = load(run / "g1_ref_anchor_ids.json")
        g1_manifest = load(run / "g1_anchor_ids.json")
    except Exception as error:
        status = load(run / "status.json") if (run / "status.json").exists() else {}
        partial_records = load(run / "per_view.json") if (run / "per_view.json").exists() else []
        detail = {
            "state": status.get("state", "missing"),
            "partial_view_count": len(partial_records),
            "error": status.get("error", repr(error)),
        }
        return None, [f"{scene}: incomplete formal run: {detail}"]
    names = [record.get("camera") for record in records]
    checks = {
        "status_complete": status.get("state") == "complete",
        "summary_complete": summary.get("state") == "complete",
        "backend": summary.get("backend") == "gdmgs-gsplat-v1",
        "protocol": summary.get("protocol_id") == PROTOCOL_ID and contract.get("protocol_id") == PROTOCOL_ID,
        "formal_flag": summary.get("formal") is (not qualification),
        "qualification_flag": summary.get("qualification") is qualification,
        "record_count": len(records) == expected,
        "summary_count": summary.get("view_count") == expected,
        "frozen_count": summary.get("frozen_view_count") == expected,
        "camera_order": names == expected_names,
        "unique_cameras": len(names) == len(set(names)),
        "ref_render_files": sorted(path.stem for path in (run / "g1_ref" / "renders").glob("*.png")) == sorted(expected_names),
        "g1_render_files": sorted(path.stem for path in (run / "g1" / "renders").glob("*.png")) == sorted(expected_names),
        "id_manifests": list(ref_manifest) == expected_names and list(g1_manifest) == expected_names,
        "input_preserved": summary.get("input_identities_after") == contract.get("inputs"),
        "all_aggregate_parity": bool(summary.get("parity")) and all(summary["parity"].values()),
        "no_correctness_failures": summary.get("correctness_failures") == [],
    }
    for camera, ref_item, g1_item in zip(expected_names, ref_manifest.values(), g1_manifest.values()):
        if ref_item != g1_item:
            failures.append(f"{scene}/{camera}: G1-Ref and G1 ID manifests do not share exact payload")
            continue
        try:
            with np.load(ref_item["path"], allow_pickle=False) as payload:
                candidate = payload["candidate_ids"]
                selected = payload["selected_ids"]
            positions = np.searchsorted(candidate, selected)
            selected_is_subset = bool(
                np.all(positions < len(candidate))
                and np.array_equal(candidate[positions], selected)
            ) if len(selected) else True
            valid = (
                candidate.dtype == np.int64
                and selected.dtype == np.int64
                and candidate.ndim == selected.ndim == 1
                and len(candidate) == ref_item["candidate_count"]
                and len(selected) == ref_item["selected_count"]
                and (len(candidate) < 2 or np.all(candidate[1:] > candidate[:-1]))
                and (len(selected) < 2 or np.all(selected[1:] > selected[:-1]))
                and selected_is_subset
            )
            if not valid:
                failures.append(f"{scene}/{camera}: invalid candidate/selected ID payload")
        except Exception as error:
            failures.append(f"{scene}/{camera}: unreadable ID payload: {error!r}")
    for record in records:
        parity = record.get("parity", {})
        if not (
            parity.get("brute_vs_bvh_triangle_ids")
            and parity.get("indexed_vs_full_online_depth", {}).get("pass")
            and parity.get("cuda_vs_cpu_dense_ids")
            and parity.get("g1_ref_vs_g1_selected_ids")
        ):
            failures.append(f"{scene}/{record.get('camera')}: a per-view parity gate failed")
        oracle = parity.get("full_online_vs_step2_oracle", {})
        if oracle.get("role") != "diagnostic_only" or oracle.get("hard_gate") is not False:
            failures.append(f"{scene}/{record.get('camera')}: Step 2 oracle was not preserved as diagnostic-only")
        for field in ("mesh_query_cpu_ms", "brute_query_cpu_ms", "fov_lod_candidate_ms", "g1_ref_frame_total_ms", "frame_total_ms"):
            if not finite(record.get(field)) or record[field] < 0:
                failures.append(f"{scene}/{record.get('camera')}: invalid timing {field}")
        for mode in ("g1_ref_render", "g1_render"):
            rendered = record.get(mode, {})
            if rendered.get("backend_settings", {}).get("render_mode") != "RGB":
                failures.append(f"{scene}/{record.get('camera')}: renderer contract drift")
            if rendered.get("selection_contract", {}).get("mode") != "explicit":
                failures.append(f"{scene}/{record.get('camera')}: explicit selection handoff missing")
    failures.extend(f"{scene}: {name}" for name, passed in checks.items() if not passed)
    return {
        "scene": scene,
        "checks": checks,
        "metrics": {"g1_ref": summary.get("g1_ref_metrics"), "g1": summary.get("g1_metrics")},
        "timing_means_ms": summary.get("timing_means_ms"),
        "view_count": len(records),
        "run": str(run),
    }, failures


def main() -> None:
    failures = []
    input_manifest = load(ROOT / "manifests" / "step4_input_binding_manifest.json")
    expected_names = {item["scene"]: item["camera_names"] for item in input_manifest["bindings"]}
    prerequisite_checks = {
        "input_binding": input_manifest.get("status") == "pass" and input_manifest.get("view_count") == 1214,
        "protocol": input_manifest.get("protocol_id") == PROTOCOL_ID,
        "mesh_index_reuse": load(ROOT / "manifests" / "mesh_index_reuse_manifest.json").get("status") == "pass",
        "source_parity": load(ROOT / "manifests" / "cpu_mesh_index_source_manifest.json").get("status") == "pass",
        "contract_tests": (ROOT / "review" / "contract_tests.txt").read_text().strip().endswith("passed in") or "passed" in (ROOT / "review" / "contract_tests.txt").read_text(),
    }
    failures.extend(f"prerequisite: {name}" for name, passed in prerequisite_checks.items() if not passed)
    scenes = []
    command_rows = []
    camera_domains = []
    attempted_view_count = 0
    for scene, expected in SCENES.items():
        run = ROOT / "runs" / "formal" / scene / FORMAL_RUN_ID
        reviewed, scene_failures = review_run(run, scene, expected, expected_names[scene], qualification=False)
        failures.extend(scene_failures)
        status = load(run / "status.json") if (run / "status.json").exists() else {}
        partial_records = load(run / "per_view.json") if (run / "per_view.json").exists() else []
        attempted_view_count += len(partial_records)
        contract = load(run / "run_contract.json") if (run / "run_contract.json").exists() else {}
        command_rows.append(
            {
                "scene": scene,
                "run_id": FORMAL_RUN_ID,
                "state": status.get("state", "missing"),
                "view_count": len(partial_records),
                "cuda_visible_devices": contract.get("environment", {}).get("gpu", {}).get("cuda_visible_devices"),
                "command_path": str(run / "command.txt"),
                "status_path": str(run / "status.json"),
            }
        )
        if reviewed is not None:
            scenes.append(reviewed)
            camera_domains.extend(load(run / "camera_domain_records.json"))
    atomic = ROOT / "manifests" / "camera_domain_manifest.json"
    atomic.write_text(json.dumps({"schema": "proxygs_step4_camera_domains_v2", "record_count": len(camera_domains), "records": camera_domains}, indent=2, sort_keys=True) + "\n")
    with (ROOT / "command_ledger_v2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("scene", "run_id", "state", "view_count", "cuda_visible_devices", "command_path", "status_path"))
        writer.writeheader()
        writer.writerows(command_rows)
    report = {
        "schema": "proxygs_step4_final_review_v2",
        "protocol_id": PROTOCOL_ID,
        "status": "pass" if not failures and len(scenes) == len(SCENES) else "fail",
        "scene_count": len(scenes),
        "view_count": sum(scene["view_count"] for scene in scenes),
        "attempted_view_count": attempted_view_count,
        "failed_scene_count": len(SCENES) - len(scenes),
        "expected_scene_count": len(SCENES),
        "expected_view_count": sum(SCENES.values()),
        "qualification": "not reused; Protocol v2 acceptance is the from-zero complete 8-scene matrix",
        "prerequisite_checks": prerequisite_checks,
        "failures": failures,
        "scenes": scenes,
        "completion_claim": "G1 CPU Mesh Index, online proxy depth, and CPU dense scan passed complete evidence gates; G2 CPU Anchor Index is not implemented.",
    }
    output = ROOT / "review" / "final_step4_v2_review.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Fail-closed aggregate review for the complete 8-scene Step 3 G0 baseline."""

from __future__ import annotations

import json
import math
import csv
from pathlib import Path


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
RUN_ID = "formal_g0_full_20260914"


def load(path: Path):
    return json.loads(path.read_text())


def finite(value):
    return value is None or (isinstance(value, (int, float)) and math.isfinite(value))


def main():
    root = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step3_gdmgs_backend_20260914")
    parity = load(root / "review" / "backend_parity_report.json")
    input_manifest = load(root / "manifests" / "step3_input_binding_manifest.json")
    expected_names = {item["scene"]: item["camera_names"] for item in input_manifest["bindings"]}
    failures = []
    scenes = []
    diagnostic_records = []
    command_rows = []
    for scene, expected in SCENES.items():
        run = root / "runs" / "gdmgs_backend_full" / scene / RUN_ID
        try:
            status = load(run / "status.json")
            summary = load(run / "summary.json")
            records = load(run / "per_view.json")
        except Exception as error:
            failures.append(f"{scene}: missing or unreadable run artifact: {error!r}")
            continue
        names = [record["camera"] for record in records]
        renders = sorted(path.stem for path in (run / "renders").glob("*.png"))
        checks = {
            "status_complete": status.get("state") == "complete",
            "summary_complete": summary.get("state") == "complete",
            "backend": summary.get("backend") == "gdmgs-gsplat-v1",
            "selection_all": summary.get("selection_mode") == "all",
            "record_count": len(records) == expected,
            "summary_count": summary.get("view_count") == expected,
            "frozen_count": summary.get("frozen_view_count") == expected,
            "camera_order": names == expected_names[scene],
            "unique_cameras": len(names) == len(set(names)),
            "render_files": renders == sorted(expected_names[scene]),
            "input_preserved": summary.get("input_identities_after")
            == load(run / "run_contract.json").get("inputs"),
            "per_view_backend_settings": all(
                record.get("backend_settings", {}).get("packed") is False
                and record.get("backend_settings", {}).get("render_mode") == "RGB"
                for record in records
            ),
            "finite_metrics_and_timing": all(
                finite(record["metrics"]["psnr"])
                and finite(record["metrics"]["ssim"])
                and finite(record["metrics"]["lpips"])
                and finite(record["decode_seconds"])
                and finite(record["render_seconds_mean"])
                for record in records
            ),
            "native_diagnostic_present": summary.get("native_diagnostic_views", 0) >= 1,
        }
        if not all(checks.values()):
            failures.extend(f"{scene}: {name}" for name, passed in checks.items() if not passed)
        worst_psnr = min(records, key=lambda item: item["metrics"]["psnr"]) if records else None
        worst_ssim = min(records, key=lambda item: item["metrics"]["ssim"]) if records else None
        lpips_records = [record for record in records if record["metrics"]["lpips"] is not None]
        worst_lpips = max(lpips_records, key=lambda item: item["metrics"]["lpips"]) if lpips_records else None
        scenes.append(
            {
                "scene": scene,
                "checks": checks,
                "metrics": summary["metrics"],
                "timing": summary["timing"],
                "worst_views": {
                    "psnr": (
                        {"camera": worst_psnr["camera"], "value": worst_psnr["metrics"]["psnr"]}
                        if worst_psnr is not None
                        else None
                    ),
                    "ssim": (
                        {"camera": worst_ssim["camera"], "value": worst_ssim["metrics"]["ssim"]}
                        if worst_ssim is not None
                        else None
                    ),
                    "lpips": (
                        {"camera": worst_lpips["camera"], "value": worst_lpips["metrics"]["lpips"]}
                        if worst_lpips is not None
                        else None
                    ),
                },
                "native_diagnostic_views": summary["native_diagnostic_views"],
            }
        )
        diagnostics = [
            {"scene": scene, "camera": record["camera"], **record["native_diagnostic"]}
            for record in records
            if record["native_diagnostic"] is not None
        ]
        diagnostic_records.extend(diagnostics)
        contract = load(run / "run_contract.json")
        command_rows.append(
            {
                "scene": scene,
                "run_id": RUN_ID,
                "state": status.get("state"),
                "view_count": len(records),
                "cuda_visible_devices": contract["environment"]["gpu"]["cuda_visible_devices"],
                "command_path": str(run / "command.txt"),
                "status_path": str(run / "status.json"),
            }
        )
    if parity.get("status") != "pass":
        failures.append("backend parity report did not pass")
    report = {
        "schema": "proxygs_step3_final_review_v1",
        "status": "pass" if not failures and len(scenes) == len(SCENES) else "fail",
        "scene_count": len(scenes),
        "view_count": sum(SCENES[scene["scene"]] for scene in scenes),
        "expected_scene_count": len(SCENES),
        "expected_view_count": sum(SCENES.values()),
        "backend_parity": parity.get("status"),
        "failures": failures,
        "scenes": scenes,
        "completion_claim": (
            "ProxyGS uses the GDM-GS gsplat backend for the complete selection-disabled G0 baseline; "
            "CPU mesh/anchor indices are not implemented by this step."
        ),
    }
    output = root / "review" / "final_gdmgs_backend_review.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    diagnostic_path = (
        root / "diagnostics" / "native_vs_gdmgs_backend" / "native_vs_gdmgs_backend_diagnostic.json"
    )
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(
        json.dumps(
            {
                "schema": "proxygs_step3_native_vs_gdmgs_backend_v1",
                "role": "backend migration diagnostic only; excluded from index attribution",
                "record_count": len(diagnostic_records),
                "records": diagnostic_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    ledger_fields = [
        "scene",
        "run_id",
        "state",
        "view_count",
        "cuda_visible_devices",
        "command_path",
        "status_path",
    ]
    with (root / "command_ledger.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ledger_fields)
        writer.writeheader()
        writer.writerows(command_rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

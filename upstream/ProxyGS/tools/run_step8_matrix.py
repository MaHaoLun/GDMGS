"""Launch the frozen eight-scene Step 8 matrix sequentially on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCENES = {
    "amsterdam": (161, "proxy_mesh_20260911:amsterdam:490999092:1789212176500569557"),
    "barcelona": (160, "proxy_mesh_20260911:barcelona:701003468:1789207614899027427"),
    "bilbao": (129, "proxy_mesh_20260911:bilbao:386986940:1789208693675434861"),
    "chicago": (160, "proxy_mesh_20260911:chicago:356287956:1789209689992315892"),
    "hollywood": (125, "proxy_mesh_20260911:hollywood:528543476:1789211808848641393"),
    "pompidou": (161, "proxy_mesh_20260911:pompidou:544336076:1789240251869306554"),
    "quebec": (160, "proxy_mesh_20260911:quebec:529247380:1789226351686149504"),
    "rome": (158, "proxy_mesh_20260911:rome:471337100:1789230896674517787"),
}

ROOT = Path("/ssddata/lun/gdmgs_artifacts")
STEP8 = ROOT / "proxygs_step8_online_schedule_20260916"
STEP2 = ROOT / "proxygs_step2_20260913"
STEP3 = ROOT / "proxygs_step3_gdmgs_backend_20260914"
STEP4 = ROOT / "proxygs_step4_cpu_mesh_index_g1_v2_20260914"
STEP5 = ROOT / "proxygs_step5_cpu_anchor_index_g2_20260915"
STEP6 = ROOT / "proxygs_step6_high_overlap_cpu_index_20260915"
STEP6_DIAG = ROOT / "proxygs_step6_bvh_diagnosis_20260915"
STEP7 = ROOT / "proxygs_step7bc_temporal_20260916"


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def command(scene, run_id, repeats, modes):
    expected, token = SCENES[scene]
    model = STEP2 / f"runs/bungee/{scene}/formal_proxygs_native_40k_20260913"
    result = [
        sys.executable,
        "render_step8_schedule.py",
        "--scene", scene,
        "--source-path", f"/ssddata/lun/data/bungeenerf/{scene}",
        "--model-path", str(model),
        "--mesh-index", str(STEP4 / f"indices/{scene}/mesh_bvh.npz"),
        "--mesh-token", token,
        "--anchor-index", str(STEP5 / f"indices/{scene}/anchor_point_bvh.npz"),
        "--step4-run", str(STEP4 / f"runs/formal/{scene}/formal_step4_g1_v2_20260914"),
        "--step5-run", str(STEP5 / f"runs/formal/{scene}/formal_step5_g2_v1_20260915"),
        "--selection", str(STEP6 / "review/speed_selected_high_overlap_windows.json"),
        "--mesh-window-profile", str(STEP6_DIAG / "qualification/retained_profile_conservative.json"),
        "--renderer-settings", str(STEP3 / "manifests/renderer_settings.json"),
        "--explicit-selection-contract", str(STEP3 / "manifests/explicit_selection_contract.json"),
        "--cache-trace", str(STEP6 / "review/cache_trace_manifest.json"),
        "--selected-cache-contract", str(STEP7 / "review/selected_cache_contract.json"),
        "--step7-confirmation", str(STEP7 / "review/final_confirmation_v1.json"),
        "--mesh-native-dir", str(STEP8 / "native"),
        "--output-root", str(STEP8 / "runs/formal"),
        "--run-id", run_id,
        "--expected-views", str(expected),
        "--leaf-capacity", "4096",
        "--max-depth", "32",
        "--mesh-threads", "16",
        "--index-repeat", "1",
        "--warmup", "0",
        "--repeat", "1",
        "--performance-repeats", str(repeats),
        "--no-save-images",
        "--formal",
    ]
    return result + ["--modes", *modes]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-id", default="formal_step8_schedule_v1_20260919")
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    cli.add_argument("--resume", action="store_true")
    cli.add_argument(
        "--modes",
        nargs="+",
        default=[
            "serial_fresh",
            "serial_fresh_mesh32",
            "serial_cache",
            "scheduled_cache",
            "serial_cache_mesh32",
            "scheduled_cache_q2_mesh32",
        ],
    )
    args = cli.parse_args()
    ledger = []
    ledger_path = STEP8 / "manifests/command_ledger.json"
    for scene in args.scenes:
        output = STEP8 / "runs/formal" / scene / args.run_id
        if args.resume and (output / "summary.json").is_file():
            summary = json.loads((output / "summary.json").read_text())
            if summary.get("status") == "pass":
                ledger.append({"scene": scene, "state": "skipped_existing_pass"})
                atomic_json(ledger_path, ledger)
                continue
        cmd = command(scene, args.run_id, args.repeats, args.modes)
        started = time.time()
        result = subprocess.run(cmd, text=True)
        ledger.append(
            {
                "scene": scene,
                "state": "complete" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "started_unix": started,
                "completed_unix": time.time(),
                "command": cmd,
            }
        )
        atomic_json(ledger_path, ledger)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

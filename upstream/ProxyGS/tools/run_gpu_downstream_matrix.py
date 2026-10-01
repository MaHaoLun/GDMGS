"""Run additive GPU downstream requalification with immutable run directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_step8_matrix import SCENES, command

MODES = [
    "cpu_serial_fresh_mesh32", "cpu_serial_cache_mesh32",
    "cpu_scheduled_cache_q2_mesh32", "gpu_serial_fresh_mesh32",
    "gpu_serial_cache_mesh32", "gpu_scheduled_cache_q1_mesh32",
    "gpu_scheduled_cache_q2_mesh32",
]


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def verify_completed(folder, modes, repeats):
    try:
        summary, contract, performance, status = (
            json.loads((folder / name).read_text())
            for name in ("summary.json", "contract.json", "performance.json", "status.json")
        )
        return (
            summary.get("status") == "pass" and summary.get("frame_count") == 32
            and summary.get("quality_pass") is True
            and summary.get("selection_oracle_exact") is True
            and contract.get("formal") is True and contract.get("modes") == modes
            and len(contract.get("camera_ids", [])) == 32
            and status == {"state": "complete", "status": "pass"}
            and set(performance) == set(modes)
            and all(len(performance[m]) == repeats and all(
                r.get("frame_count") == 32 and r.get("selection_oracle_exact") is True
                and r.get("mode") == m for r in performance[m]
            ) for m in modes)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_scope(phase, scenes, modes, repeats):
    if repeats < 3 or len(set(modes)) != len(modes) or len(set(scenes)) != len(scenes):
        raise ValueError("Require at least three repeats and unique scenes/modes")
    if phase == "formal" and (set(scenes) != set(SCENES) or set(modes) != set(MODES)):
        raise ValueError("Formal acceptance requires all eight scenes and all seven modes")


def require_qualification(root, run_id, sources, modes, repeats):
    source_path = root / "manifests" / (run_id + "_source.json")
    original = json.loads(source_path.read_text())
    if (original.get("sources") != sources or original.get("modes") != modes
            or original.get("repeats") != repeats or original.get("phase") != "qualification"
            or "amsterdam" not in original.get("scenes", [])):
        raise RuntimeError("Qualification code, inputs or protocol differ from formal run")
    if not verify_completed(root / "runs/qualification/amsterdam" / run_id, modes, repeats):
        raise RuntimeError("Complete passing Amsterdam qualification is required before formal execution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--phase", choices=["qualification", "formal"], default="formal")
    parser.add_argument("--qualification-run-id", default="qualification_gpu_downstream_v1_20260921")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    validate_scope(args.phase, args.scenes, args.modes, args.repeats)
    root = args.experiment_root.resolve()
    runtime = Path(__file__).resolve().parents[1]
    output = root / "runs" / args.phase
    manifest = root / "manifests" / (args.run_id + "_ledger.json")
    ledger = json.loads(manifest.read_text()) if manifest.exists() else []
    sources = {}
    for p in sorted(runtime.rglob("*.py")):
        if "__pycache__" not in p.parts:
            sources[str(p.relative_to(runtime))] = hashlib.sha256(p.read_bytes()).hexdigest()
    lineage_path = root / "manifests/step6_gpu_lineage.json"
    lineage = json.loads(lineage_path.read_text())
    for binding in lineage["evidence"].values():
        p = Path(binding["remote_path"])
        if hashlib.sha256(p.read_bytes()).hexdigest() != binding["sha256"]:
            raise RuntimeError(f"Frozen evidence drift: {p}")
    gpu_review = json.loads(Path(lineage["evidence"]["step5_gpu_review"]["remote_path"]).read_text())
    if gpu_review.get("status", "").lower() != "pass" or gpu_review.get("failures") != []:
        raise RuntimeError("Step 5 GPU gate is not passing")
    sources["step6_gpu_lineage.json"] = hashlib.sha256(lineage_path.read_bytes()).hexdigest()
    for name, directory in (
        ("gpu_anchor_native", Path(os.environ["GDMGS_GPU_NATIVE_DIR"])),
        ("mesh_native", Path("/ssddata/lun/gdmgs_artifacts/proxygs_step8_online_schedule_20260916/native")),
    ):
        libraries = sorted(directory.glob("*.so"))
        if not libraries:
            raise RuntimeError(f"Missing {name} extension")
        for p in libraries:
            sources[name + "/" + p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    if args.phase == "formal":
        require_qualification(root, args.qualification_run_id, sources, args.modes, args.repeats)
    source_path = root / "manifests" / (args.run_id + "_source.json")
    proposed = {
        "sources": sources, "modes": args.modes, "repeats": args.repeats,
        "scenes": args.scenes, "phase": args.phase,
    }
    if source_path.exists():
        original = json.loads(source_path.read_text())
        if not args.resume or any(original.get(k) != v for k, v in proposed.items()):
            raise RuntimeError("Existing run provenance differs; preserve it and use a new run ID")
    else:
        if ledger:
            raise RuntimeError("Existing ledger lacks immutable source manifest")
        atomic(source_path, {**proposed, "started_unix": time.time()})
    for scene in args.scenes:
        scene_root = output / scene / args.run_id
        if args.resume and verify_completed(scene_root, args.modes, args.repeats):
            continue
        if scene_root.exists():
            raise RuntimeError(f"Retain existing attempt and choose a new run ID: {scene_root}")
        cmd = command(scene, args.run_id, args.repeats, args.modes)
        cmd[1] = "render_step8_gpu_schedule.py"
        cmd[cmd.index("--output-root") + 1] = str(output)
        item = {"scene": scene, "command": cmd, "started_unix": time.time(), "state": "running"}
        ledger.append(item)
        atomic(manifest, ledger)
        atomic(root / "review" / (args.run_id + "_status.json"), {
            "state": "running", "scene": scene, "scenes": args.scenes,
        })
        result = subprocess.run(cmd, cwd=runtime)
        item.update(completed_unix=time.time(), returncode=result.returncode)
        summary_file = scene_root / "summary.json"
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        passed = result.returncode == 0 and verify_completed(scene_root, args.modes, args.repeats)
        item["state"] = "complete" if passed else "failed"
        atomic(manifest, ledger)
        if not passed:
            atomic(root / "review" / (args.run_id + "_status.json"), {"state": "failed", "scene": scene})
            raise SystemExit(result.returncode or 1)
    atomic(root / "review" / (args.run_id + "_status.json"), {
        "state": "complete", "scenes": args.scenes, "frame_count": len(args.scenes) * 32,
        "modes": args.modes,
    })


if __name__ == "__main__":
    main()

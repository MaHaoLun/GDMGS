"""Run the isolated all-GPU pair2 matrix with immutable per-scene outputs."""
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

MODES = ["cpu_fresh_reference", "gpu_fresh", "gpu_pair2_control", "gpu_pair2_B",
         "gpu_pair2_AB", "gpu_pair2_bitmap"]
ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step7d_pair2_gpu_pipeline_20260923")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(runtime):
    names = [
        "render_step7d_pair2_gpu.py", "render_step8_gpu_schedule.py",
        "online_proxy_depth.py", "step7d_execution_optimizations.py",
        "step7d_gpu_merge.py",
        "gdmgs/mesh_index/index.py", "gdmgs/mesh_index/gpu_index.py",
        "gdmgs/mesh_index/native/gpu_query.cu",
        "gdmgs/anchor_index/gpu_index.py",
        "gdmgs/cache/temporal_bundle_cache_v3.py",
        "gaussian_renderer/gdmgs_gsplat_backend.py",
    ]
    return {name: hash_file(runtime / name) for name in names}


def complete(scene_root, repeats):
    try:
        summary = json.loads((scene_root / "summary.json").read_text())
        performance = json.loads((scene_root / "performance.json").read_text())
        return (summary["status"] == "pass" and summary["frames"] == 32
                and all(len(performance[m]) == repeats for m in
                        ("gpu_fresh", "gpu_pair2_control", "gpu_pair2_B",
                         "gpu_pair2_AB", "gpu_pair2_bitmap"))
                and summary["qualification"]["gpu_fresh"]["fresh_exact"]
                and all(summary["qualification"][m]["payload_metadata_exact"]
                        for m in ("gpu_pair2_B", "gpu_pair2_AB", "gpu_pair2_bitmap")))
    except (OSError, KeyError, ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("qualification", "formal"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scenes", nargs="+", choices=SCENES)
    args = parser.parse_args()
    scenes = args.scenes or (["amsterdam"] if args.phase == "qualification" else list(SCENES))
    if args.repeats < 3 or (args.phase == "formal" and set(scenes) != set(SCENES)):
        raise ValueError("formal matrix requires all eight scenes and >=3 repeats")
    runtime = Path(__file__).resolve().parents[1]
    sources = source_manifest(runtime)
    qualification_manifest = ROOT / "manifests/qualification_source.json"
    if args.phase == "formal":
        prior = json.loads(qualification_manifest.read_text())
        if prior["sources"] != sources or prior["modes"] != MODES or not complete(
                ROOT / "runs/qualification/amsterdam" / prior["run_id"], prior["repeats"]):
            raise RuntimeError("completed same-source Amsterdam qualification required")
    source_file = ROOT / "manifests" / (args.run_id + "_source.json")
    if source_file.exists():
        raise RuntimeError("run ID already exists; preserve earlier attempt")
    manifest = {"run_id": args.run_id, "phase": args.phase, "scenes": scenes,
                "modes": MODES, "repeats": args.repeats, "sources": sources,
                "started_unix": time.time()}
    write(source_file, manifest)
    if args.phase == "qualification":
        write(qualification_manifest, manifest)
    ledger = []
    for scene in scenes:
        output = ROOT / "runs" / args.phase / scene / args.run_id
        if output.exists():
            raise RuntimeError(f"run directory already exists: {output}")
        cmd = command(scene, args.run_id, args.repeats, MODES)
        cmd[1] = "render_step7d_pair2_gpu.py"
        cmd[cmd.index("--output-root") + 1] = str(ROOT / "runs" / args.phase)
        record = {"scene": scene, "command": cmd, "started_unix": time.time(), "state": "running"}
        ledger.append(record)
        write(ROOT / "manifests" / (args.run_id + "_ledger.json"), ledger)
        result = subprocess.run(cmd, cwd=runtime)
        record.update(completed_unix=time.time(), returncode=result.returncode,
                      state="complete" if result.returncode == 0 and complete(output, args.repeats) else "failed")
        write(ROOT / "manifests" / (args.run_id + "_ledger.json"), ledger)
        if record["state"] != "complete":
            raise SystemExit(result.returncode or 1)
    write(ROOT / "manifests" / (args.run_id + "_status.json"),
          {"status": "complete", "scenes": scenes, "frames": 32 * len(scenes)})


if __name__ == "__main__":
    main()

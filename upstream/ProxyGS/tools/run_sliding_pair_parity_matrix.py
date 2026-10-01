"""Immutable four-mode GPU pair-parity sliding matrix launcher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from run_step8_matrix import SCENES, command

ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step7d_sliding_pairparity_20260924")
MODES = ["gpu_fresh", "pair2_control", "slide_pair_age1", "slide_pair_age2"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sources(runtime):
    names = [
        "render_step7d_sliding_pairparity.py",
        "gdmgs/cache/sliding_pair_parity_cache.py",
        "gdmgs/cache/sliding_window_cache.py",
        "render_step7d_pair2_gpu.py", "render_step8_gpu_schedule.py",
        "online_proxy_depth.py", "gdmgs/mesh_index/index.py",
        "gdmgs/mesh_index/gpu_index.py", "gdmgs/mesh_index/native/gpu_query.cu",
        "gdmgs/anchor_index/gpu_index.py",
        "gdmgs/cache/temporal_bundle_cache_v3.py",
        "gaussian_renderer/gdmgs_gsplat_backend.py",
        "tests/test_sliding_pair_parity.py", "tests/test_sliding_window2.py",
        "tools/run_sliding_pair_parity_matrix.py",
    ]
    return {name: hashlib.sha256((runtime / name).read_bytes()).hexdigest() for name in names}


def complete(folder, repeats):
    try:
        summary, quality, perf, contract = (
            json.loads((folder / name).read_text()) for name in
            ("summary.json", "qualification.json", "performance.json", "contract.json"))
        return (summary["status"] == "complete" and summary["frames"] == 32
                and contract["status"] == "complete" and contract["modes"] == MODES
                and quality["gpu_fresh"]["fresh_exact"]["passed"]
                and quality["pair2_control"]["quality"]["passed"]
                and quality["slide_pair_age1"]["payload_metadata_exact"]
                and all(quality[m]["selection_oracle_exact"]
                        and quality[m]["mesh_oracle_exact"] for m in MODES)
                and all(len(perf[m]) == repeats and all(r["frame_count"] == 32 for r in perf[m])
                        for m in MODES))
    except (OSError, KeyError, ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("qualification", "formal"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--scenes", nargs="+", choices=SCENES)
    args = parser.parse_args()
    scenes = args.scenes or (["amsterdam"] if args.phase == "qualification" else list(SCENES))
    if args.repeats < 4 or (args.phase == "formal" and set(scenes) != set(SCENES)):
        raise ValueError("formal pair-parity matrix requires eight scenes and >=4 repeats")
    runtime = Path(__file__).resolve().parents[1]
    source_hashes = sources(runtime)
    if args.phase == "formal":
        manifest = json.loads((ROOT / "manifests/qualification_source.json").read_text())
        if (manifest["sources"] != source_hashes or manifest["modes"] != MODES
                or not complete(ROOT / "runs/qualification/amsterdam" / manifest["run_id"],
                                manifest["repeats"])):
            raise RuntimeError("same-source complete Amsterdam qualification required")
    manifest_path = ROOT / "manifests" / (args.run_id + "_source.json")
    if manifest_path.exists():
        raise RuntimeError("run ID already used; preserve the attempt")
    manifest = {"run_id": args.run_id, "phase": args.phase, "scenes": scenes,
                "repeats": args.repeats, "modes": MODES, "sources": source_hashes,
                "started_unix": time.time()}
    save(manifest_path, manifest)
    ledger = []
    for scene in scenes:
        folder = ROOT / "runs" / args.phase / scene / args.run_id
        if folder.exists():
            raise RuntimeError("existing scene attempt must be preserved: " + str(folder))
        cmd = command(scene, args.run_id, args.repeats, MODES)
        cmd[1] = "render_step7d_sliding_pairparity.py"
        cmd[cmd.index("--output-root") + 1] = str(ROOT / "runs" / args.phase)
        item = {"scene": scene, "command": cmd, "started_unix": time.time(), "state": "running"}
        ledger.append(item)
        save(ROOT / "manifests" / (args.run_id + "_ledger.json"), ledger)
        result = subprocess.run(cmd, cwd=runtime)
        item.update(completed_unix=time.time(), returncode=result.returncode,
                    state="complete" if result.returncode == 0 and complete(folder, args.repeats)
                    else "failed")
        save(ROOT / "manifests" / (args.run_id + "_ledger.json"), ledger)
        if item["state"] != "complete":
            raise SystemExit(result.returncode or 1)
    save(ROOT / "manifests" / (args.run_id + "_status.json"),
         {"state": "complete", "scenes": scenes, "frames": len(scenes) * 32,
          "modes": MODES, "repeats": args.repeats})
    if args.phase == "qualification":
        save(ROOT / "manifests/qualification_source.json", manifest)


if __name__ == "__main__":
    main()

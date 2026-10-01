"""Run bounded profile and partial-refresh screens after formal GPU matrix."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from run_step8_matrix import command

ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step7d_pair2_gpu_pipeline_20260923")
MODES = ["cpu_fresh_reference", "gpu_fresh", "gpu_pair2_control", "gpu_pair2_B",
         "gpu_pair2_AB", "gpu_pair2_bitmap"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("profile", "partial_refresh"))
    args = parser.parse_args()
    scene = "barcelona" if args.kind == "profile" else "hollywood"
    run_id = ("pack_stage_profile_v2_20260923" if args.kind == "profile"
              else "partial_refresh_screen_v1_20260923")
    output_root = ROOT / "screens" / args.kind
    target = output_root / scene / run_id
    if target.exists():
        raise RuntimeError("screen already exists; preserve it and use a new run ID")
    cmd = command(scene, run_id, 3, MODES)
    cmd[1] = "tools/profile_cache_parts.py" if args.kind == "profile" else "tools/screen_partial_refresh.py"
    cmd[cmd.index("--output-root") + 1] = str(output_root)
    ledger = {"kind": args.kind, "scene": scene, "command": cmd,
              "started_unix": time.time(), "state": "running"}
    ROOT.joinpath("screens").mkdir(parents=True, exist_ok=True)
    path = ROOT / "screens" / (args.kind + "_ledger.json")
    path.write_text(json.dumps(ledger, indent=2) + "\n")
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1])
    ledger.update(completed_unix=time.time(), returncode=result.returncode,
                  state="complete" if result.returncode == 0 else "failed")
    path.write_text(json.dumps(ledger, indent=2) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

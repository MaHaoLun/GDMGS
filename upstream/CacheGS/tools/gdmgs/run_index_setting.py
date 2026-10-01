"""Execute and merge the frozen twelve-scene index experiment serially.

The input selection file names already chosen mesh artifacts and parameters.
It does not build meshes, change quality thresholds, or select a smaller set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_index_results import EXPECTED_SCENES
from benchmark_index import write_json


def validate_selection(selection):
    scenes = selection.get("scenes", [])
    names = [scene.get("scene") for scene in scenes]
    if len(names) != len(set(names)) or set(names) != set(EXPECTED_SCENES):
        raise ValueError("The frozen selection must include every in-setting scene once.")
    if not selection.get("run_id") or selection.get("excluded_scenes") != ["small_city"]:
        raise ValueError("Missing run identity or incorrect setting exclusion.")
    if selection.get("query_device") != "gpu":
        raise ValueError("Formal selection must explicitly select the GPU query path; CPU runs remain diagnostic.")
    for scene in scenes:
        if not all(scene.get(key) for key in ("model_path", "mesh", "selection_reason")):
            raise ValueError("Each scene needs a checkpoint, chosen mesh and selection evidence.")
        if "matrixcity" in str(scene["model_path"]).lower():
            raise ValueError("MatrixCity is outside the setting.")
        if not Path(scene["mesh"]).is_file():
            raise FileNotFoundError(scene["mesh"])
        if scene.get("support_profile", "decoder_all_view") not in ("decoder_all_view", "native_anchor_proxy"):
            raise ValueError("Unknown frozen support profile.")
    return scenes


def merge_complete_scenes(output, run_id):
    selection = json.loads((output / "mesh_selection.json").read_text())
    selected = {item["scene"]: item for item in validate_selection(selection)}
    if selection["run_id"] != run_id:
        raise ValueError("Run identity differs from the frozen selection.")
    manifests = []
    for scene in EXPECTED_SCENES:
        root = output / "scenes" / scene
        manifest = json.loads((root / "manifest.json").read_text())
        status = json.loads((root / "status.json").read_text())
        process = json.loads((root / "process.json").read_text())
        if (manifest["run_id"] != run_id or manifest["scope"] != "full_scene"
                or manifest["timing_repeats"] != 3 or len(manifest["scenes"]) != 1
                or manifest["scenes"][0]["scope"] != "full_scene"):
            raise ValueError(f"{scene}: development or incomplete experiment cannot be promoted.")
        actual, frozen = manifest["scenes"][0], selected[scene]
        if (actual["scene"] != scene
                or Path(actual["model_path"]).resolve() != Path(frozen["model_path"]).resolve()
                or actual["iteration"] != frozen.get("iteration", 40000)
                or manifest["settings"]["torch_threads"] != selection.get("torch_threads", 4)
                or manifest["settings"]["warmup_frames"] != selection.get("warmup", 1)
                or manifest["settings"]["query_device"] != selection["query_device"]
                or actual["settings"]["query_device"] != selection["query_device"]
                or actual["settings"]["support_profile"] != frozen.get("support_profile", "decoder_all_view")
                or Path(actual["mesh_source"]).resolve() != Path(frozen["mesh"]).resolve()
                or actual["settings"]["depth_margin"] != frozen.get("depth_margin", .01)
                or actual["settings"]["tile_size"] != frozen.get("tile_size", 8)
                or actual["settings"]["bvh_method"] != frozen.get("bvh_method", "binned_sah")
                or actual["settings"]["mesh_leaf_size"] != frozen.get("mesh_leaf_size", 8)
                or actual["settings"]["anchor_leaf_size"] != frozen.get("anchor_leaf_size", 64)):
            raise ValueError(f"{scene}: actual device/support/mesh/settings differ from the frozen selection.")
        if (status["status"] != "complete" or status.get("scene") != scene
                or status.get("run_id") != run_id or status.get("scope") != "full_scene"
                or process.get("scene") != scene or process.get("run_id") != run_id
                or not status.get("checkpoint_stats_unchanged")
                or status["completed_frames"] != EXPECTED_SCENES[scene]
                or status["expected_frames"] != EXPECTED_SCENES[scene]
                or status["full_trajectory_frames"] != EXPECTED_SCENES[scene]
                or process["returncode"] != 0):
            raise ValueError(f"{scene}: actual process/frame/input evidence is incomplete.")
        manifests.append(manifest)
    first = manifests[0]
    for manifest in manifests[1:]:
        for field in ("settings", "timing_mode_order", "required_modes", "reference_survey_path"):
            if manifest[field] != first[field]:
                raise ValueError(f"Cannot merge changed {field} settings.")
        for field in ("hostname", "cpu_model", "gpu_uuid", "gpu_name", "visible_device"):
            if manifest["machine"][field] != first["machine"][field]:
                raise ValueError("Cannot merge different benchmark hardware.")
    combined = dict(first, scope="full_setting", scenes=[])
    combined["machine_observations"] = []
    for manifest in manifests:
        combined["scenes"].extend(manifest["scenes"])
        combined["machine_observations"].append({"scene": manifest["scenes"][0]["scene"],
                                                 **manifest["machine"]})
    write_json(output / "manifest.json", combined)
    temporary = output / "records.jsonl.partial"
    with temporary.open("w") as destination:
        for scene in EXPECTED_SCENES:
            with (output / "scenes" / scene / "records.jsonl").open() as source:
                for line in source:
                    destination.write(line)
    temporary.replace(output / "records.jsonl")
    return combined


def run(args):
    selection = json.loads(args.selection.read_text())
    scenes = validate_selection(selection)
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "process_state.json"
    if state_path.exists():
        raise ValueError("Run state already exists; inspect its live process before attempting a new run.")
    write_json(args.output / "mesh_selection.json", selection)
    state = {"run_id": selection["run_id"], "pid": os.getpid(), "status": "running",
             "started_unix": time.time(), "completed_scenes": [], "active_scene": None,
             "active_pid": None, "expected_scenes": list(EXPECTED_SCENES), "expected_frames": 2254}
    write_json(state_path, state)
    benchmark = Path(__file__).with_name("benchmark_index.py")
    child = None
    try:
        for scene in scenes:
            name = scene["scene"]
            command = [sys.executable, str(benchmark), "--scene", name,
                       "--model-path", scene["model_path"], "--mesh", scene["mesh"],
                       "--output", str(args.output), "--run-id", selection["run_id"],
                       "--iteration", str(scene.get("iteration", 40000)),
                       "--threads", str(selection.get("torch_threads", 4)),
                       "--ori-backend", "pixel_depth", "--tile-size", str(scene.get("tile_size", 8)),
                       "--query-device", selection["query_device"],
                       "--support-profile", scene.get("support_profile", "decoder_all_view"),
                       "--depth-margin", str(scene.get("depth_margin", .01)),
                       "--bvh-method", scene.get("bvh_method", "binned_sah"),
                       "--mesh-leaf-size", str(scene.get("mesh_leaf_size", 8)),
                       "--anchor-leaf-size", str(scene.get("anchor_leaf_size", 64)),
                       "--repeats", "3", "--warmup", str(selection.get("warmup", 1))]
            log_path = args.output / f"{name}.log"
            started = time.time()
            with log_path.open("w") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                state.update(active_scene=name, active_pid=child.pid)
                write_json(state_path, state)
                print(json.dumps({"scene": name, "pid": child.pid, "log": str(log_path)}), flush=True)
                returncode = child.wait()
            process = {"scene": name, "run_id": selection["run_id"], "pid": child.pid, "returncode": returncode,
                       "started_unix": started, "finished_unix": time.time(), "command": command}
            write_json(args.output / "scenes" / name / "process.json", process)
            if returncode != 0:
                raise RuntimeError(f"{name} process failed with code {returncode}; inspect {log_path}")
            state["completed_scenes"].append(name)
            state.update(active_scene=None, active_pid=None)
            write_json(state_path, state)
        merge_complete_scenes(args.output, selection["run_id"])
        state.update(status="auditing", active_pid=None)
        write_json(state_path, state)
        audit_command = [sys.executable, str(Path(__file__).with_name("audit_index_results.py")),
                         "--run-root", str(args.output), "--output", str(args.output / "audit")]
        with (args.output / "audit.log").open("w") as log:
            audited = subprocess.run(audit_command, stdout=log, stderr=subprocess.STDOUT)
        report = json.loads((args.output / "audit" / "independent_summary.json").read_text())
        if audited.returncode != 0 and report.get("status") == "pass":
            raise RuntimeError("Audit process failed despite a success-shaped summary.")
        state.update(status="complete" if report["status"] == "pass" else "validation_failed",
                     audit_status=report["status"], finished_unix=time.time())
        write_json(state_path, state)
        return state
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_unix=time.time())
        write_json(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = run(args)
    print(json.dumps(state), flush=True)
    return 0 if state["status"] == "complete" and state["audit_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

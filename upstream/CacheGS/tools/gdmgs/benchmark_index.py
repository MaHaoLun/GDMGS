"""Run real paired, cache-off mesh/index experiments without content digests.

One invocation processes one complete scene. Development invocations may name
explicit frame IDs, are marked development, and cannot pass the final audit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np


ORDERS = [["R0", "R1", "R2"], ["R1", "R2", "R0"], ["R2", "R0", "R1"]]
QUALITY = {"psnr_drop_db": 0.1, "ssim_drop": 0.002,
           "preprocessing": "torchvision_png_equivalent", "ssim_window": 11, "ssim_sigma": 1.5}


def validate_mesh_binding(mesh_path, metadata, model_path, iteration, frames, checkpoint_stats):
    """Compare existing mesh/export records; no duplicate source scan or digest."""
    model_path = str(Path(model_path).resolve())
    if str(Path(metadata["model_path"]).resolve()) != model_path or metadata["iteration"] != iteration:
        raise ValueError("Mesh source checkpoint or iteration differs from this experiment.")
    if str(Path(metadata["mesh_file"]).resolve()) != str(Path(mesh_path).resolve()):
        raise ValueError("Mesh metadata names a different artifact.")
    depth_path = Path(metadata["depth_manifest"])
    depth = json.loads(depth_path.read_text())
    if (depth["status"] != "complete" or not depth.get("checkpoint_stats_unchanged")
            or str(Path(depth["model_path"]).resolve()) != model_path
            or depth["iteration"] != iteration or depth["run_token"] != metadata["depth_run_token"]):
        raise ValueError("Mesh depth source is incomplete or belongs to another checkpoint.")
    if depth["checkpoint_stats"] != checkpoint_stats:
        raise ValueError("Mesh source checkpoint metadata differs from the loaded input.")
    bare_frames = [{k: v for k, v in frame.items() if k != "camera_token"} for frame in frames]
    if depth["evaluation_frames"] != bare_frames:
        raise ValueError("Mesh source camera calibration/order differs from this experiment.")
    fusion = [dict(frame, fusion_index=i) for i, frame in enumerate(
        frame for frame in bare_frames if frame["split"] == "train")]
    if metadata["fusion_frames"] != fusion or depth["fusion_frames"] != fusion:
        raise ValueError("Mesh does not use the complete frozen fusion-view collection.")
    completed = depth["completed"]
    if len(completed) != len(fusion) or any(
            actual["frame_index"] != expected["frame_index"] or actual["fusion_index"] != expected["fusion_index"]
            for actual, expected in zip(completed, fusion)):
        raise ValueError("Depth source contains missing or reordered observations.")
    if not metadata.get("mesh_token"):
        raise ValueError("Mesh token is missing.")
    return {"model_path": model_path, "iteration": iteration, "mesh_token": metadata["mesh_token"],
            "depth_manifest": str(depth_path.resolve()), "validated_depth_binding": True}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def machine_record():
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not device or "," in device:
        raise ValueError("Select exactly one CUDA_VISIBLE_DEVICES device before benchmarking.")
    output = subprocess.check_output(["nvidia-smi", "-i", device,
        "--query-gpu=uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    fields = [item.strip() for item in output.split(",")]
    return {"hostname": platform.node(), "cpu_model": cpu, "gpu_uuid": fields[0],
            "gpu_name": fields[1], "visible_device": device,
            "initial_memory_mib": int(fields[2]), "initial_utilization_percent": int(fields[3]),
            "note": "Shared host; record occupancy and paired timings, no claim of exclusive machine use."}


def timing_record(args, frame, scene, mode, repeat, result):
    return {"run_id": args.run_id, "scene": args.scene, "frame_index": frame["frame_index"],
            "camera_token": frame["camera_token"], "settings_token": scene["settings_token"],
            "mode": mode, "repeat": repeat, "status": "complete",
            "timings_ms": result.timings_ms, "counts": result.counters}


def quality_arrays(args, frame, outputs, repeat_r0, camera, mesh_token):
    def cpu_array(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return value
    arrays = {"run_id": np.array(args.run_id), "scene": np.array(args.scene),
              "frame_index": np.int64(frame["frame_index"]), "camera_token": np.array(frame["camera_token"]),
              "mesh_token": np.array(mesh_token), "gt": camera.original_image[:3].detach().cpu().numpy().astype(np.float32)}
    for mode in ("R0", "R1", "R2"):
        result = outputs[mode]
        arrays[mode.lower()] = result.package["render"].detach().cpu().numpy().astype(np.float32)
    arrays["r0_repeat"] = repeat_r0.package["render"].detach().cpu().numpy().astype(np.float32)
    arrays["fov_ids"] = cpu_array(outputs["R0"].fov_ids)
    for mode in ("R1", "R2"):
        result = outputs[mode]
        prefix = mode.lower()
        arrays[prefix + "_fov_ids"] = cpu_array(result.fov_ids)
        arrays[prefix + "_selected_ids"] = cpu_array(result.selected_ids)
        arrays[prefix + "_decoded_anchor_ids"] = result.package["decoded_anchor_ids"].detach().cpu().numpy()
        arrays[prefix + "_raw_ranges"] = cpu_array(result.raw_ranges)
        arrays[prefix + "_formal_ranges"] = cpu_array(result.formal_ranges)
    arrays["scan_triangle_ids"] = cpu_array(outputs["R1"].triangle_ids)
    arrays["bvh_triangle_ids"] = cpu_array(outputs["R2"].triangle_ids)
    return arrays


def preliminary_gpu_quality(outputs, repeat_r0, camera):
    """Fast progress checks, outside timing; final raw CPU audit is authoritative."""
    import torch
    from utils.loss_utils import ssim

    device = outputs["R0"].package["render"].device
    raw = {mode.lower(): outputs[mode].package["render"].detach() for mode in ("R0", "R1", "R2")}
    raw["r0_repeat"] = repeat_r0.package["render"].detach()
    raw["gt"] = camera.original_image[:3].to(device=device, dtype=torch.float32)
    scores = {}
    with torch.no_grad():
        quantized = {key: (value * 255 + .5).clamp(0, 255).to(torch.uint8).float().div(255)
                     for key, value in raw.items()}
        for name in ("r0", "r0_repeat", "r1", "r2"):
            if not bool(torch.isfinite(raw[name]).all()):
                raise ValueError("Nonfinite rendered pixels")
            mse = (quantized[name] - quantized["gt"]).square().mean()
            scores[name] = {"psnr_db": float((-10 * torch.log10(mse)).item()),
                            "ssim": float(ssim(quantized[name][None], quantized["gt"][None]).item())}
    quality = {}
    for name in ("r0_repeat", "r1", "r2"):
        first, second = scores["r0"]["psnr_db"], scores[name]["psnr_db"]
        drop = 0.0 if first == second else first - second
        ssim_drop = scores["r0"]["ssim"] - scores[name]["ssim"]
        quality[name] = {"psnr_drop_db": drop, "ssim_drop": ssim_drop,
                         "pass": drop <= QUALITY["psnr_drop_db"] and ssim_drop <= QUALITY["ssim_drop"]}
    return {"metrics": scores, "quality": quality,
            "scope": "preliminary_gpu_progress_only; final independent raw CPU audit required"}


def run_scene(args):
    import torch
    from survey_inputs import camera_infos, enumerate_records, file_stat, load_model, read_config
    from gdmgs.runtime.session import InferenceSession
    from gdmgs.mesh_index import MeshIndex
    from gdmgs.unified_index import AnchorIndex, SupportSettings
    from gdmgs.index_pipeline import IndexPipeline
    from audit_index_results import json_safe

    if args.scene == "small_city" or "matrixcity" in str(args.model_path).lower():
        raise ValueError("MatrixCity / Smart City is outside the user-defined setting.")
    if os.environ.get("CACHE_ENABLE", "0") not in ("", "0") or os.environ.get("PRECOMP_INDICES_PATH"):
        raise ValueError("Legacy cache and precompute must be disabled.")
    if args.output.resolve().is_relative_to(args.model_path.resolve()):
        raise ValueError("Experiment output must not be in a checkpoint directory.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    scene_root = args.output / "scenes" / args.scene
    if scene_root.exists():
        raise ValueError("Scene output already exists; inspect/resume explicitly or choose a new run root.")
    scene_root.mkdir(parents=True)
    quality_root = args.output / "quality" / args.scene
    quality_root.mkdir(parents=True, exist_ok=True)
    dataset, opt, pipe = read_config(args.model_path)
    if args.output.resolve().is_relative_to(Path(dataset.source_path).resolve()):
        raise ValueError("Experiment output must not be inside the source dataset.")
    splits, kind = camera_infos(dataset)
    from utils.camera_utils import loadCam
    groups, frames = enumerate_records(dataset, splits, kind)
    for frame in frames:
        frame["camera_token"] = f"{args.run_id}:{args.scene}:frame-{frame['frame_index']}"
    checkpoint = args.model_path / "point_cloud" / f"iteration_{args.iteration}"
    before = [file_stat(p) for p in [args.model_path / "config.yaml", *sorted(checkpoint.iterdir())] if p.is_file()]
    machine = machine_record()
    setup_started = time.perf_counter()
    model = load_model(dataset, opt, args.iteration, len(splits["train"]))
    session = InferenceSession(model, str(args.model_path), args.iteration)
    model_load_seconds = time.perf_counter() - setup_started
    background = torch.ones(3, device="cuda") if dataset.white_background else torch.zeros(3, device="cuda")
    if dataset.random_background:
        background = torch.rand(3, device="cuda")
    dataset.data_device = "cpu"

    with np.load(args.mesh, allow_pickle=False) as data:
        vertices = np.ascontiguousarray(data["vertices"], dtype=np.float64)
        key = "faces" if "faces" in data else "triangles"
        triangles = np.ascontiguousarray(data[key], dtype=np.int64)
    mesh_source = json.loads(args.mesh.with_suffix(".json").read_text())
    mesh_binding = validate_mesh_binding(args.mesh, mesh_source, args.model_path, args.iteration, frames, before)
    if mesh_source["vertices"] != len(vertices) or mesh_source["triangles"] != len(triangles):
        raise ValueError("Mesh triangle/vertex counts differ from its source record.")
    mesh_token = mesh_binding["mesh_token"]
    mesh = MeshIndex(vertices, triangles, method=args.bvh_method, leaf_size=args.mesh_leaf_size, mesh_token=mesh_token)
    mesh_path = scene_root / "mesh_bvh.npz"
    mesh.save(mesh_path)
    mesh = MeshIndex.load(mesh_path, mesh_token=mesh_token)
    support_settings = (SupportSettings(profile="native_anchor_proxy_v1")
                        if args.support_profile == "native_anchor_proxy" else SupportSettings())
    anchor = AnchorIndex.from_finalized(session.scene, leaf_capacity=args.anchor_leaf_size,
                                       support_settings=support_settings)
    anchor_path = scene_root / "anchor_index.npz"
    anchor.save(anchor_path)
    anchor = AnchorIndex.load(anchor_path, finalized_scene=session.scene)
    dfs_path = scene_root / "dfs_to_row.npy"
    np.save(dfs_path, anchor.dfs_to_row)
    scope = "development" if args.frames or args.development else "full_scene"
    scene = {"scene": args.scene, "scope": scope, "model_path": str(args.model_path), "iteration": args.iteration,
             "kind": kind, "frames": frames, "checkpoint_stats": before, "mesh_token": mesh_token,
             "index_token": anchor.index_token, "settings_token": f"{args.run_id}:{args.scene}:settings-v1",
             "dfs_to_row_path": str(dfs_path.relative_to(args.output)),
             "mesh_source": str(args.mesh.resolve()), "mesh_source_metadata": mesh_source,
             "settings": {"ori_shape": [args.ori_height, args.ori_width], "depth_margin": args.depth_margin,
                          "ori_definition": "native_pixel_depth_tiles_v1" if args.ori_backend == "pixel_depth" else "continuous_exact_reference",
                          "ori_image_size_source": "frame.render_dimensions",
                          "ori_backend": args.ori_backend, "tile_size": args.tile_size,
                          "materialization": "fresh-no-cache-metadata-v1",
                          "query_device": args.query_device, "support_profile": args.support_profile,
                          "bvh_method": args.bvh_method, "mesh_leaf_size": args.mesh_leaf_size,
                          "anchor_leaf_size": args.anchor_leaf_size, "support": asdict(anchor.support_settings)},
             "build_load": {"model_load_seconds": model_load_seconds, "mesh_build_ms": mesh.build_ms,
                            "mesh_load_ms": mesh.load_ms, "anchor_build_seconds": anchor.build_seconds,
                            "anchor_load_seconds": anchor.load_seconds},
             "vertices": len(vertices), "triangles": len(triangles), "anchors": anchor.anchor_count}
    manifest = {"schema_version": 1, "run_id": args.run_id, "excluded_scenes": ["small_city"],
                "required_modes": ["R0", "R1", "R2"], "timing_repeats": args.repeats,
                "timing_mode_order": ORDERS[:args.repeats], "machine": machine,
                "reference_survey_path": "/ssddata/lun/gdmgs_artifacts/ab_validation_20260905/new_survey.json",
                "settings": {"cpu_threads": 1, "torch_threads": args.threads,
                             "thread_environment": {key: os.environ.get(key) for key in
                                                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                             "query_device": args.query_device,
                             "online_quality_checks": "pytorch_gpu_preliminary; independent_raw_cpu_final",
                             "cache_enabled": False, "precompute_enabled": False, "scheduling_enabled": False,
                             "quality": QUALITY, "warmup_frames": args.warmup},
                "scope": scope, "scenes": [scene]}
    write_json(scene_root / "manifest.json", manifest)
    selected = [int(value) for value in args.frames.split(",")] if args.frames else list(range(len(frames)))
    if len(set(selected)) != len(selected) or any(index < 0 or index >= len(frames) for index in selected):
        raise ValueError("Development frame IDs must be unique and inside the full trajectory.")
    pipeline = IndexPipeline(model, args.model_path, args.iteration, mesh, anchor,
                             ori_shape=(args.ori_height, args.ori_width), depth_margin=args.depth_margin,
                             session=session, mesh_source_record=mesh_binding,
                             ori_backend=args.ori_backend, tile_size=args.tile_size, query_device=args.query_device)
    scene["build_load"]["ori_gpu_initialization_ms"] = pipeline.ori_initialization_ms
    scene["build_load"]["query_gpu_initialization_ms"] = pipeline.query_gpu_initialization_ms
    scene["scene_token"] = session.scene.token
    scene["mesh_query_index_token"] = pipeline.mesh_query_index_token
    scene["anchor_nodes"] = len(anchor.intervals)
    write_json(scene_root / "manifest.json", manifest)
    status = {"status": "running", "run_id": args.run_id, "scene": args.scene,
              "scope": manifest["scope"], "expected_frames": len(selected), "completed_frames": 0,
              "full_trajectory_frames": len(frames), "quality_failures": 0}
    write_json(scene_root / "status.json", status)
    try:
        first = frames[selected[0]]
        warm_camera = loadCam(dataset, first["uid"], splits[first["split"]][first["uid"]], first["resolution_scale"], background)
        with (scene_root / "warmup.jsonl").open("w") as warm_stream:
            for warm_repeat in range(args.warmup):
                for mode in ORDERS[0]:
                    warm_result = pipeline.render(warm_camera, pipe, background, mode, camera_token=first["camera_token"])
                    warm_stream.write(json.dumps({"scope": "warmup", "mode": mode,
                        "repeat": warm_repeat, "frame_index": first["frame_index"],
                        "timings_ms": warm_result.timings_ms, "counts": warm_result.counters}, allow_nan=False) + "\n")
                    warm_stream.flush()
                    del warm_result
        del warm_camera
        with (scene_root / "records.jsonl").open("w") as record_stream:
            for frame_index in selected:
                frame = frames[frame_index]
                camera = loadCam(dataset, frame["uid"], splits[frame["split"]][frame["uid"]], frame["resolution_scale"], background)
                outputs = {}
                for repeat in range(args.repeats):
                    for mode in ORDERS[repeat]:
                        result = pipeline.render(camera, pipe, background, mode, camera_token=frame["camera_token"], diagnostics=repeat == 0)
                        record_stream.write(json.dumps(timing_record(args, frame, scene, mode, repeat, result), allow_nan=False) + "\n")
                        record_stream.flush()
                        if repeat == 0:
                            outputs[mode] = result
                        else:
                            del result
                repeat_r0 = pipeline.render(camera, pipe, background, "R0", camera_token=frame["camera_token"])
                arrays = quality_arrays(args, frame, outputs, repeat_r0, camera, mesh_token)
                quality_file = quality_root / f"{frame_index:06d}.npz"
                np.savez(quality_file, **arrays)
                selection = {"fov": len(arrays["fov_ids"]),
                             "r1_selected": len(arrays["r1_selected_ids"]),
                             "r2_selected": len(arrays["r2_selected_ids"]),
                             "triangles": len(arrays["scan_triangle_ids"])}
                scores = preliminary_gpu_quality(outputs, repeat_r0, camera)
                quality_pass = all(value["pass"] for value in scores["quality"].values())
                row = {"frame_index": frame_index, "selection": selection, "quality": scores,
                       "quality_pass": quality_pass, "quality_path": str(quality_file.relative_to(args.output)),
                       "times_ms": {mode: outputs[mode].timings_ms for mode in ORDERS[0]},
                       "counters": {mode: outputs[mode].counters for mode in ORDERS[0]}}
                status["completed_frames"] += 1
                status["quality_failures"] += int(not quality_pass)
                write_json(scene_root / "status.json", status)
                # JSON cannot represent infinite PSNR for exact equality; the
                # independent auditor retains raw arrays and recomputes it.
                with (scene_root / "quality.jsonl").open("a") as quality_stream:
                    quality_stream.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
                print(json.dumps({"scene": args.scene, "completed": status["completed_frames"],
                                  "frames": len(selected), "quality_pass": quality_pass, "selection": selection,
                                  "total_ms": {mode: outputs[mode].timings_ms["total"] for mode in ORDERS[0]}}), flush=True)
                del arrays, repeat_r0, outputs, camera, scores
        status.update(status="complete", quality_pass=status["quality_failures"] == 0,
                      quality_check_scope="preliminary_gpu; independent_raw_cpu_audit_required",
                      checkpoint_stats_unchanged=before == [file_stat(row["path"]) for row in before],
                      max_cuda_memory_allocated=torch.cuda.max_memory_allocated(),
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if not status["checkpoint_stats_unchanged"]:
            raise RuntimeError("Checkpoint metadata changed during experiment.")
        write_json(scene_root / "status.json", status)
        return status
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(scene_root / "status.json", status)
        raise
    finally:
        pipeline.close()
        for group in splits.values():
            for info in group:
                info.image.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--iteration", type=int, default=40000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ori-height", type=int, default=128)
    parser.add_argument("--ori-width", type=int, default=128)
    parser.add_argument("--ori-backend", choices=("pixel_depth", "exact_reference"), default="pixel_depth")
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--query-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--support-profile", choices=("decoder_all_view", "native_anchor_proxy"), default="decoder_all_view")
    parser.add_argument("--depth-margin", type=float, default=.01)
    parser.add_argument("--bvh-method", choices=("median", "binned_sah"), default="binned_sah")
    parser.add_argument("--mesh-leaf-size", type=int, default=8)
    parser.add_argument("--anchor-leaf-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--frames", help="Explicit development frame IDs; never full experiment acceptance")
    parser.add_argument("--development", action="store_true", help="Candidate evaluation, including complete-scene development runs")
    args = parser.parse_args()
    if args.threads <= 0 or args.warmup < 0:
        parser.error("Invalid thread or warmup count")
    if not (args.frames or args.development) and args.repeats != 3:
        parser.error("Formal complete-scene experiments require all three timing repetitions")
    if not (args.frames or args.development) and args.ori_backend != "pixel_depth":
        parser.error("The slow exact ORI is a development reference, not the adopted experiment definition")
    print(json.dumps(run_scene(args)), flush=True)


if __name__ == "__main__":
    main()

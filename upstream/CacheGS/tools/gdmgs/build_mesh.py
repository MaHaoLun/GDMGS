"""Export full frozen depth observations, build candidates, and measure geometry.

Example: CUDA_VISIBLE_DEVICES=3 python tools/gdmgs/build_mesh.py export ...
Offline geometry measurements do not replace the full R0/R1/R2 image gate.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from gdmgs.geometry.depth_mesh import (DepthFrame, TriangleTable, backproject,
    assert_boundary_preserved, compact_triangles, cross_view_support, grid_triangles, load_triangle_table,
    mesh_topology, validate_depth_source, weld_same_source_vertices,
    save_triangle_table, valid_depth)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def export_depth(args):
    import torch
    from survey_inputs import camera_infos, enumerate_records, file_stat, load_model, read_config
    from gdmgs.pipeline import FreshPipeline
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("An explicit CUDA_VISIBLE_DEVICES lease is required")
    dataset, opt, pipe = read_config(args.model_path)
    if args.output.resolve().is_relative_to(Path(args.model_path).resolve()) or args.output.resolve().is_relative_to(Path(dataset.source_path).resolve()):
        raise ValueError("Output must be separate from checkpoint and dataset inputs")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "depth"
    output.mkdir(exist_ok=True)
    manifest_path = args.output / "depth_manifest.json"
    if manifest_path.exists():
        raise ValueError("Depth manifest already exists; select a new explicit output run")
    checkpoints = [Path(args.model_path) / "config.yaml"] + sorted((Path(args.model_path) / "point_cloud" / f"iteration_{args.iteration}").iterdir())
    before = [file_stat(p) for p in checkpoints if p.is_file()]
    splits, kind = camera_infos(dataset)
    from utils.camera_utils import loadCam
    groups, records = enumerate_records(dataset, splits, kind)
    fusion = [dict(record, fusion_index=i) for i, record in enumerate(record for record in records if record["split"] == "train")]
    manifest = {"status": "running", "run_token": args.run_token,
                "model_path": str(args.model_path), "source_path": dataset.source_path,
                "iteration": args.iteration, "kind": kind, "evaluation_frames": records,
                "fusion_frames": fusion, "fusion_policy": "all enumerated train cameras at original configured render dimensions",
                "depth_source": "frozen fresh RGB+ED expected camera-z depth; not ground truth or Proxy-GS reconstruction",
                "pixel_center": .5, "camera_convention": "W2C column-vector world-to-camera; positive camera z",
                "alpha_policy": "export all finite signals; constructor independently applies recorded confidence gate",
                "checkpoint_stats": before, "completed": []}
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    torch.manual_seed(0)
    model = load_model(dataset, opt, args.iteration, len(splits["train"]))
    pipeline = FreshPipeline(model, str(args.model_path), args.iteration)
    bg = torch.ones(3, device="cuda") if dataset.white_background else torch.zeros(3, device="cuda")
    if dataset.random_background:
        bg = torch.rand(3, device="cuda")
    manifest["background_rgb"] = bg.cpu().tolist()
    dataset.data_device = "cpu"
    for position, record in enumerate(fusion):
        begin = time.perf_counter()
        info = splits[record["split"]][record["uid"]]
        camera = loadCam(dataset, record["uid"], info, record["resolution_scale"], bg)
        with torch.no_grad():
            package = pipeline.render(camera, pipe, bg, "RGB+ED")
        torch.cuda.synchronize()
        depth = package["render_depth"].detach().cpu().numpy()[0].copy()
        alpha = package["render_alpha"].detach().cpu().numpy()[0].copy()
        rgb = package["render"].detach().cpu().numpy().transpose(1, 2, 0).copy()
        if not all(np.isfinite(value).all() for value in (depth, alpha, rgb)):
            raise ValueError(f"Nonfinite source signal in frame {record['frame_index']}")
        h, w = depth.shape
        intrinsics = np.array([[w / (2 * math.tan(camera.FoVx / 2)), 0, w / 2],
                               [0, h / (2 * math.tan(camera.FoVy / 2)), h / 2], [0, 0, 1]], dtype=np.float64)
        world_to_camera = camera.world_view_transform.T.detach().cpu().double().numpy()
        path = output / f"frame_{position:05d}.npz"
        np.savez(path, depth=depth, alpha=alpha, rgb=rgb, intrinsics=intrinsics,
                 world_to_camera=world_to_camera, frame_index=np.int64(record["frame_index"]))
        valid = (depth > 0) & (alpha >= .995)
        values = depth[valid]
        row = {"fusion_index": position, "frame_index": record["frame_index"], "file": str(path.resolve()),
               "image_name": record["image_name"], "shape": [h, w], "high_confidence_pixels": int(valid.sum()),
               "pixels": int(valid.size), "depth_quantiles": np.quantile(values, [0, .5, .95, .99, 1]).tolist() if len(values) else [],
               "seconds": time.perf_counter() - begin, "file_bytes": path.stat().st_size}
        manifest["completed"].append(row)
        write_json(manifest_path, manifest)
        print(json.dumps({"phase": "export", "completed": position + 1, "total": len(fusion), **row}), flush=True)
        del package, camera, depth, alpha, rgb
    pipeline.close()
    del pipeline, model
    gc.collect()
    torch.cuda.empty_cache()
    unchanged = before == [file_stat(item["path"]) for item in before]
    if not unchanged:
        raise ValueError("Checkpoint stat inventory changed during read-only export")
    for infos in splits.values():
        for info in infos:
            info.image.close()
    manifest.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                    checkpoint_stats_unchanged=unchanged, fusion_count=len(fusion),
                    cuda_max_memory_allocated=int(torch.cuda.max_memory_allocated()))
    write_json(manifest_path, manifest)


def export_scene_set(args):
    """Sequential independent processes release each scene's CUDA allocations."""
    inventory = json.loads(args.scene_inventory.read_text())
    selected = args.scenes.split(",")
    if len(set(selected)) != len(selected) or "small_city" in selected:
        raise ValueError("Explicit unique in-setting scene names required; small_city is excluded")
    by_name = {}
    for row in inventory["scenes"]:
        name = Path(row["source_path"]).name
        if name in selected:
            if name in by_name:
                raise ValueError(f"Ambiguous source scene {name}")
            by_name[name] = row
    if set(by_name) != set(selected):
        raise ValueError("Requested scenes do not match source inventory")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / (args.run_token + "_export_set.json")
    if report_path.exists():
        raise ValueError("Export set label already exists; preserve its evidence")
    report = {"status": "running", "run_token": args.run_token,
              "source_inventory": str(args.scene_inventory.resolve()), "scene_order": selected,
              "excluded": ["small_city"], "scene_results": []}
    write_json(report_path, report)
    for name in selected:
        model_path = by_name[name]["model_path"]
        output = args.output / name
        output.mkdir(exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "export", "--model-path", model_path,
            "--iteration", str(args.iteration), "--output", str(output), "--run-token", args.run_token + "_" + name + "_depth"]
        started = time.perf_counter()
        with (output / "export.log").open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            report["active"] = {"scene": name, "pid": process.pid, "command": command,
                                "log": str((output / "export.log").resolve())}
            write_json(report_path, report)
            print(json.dumps(report["active"]), flush=True)
            result = process.wait()
        item = {"scene": name, "returncode": result, "seconds": time.perf_counter() - started,
                "depth_manifest": str((output / "depth_manifest.json").resolve())}
        report["scene_results"].append(item)
        report.pop("active", None)
        write_json(report_path, report)
        print(json.dumps(item), flush=True)
        if result:
            report["status"] = "failure"
            write_json(report_path, report)
            raise RuntimeError(f"Depth export failed for {name}; inspect retained log")
    report["status"] = "complete"
    write_json(report_path, report)


def load_depth_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest["status"] != "complete" or len(manifest["completed"]) != len(manifest["fusion_frames"]):
        raise ValueError("Complete fusion inventory required before mesh construction")
    for expected, actual in zip(manifest["fusion_frames"], manifest["completed"]):
        if (expected["frame_index"] != actual["frame_index"] or expected["fusion_index"] != actual["fusion_index"]):
            raise ValueError("Fusion observation order or identity mismatch")
        if not Path(actual["file"]).is_file():
            raise FileNotFoundError(actual["file"])
    return manifest


def source_metadata(args, manifest, started):
    return {"method": args.method, "status": "candidate_not_adopted", "run_token": args.run_token,
            "model_path": manifest["model_path"], "iteration": manifest["iteration"],
            "depth_manifest": str(args.depth_manifest.resolve()),
            "depth_run_token": manifest["run_token"], "fusion_frames": manifest["fusion_frames"],
            "settings": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
            "depth_source": manifest["depth_source"], "pixel_center": .5,
            "construction_seconds": time.perf_counter() - started,
            "surface_sidedness": "two-sided for occlusion; geometric orientation does not certify opacity"}


def build_grid(args, manifest):
    started = time.perf_counter()
    frames = [DepthFrame.load(row["file"]) for row in manifest["completed"]]
    centers = np.stack([np.linalg.inv(frame.world_to_camera)[:3, 3] for frame in frames])
    chunks, face_chunks, confidences, source_ids, support_counts = [], [], [], [], []
    total_vertices = 0
    per_frame = []
    for i, frame in enumerate(frames):
        vertices, faces, confidence = grid_triangles(frame, stride=args.grid_stride,
            alpha_threshold=args.alpha_threshold, relative_depth_jump=args.relative_depth_jump)
        initial = len(faces)
        support = np.zeros(len(faces), dtype=np.int16)
        conflicts = np.zeros(len(faces), dtype=np.int16)
        # Both triangles' vertices must be supported. Centroid-only agreement
        # could retain a triangle crossing a depth discontinuity in another view.
        points = vertices[faces].reshape(-1, 3)
        neighbors = np.argsort(np.linalg.norm(centers - centers[i], axis=1), kind="stable")
        neighbors = [j for j in neighbors if j != i][:args.consistency_views]
        for j in neighbors:
            agrees, conflict = cross_view_support(points, frames[j],
                relative_tolerance=args.consistency_tolerance, alpha_threshold=args.alpha_threshold)
            support += agrees.reshape(-1, 3).all(axis=1)
            conflicts += conflict.reshape(-1, 3).any(axis=1)
        keep = (support >= args.minimum_support_views) & (conflicts <= args.maximum_conflict_views)
        faces, confidence, support = faces[keep], confidence[keep], support[keep]
        vertices, faces, nondegenerate = compact_triangles(vertices, faces)
        confidence, support = confidence[nondegenerate], support[nondegenerate]
        chunks.append(vertices)
        face_chunks.append(faces + total_vertices)
        confidences.append(confidence)
        source_ids.append(np.full(len(faces), frame.frame_index, dtype=np.int64))
        support_counts.append(support)
        total_vertices += len(vertices)
        row = {"frame_index": frame.frame_index, "candidate_triangles": initial,
               "retained_triangles": len(faces), "neighbor_frame_indices": [frames[j].frame_index for j in neighbors]}
        per_frame.append(row)
        print(json.dumps({"phase": "grid", "completed": i + 1, "total": len(frames), **row}), flush=True)
    vertices, faces = np.concatenate(chunks), np.concatenate(face_chunks)
    metadata = source_metadata(args, manifest, started)
    metadata.update(per_frame=per_frame, boundary_policy="whole source rectangles checked; incomplete outer strips and all rejected cells remain holes; no cross-view welding or filling")
    table = TriangleTable(vertices, faces, metadata, args.run_token + ":grid")
    save_triangle_table(args.output, table, source_frame=np.concatenate(source_ids),
                        confidence_alpha=np.concatenate(confidences), support_view_count=np.concatenate(support_counts))


def build_tsdf(args, manifest):
    import open3d as o3d
    started = time.perf_counter()
    medians = [row["depth_quantiles"][1] for row in manifest["completed"] if row["depth_quantiles"]]
    if not medians:
        raise ValueError("No high-confidence positive depth observations")
    voxel = args.voxel_length or float(np.median(medians)) * args.voxel_relative_depth
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel,
        sdf_trunc=voxel * args.truncation_voxels,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        depth_sampling_stride=args.integration_stride)
    for i, row in enumerate(manifest["completed"]):
        frame = DepthFrame.load(row["file"])
        mask = valid_depth(frame, args.alpha_threshold)
        depth = np.where(mask, frame.depth, 0).astype(np.float32)
        h, w = depth.shape
        # Open3D backprojects integer pixel coordinates, unlike gsplat + .5.
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, frame.intrinsics[0, 0], frame.intrinsics[1, 1],
            frame.intrinsics[0, 2] - .5, frame.intrinsics[1, 2] - .5)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8)), o3d.geometry.Image(depth),
            depth_scale=1., depth_trunc=float(max(depth.max() * 1.01, 1.)), convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, frame.world_to_camera)
        print(json.dumps({"phase": "tsdf", "completed": i + 1, "total": len(manifest["completed"]),
                          "frame_index": frame.frame_index, "voxel_length": voxel}), flush=True)
    mesh = volume.extract_triangle_mesh()
    vertices, faces, _ = compact_triangles(np.asarray(mesh.vertices).astype(np.float64), np.asarray(mesh.triangles).astype(np.int64))
    metadata = source_metadata(args, manifest, started)
    metadata.update(voxel_length=voxel, sdf_trunc=voxel * args.truncation_voxels,
        open3d_version=o3d.__version__, boundary_policy="TSDF zero crossing from observed weighted voxels; no post-extraction hole filling",
        source_frame_policy="all fusion frames contribute to common TSDF; no fabricated single-view per-face source")
    save_triangle_table(args.output, TriangleTable(vertices, faces, metadata, args.run_token + ":tsdf"))


def assess_mesh(args):
    import open3d as o3d
    started = time.perf_counter()
    manifest = load_depth_manifest(args.depth_manifest)
    table = load_triangle_table(args.mesh)
    validate_depth_source(table, manifest)
    mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(table.vertices.astype(np.float32)),
                                      o3d.core.Tensor(table.faces.astype(np.uint32)))
    scene = o3d.t.geometry.RaycastingScene(nthreads=args.cpu_threads)
    scene.add_triangles(mesh)
    rows = []
    for row in manifest["completed"]:
        frame = DepthFrame.load(row["file"])
        h, w = frame.depth.shape
        rr, cc = np.meshgrid(np.arange(0, h, args.diagnostic_stride), np.arange(0, w, args.diagnostic_stride), indexing="ij")
        origin = np.linalg.inv(frame.world_to_camera)[:3, 3]
        endpoints = backproject(frame, rr, cc, np.ones_like(rr))
        rays = np.concatenate([np.broadcast_to(origin, endpoints.shape), endpoints - origin], axis=-1).astype(np.float32)
        result = scene.cast_rays(o3d.core.Tensor(rays))
        depths = result["t_hit"].numpy().astype(np.float64)
        reference = frame.depth[rr, cc]
        observed = valid_depth(frame, args.alpha_threshold)[rr, cc]
        hit = np.isfinite(depths)
        both = hit & observed
        relative_error = (depths[both] - reference[both]) / reference[both]
        current = {"frame_index": frame.frame_index, "image_name": row["image_name"],
                   "diagnostic_rays": int(hit.size), "mesh_hits": int(hit.sum()),
                   "source_high_confidence_rays": int(observed.sum()), "paired_rays": int(both.sum()),
                   "observed_coverage": float(both.sum() / max(observed.sum(), 1)),
                   "relative_depth_error_quantiles": np.quantile(relative_error, [0, .01, .5, .99, 1]).tolist() if relative_error.size else [],
                   "depth_agreement_2percent": int((np.abs(relative_error) <= .02).sum()),
                   "mesh_in_front_more_than_2percent": int((relative_error < -.02).sum())}
        rows.append(current)
        print(json.dumps({"phase": "assess", "completed": len(rows), "total": len(manifest["completed"]), **current}), flush=True)
    total = {key: sum(row[key] for row in rows) for key in ("diagnostic_rays", "mesh_hits", "source_high_confidence_rays", "paired_rays", "depth_agreement_2percent", "mesh_in_front_more_than_2percent")}
    total["observed_coverage"] = total["paired_rays"] / max(total["source_high_confidence_rays"], 1)
    total["depth_agreement_fraction"] = total["depth_agreement_2percent"] / max(total["paired_rays"], 1)
    write_json(args.output, {"status": "measured_not_image_accepted", "mesh": str(args.mesh.resolve()),
        "mesh_token": table.mesh_token, "depth_manifest": str(args.depth_manifest.resolve()),
        "diagnostic_stride": args.diagnostic_stride, "scope": "all fusion views, sampled pixel diagnostics; never a continuous coverage certificate",
        "frames": rows, "totals": total, "elapsed_seconds": time.perf_counter() - started})


def filter_consistency(args):
    """Reject surface triangles contradicted by any observed fusion camera.

    This improves a candidate's cross-view depth consistency; it is not a
    continuous image-coverage or true-surface certificate.
    """
    started = time.perf_counter()
    manifest = load_depth_manifest(args.depth_manifest)
    table = load_triangle_table(args.mesh)
    validate_depth_source(table, manifest)
    support = np.zeros(len(table.vertices), dtype=np.int32)
    conflicts = np.zeros(len(table.vertices), dtype=np.int32)
    for position, row in enumerate(manifest["completed"]):
        frame = DepthFrame.load(row["file"])
        agrees, conflict = cross_view_support(table.vertices, frame,
            relative_tolerance=args.relative_tolerance, alpha_threshold=args.alpha_threshold)
        support += agrees
        conflicts += conflict
        print(json.dumps({"phase": "global_consistency", "completed": position + 1,
            "total": len(manifest["completed"]), "frame_index": frame.frame_index,
            "conflicted_vertices": int((conflicts > args.maximum_conflict_views).sum())}), flush=True)
    keep = ((support[table.faces] >= args.minimum_support_views).all(axis=1) &
            (conflicts[table.faces] <= args.maximum_conflict_views).all(axis=1))
    vertices, faces, nondegenerate = compact_triangles(table.vertices, table.faces[keep])
    metadata = dict(table.source_record)
    metadata.update(method=str(metadata["method"]) + "+global_consistency", status="candidate_not_adopted",
        run_token=args.run_token, input_mesh=str(args.mesh.resolve()), input_mesh_token=table.mesh_token,
        input_triangles=len(table.faces), retained_triangles=len(faces),
        global_consistency={"views": len(manifest["completed"]), "relative_tolerance": args.relative_tolerance,
            "alpha_threshold": args.alpha_threshold, "minimum_support_views": args.minimum_support_views,
            "maximum_conflict_views": args.maximum_conflict_views,
            "scope": "all triangle vertices in all source cameras, not continuous projected interior proof"},
        refinement_seconds=time.perf_counter() - started)
    with np.load(args.mesh, allow_pickle=False) as data:
        attributes = {name: data[name][keep][nondegenerate] for name in data.files
                      if name not in ("vertices", "faces", "triangle_ids")}
    attributes["global_support_min"] = support[table.faces[keep]].min(axis=1)[nondegenerate]
    attributes["global_conflict_max"] = conflicts[table.faces[keep]].max(axis=1)[nondegenerate]
    save_triangle_table(args.output, TriangleTable(vertices, faces, metadata, args.run_token), **attributes)


def simplify_mesh(args):
    import pymeshlab
    from importlib.metadata import version
    started = time.perf_counter()
    table = load_triangle_table(args.mesh)
    if args.target_faces <= 0 or len(table.vertices) > np.iinfo(np.int32).max:
        raise ValueError("Positive face target and int32-compatible PyMeshLab vertex table required")
    before, before_boundary = mesh_topology(table.vertices, table.faces)
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(vertex_matrix=table.vertices, face_matrix=table.faces.astype(np.int32)))
    settings = {"targetfacenum": args.target_faces, "preserveboundary": True,
                "preservetopology": True, "preservenormal": True,
                "optimalplacement": False, "autoclean": False}
    mesh_set.meshing_decimation_quadric_edge_collapse(**settings)
    result = mesh_set.current_mesh()
    vertices, faces, _ = compact_triangles(result.vertex_matrix().astype(np.float64),
                                          result.face_matrix().astype(np.int64))
    after, after_boundary = mesh_topology(vertices, faces)
    audit = {"status": "checking", "input_mesh": str(args.mesh.resolve()),
             "settings": settings, "topology_before": before, "topology_after": after,
             "input_faces": len(table.faces), "actual_faces": len(faces),
             "seconds": time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = args.output.with_suffix(".simplification_audit.json")
    for key in ("components", "euler", "nonmanifold_edges"):
        if before[key] != after[key]:
            audit.update(status="rejected", reason=f"Topology statistic {key} changed")
            write_json(audit_path, audit)
            raise ValueError(f"Simplification changed topology statistic {key}: {before[key]} -> {after[key]}")
    try:
        assert_boundary_preserved(table.vertices, before_boundary, vertices, after_boundary)
    except ValueError as exc:
        audit.update(status="rejected", reason=str(exc))
        write_json(audit_path, audit)
        raise
    audit.update(status="pass", boundary_segments_exactly_preserved=True)
    write_json(audit_path, audit)
    metadata = dict(table.source_record)
    metadata.update(method=str(metadata["method"]) + "+topology_preserving_qem", status="candidate_not_adopted",
        input_mesh=str(args.mesh.resolve()), input_mesh_token=table.mesh_token, run_token=args.run_token,
        simplification_settings=settings, pymeshlab_version=version("pymeshlab"),
        topology_before=before, topology_after=after, boundary_segments_exactly_preserved=True,
        simplification_seconds=time.perf_counter() - started,
        simplification_source_policy="new face-row IDs, original mesh and full fusion lineage retained; no invented per-face source frame",
        requested_faces=args.target_faces, actual_faces=len(faces),
        target_face_count_reached=len(faces) <= args.target_faces)
    save_triangle_table(args.output, TriangleTable(vertices, faces, metadata, args.run_token))
    print(json.dumps({"phase": "simplify", "requested_faces": args.target_faces, "actual_faces": len(faces),
                      "seconds": metadata["simplification_seconds"], "topology": after}), flush=True)


def weld_grid(args):
    started = time.perf_counter()
    table = load_triangle_table(args.mesh)
    with np.load(args.mesh, allow_pickle=False) as data:
        attributes = {name: data[name] for name in data.files if name not in ("vertices", "faces", "triangle_ids")}
    if "source_frame" not in attributes:
        raise ValueError("Source-view welding requires original per-face source_frame")
    vertices, faces = weld_same_source_vertices(table.vertices, table.faces, attributes["source_frame"])
    metadata = dict(table.source_record)
    metadata.update(method=str(metadata["method"]) + "+source_view_vertex_weld",
        status="candidate_not_adopted", run_token=args.run_token,
        input_mesh=str(args.mesh.resolve()), input_mesh_token=table.mesh_token,
        source_view_weld={"input_vertices": len(table.vertices), "output_vertices": len(vertices),
            "ordered_triangle_xyz_exactly_equal": True, "cross_view_merge": False,
            "seconds": time.perf_counter() - started})
    save_triangle_table(args.output, TriangleTable(vertices, faces, metadata, args.run_token), **attributes)


def filter_native_pixels(args):
    """Delete contradictory face interiors using every native source pixel."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from gdmgs.geometry.pixel_consistency import NativePixelCaster
    started = time.perf_counter()
    manifest = load_depth_manifest(args.depth_manifest)
    table = load_triangle_table(args.mesh)
    validate_depth_source(table, manifest)
    output = args.output
    report_path = output.with_suffix(".native_pixel_audit.json")
    if output.exists() or output.with_suffix(".json").exists() or report_path.exists():
        raise FileExistsError("Use a new explicit native-pixel candidate output/run label")
    output.parent.mkdir(parents=True, exist_ok=True)
    # All observations are loaded once; no original dataset/checkpoint is read.
    frames = [DepthFrame.load(row["file"]) for row in manifest["completed"]]
    active = np.arange(len(table.faces), dtype=np.int64)
    report = {"status": "running", "run_token": args.run_token,
              "input_mesh": str(args.mesh.resolve()), "input_mesh_token": table.mesh_token,
              "depth_manifest": str(args.depth_manifest.resolve()),
              "depth_run_token": manifest["run_token"], "fusion_frame_count": len(frames),
              "fusion_frame_indices": [frame.frame_index for frame in frames],
              "input_triangles": len(active), "alpha_threshold": args.alpha_threshold,
              "relative_tolerance": args.relative_tolerance, "pixel_stride": 1,
              "block_rows": args.block_rows, "cpu_threads": args.cpu_threads, "near": .01,
              "cpu_raycast_threads": args.cpu_threads, "intersection_mode": args.intersection_mode,
              "all_intersection_screen": "every pixel first cast; list all hits only where nearest depth may be below the ED limit, with 64 float32-epsilon query-only outward margin",
              "engine": "Open3D CPU RaycastingScene float32 triangles/rays, camera-z recomputed in float64",
              "rule": "delete any face with a sampled intersection contradicting a high-alpha ED pixel; union deletions after every full view set; repeat until a full zero-conflict pass",
              "ground_truth_used": False, "passes": []}
    write_json(report_path, report)
    pass_index = 0
    while True:
        pass_index += 1
        pass_started = time.perf_counter()
        caster = NativePixelCaster(table.vertices, table.faces[active], cpu_threads=args.cpu_threads)
        remove = np.zeros(len(active), dtype=bool)
        current = {"pass_index": pass_index, "input_triangles": len(active), "frames": []}
        report["passes"].append(current)
        for position, frame in enumerate(frames):
            frame_started = time.perf_counter()
            ids, counts = caster.conflicts(frame, relative_tolerance=args.relative_tolerance,
                alpha_threshold=args.alpha_threshold, block_rows=args.block_rows,
                all_intersections=args.intersection_mode == "all")
            remove[ids] = True
            row = {"fusion_index": position, "frame_index": frame.frame_index,
                   "seconds": time.perf_counter() - frame_started, **counts}
            current["frames"].append(row)
            write_json(report_path, report)
            print(json.dumps({"phase": "native_pixel_consistency", "pass_index": pass_index,
                "completed": position + 1, "total": len(frames), "union_conflicting_faces": int(remove.sum()), **row}), flush=True)
        removed = active[remove]
        retained = active[~remove]
        id_path = output.parent / f"native_pixel_pass_{pass_index:03d}_triangle_ids.npz"
        np.savez(id_path, removed_input_triangle_ids=removed, retained_input_triangle_ids=retained)
        current.update(completed=True, removed_triangles=len(removed), retained_triangles=len(retained),
                       triangle_id_record=str(id_path.resolve()), seconds=time.perf_counter() - pass_started)
        active = retained
        write_json(report_path, report)
        del caster
        if not len(removed):
            break
    vertices, faces, nondegenerate = compact_triangles(table.vertices, table.faces[active])
    if not nondegenerate.all():
        raise RuntimeError("Native-pixel filtering unexpectedly changed source triangle geometry")
    if not np.array_equal(vertices[faces], table.vertices[table.faces[active]]):
        raise RuntimeError("Native-pixel filtering altered surviving triangle coordinates")
    final_conflicts = sum(row["conflicting_pixels"] for row in report["passes"][-1]["frames"])
    if final_conflicts or len(report["passes"][-1]["frames"]) != len(frames):
        raise RuntimeError("A complete zero-conflict final pass is required")
    with np.load(args.mesh, allow_pickle=False) as data:
        attributes = {name: data[name][active] for name in data.files if name not in ("vertices", "faces", "triangle_ids")}
    attributes["source_input_triangle_id"] = active
    metadata = dict(table.source_record)
    metadata.update(method=str(metadata["method"]) + "+native_pixel_freespace_consistency",
        run_token=args.run_token, status="candidate_not_adopted", input_mesh=str(args.mesh.resolve()),
        input_mesh_token=table.mesh_token, native_pixel_audit=str(report_path.resolve()),
        native_pixel_filter={"complete_fusion_views": len(frames), "passes": pass_index,
            "pixel_stride": 1, "relative_tolerance": args.relative_tolerance,
            "intersection_mode": args.intersection_mode, "cpu_raycast_threads": args.cpu_threads,
            "alpha_threshold": args.alpha_threshold, "final_pass_conflicting_pixels": final_conflicts,
            "surviving_triangle_xyz_exactly_equal": True,
            "input_triangles": len(table.faces), "retained_triangles": len(faces),
            "seconds": time.perf_counter() - started,
            "scope": "all native source pixel centers; ED consistency is not GT/image-quality acceptance"})
    save_triangle_table(output, TriangleTable(vertices, faces, metadata, args.run_token), **attributes)
    report.update(status="complete", retained_triangles=len(faces), passes_completed=pass_index,
                  elapsed_seconds=time.perf_counter() - started, mesh_file=str(output.resolve()))
    write_json(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--model-path", type=Path, required=True)
    export.add_argument("--iteration", type=int, default=40000)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--run-token", required=True)
    export_set = commands.add_parser("export-scenes")
    export_set.add_argument("--scene-inventory", type=Path, required=True)
    export_set.add_argument("--scenes", required=True)
    export_set.add_argument("--iteration", type=int, default=40000)
    export_set.add_argument("--output", type=Path, required=True)
    export_set.add_argument("--run-token", required=True)
    build = commands.add_parser("build")
    build.add_argument("--depth-manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--run-token", required=True)
    build.add_argument("--method", choices=("grid", "tsdf"), required=True)
    build.add_argument("--alpha-threshold", type=float, default=.995)
    build.add_argument("--grid-stride", type=int, default=16)
    build.add_argument("--relative-depth-jump", type=float, default=.02)
    build.add_argument("--consistency-views", type=int, default=4)
    build.add_argument("--minimum-support-views", type=int, default=1)
    build.add_argument("--maximum-conflict-views", type=int, default=0)
    build.add_argument("--consistency-tolerance", type=float, default=.02)
    build.add_argument("--voxel-length", type=float)
    build.add_argument("--voxel-relative-depth", type=float, default=.005)
    build.add_argument("--truncation-voxels", type=float, default=4.)
    build.add_argument("--integration-stride", type=int, default=4)
    assess = commands.add_parser("assess")
    assess.add_argument("--depth-manifest", type=Path, required=True)
    assess.add_argument("--mesh", type=Path, required=True)
    assess.add_argument("--output", type=Path, required=True)
    assess.add_argument("--alpha-threshold", type=float, default=.995)
    assess.add_argument("--diagnostic-stride", type=int, default=8)
    assess.add_argument("--cpu-threads", type=int, default=8)
    refine = commands.add_parser("filter-consistency")
    refine.add_argument("--depth-manifest", type=Path, required=True)
    refine.add_argument("--mesh", type=Path, required=True)
    refine.add_argument("--output", type=Path, required=True)
    refine.add_argument("--run-token", required=True)
    refine.add_argument("--alpha-threshold", type=float, default=.995)
    refine.add_argument("--relative-tolerance", type=float, default=.02)
    refine.add_argument("--minimum-support-views", type=int, default=2)
    refine.add_argument("--maximum-conflict-views", type=int, default=0)
    simplify = commands.add_parser("simplify")
    simplify.add_argument("--mesh", type=Path, required=True)
    simplify.add_argument("--output", type=Path, required=True)
    simplify.add_argument("--run-token", required=True)
    simplify.add_argument("--target-faces", type=int, default=100000)
    weld = commands.add_parser("weld-source-grid")
    weld.add_argument("--mesh", type=Path, required=True)
    weld.add_argument("--output", type=Path, required=True)
    weld.add_argument("--run-token", required=True)
    pixels = commands.add_parser("filter-native-pixels")
    pixels.add_argument("--mesh", type=Path, required=True)
    pixels.add_argument("--depth-manifest", type=Path, required=True)
    pixels.add_argument("--output", type=Path, required=True)
    pixels.add_argument("--run-token", required=True)
    pixels.add_argument("--alpha-threshold", type=float, default=.995)
    pixels.add_argument("--relative-tolerance", type=float, default=.02)
    pixels.add_argument("--block-rows", type=int, default=128)
    pixels.add_argument("--cpu-threads", type=int, default=8)
    pixels.add_argument("--intersection-mode", choices=("nearest", "all"), default="all")
    args = parser.parse_args()
    if args.command == "export":
        export_depth(args)
    elif args.command == "export-scenes":
        export_scene_set(args)
    elif args.command == "build":
        manifest = load_depth_manifest(args.depth_manifest)
        (build_grid if args.method == "grid" else build_tsdf)(args, manifest)
    elif args.command == "filter-consistency":
        filter_consistency(args)
    elif args.command == "simplify":
        simplify_mesh(args)
    elif args.command == "weld-source-grid":
        weld_grid(args)
    elif args.command == "filter-native-pixels":
        filter_native_pixels(args)
    else:
        assess_mesh(args)


if __name__ == "__main__":
    main()

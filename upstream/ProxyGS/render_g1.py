"""Run Step 4 G1-Ref and G1 on one complete frozen ProxyGS scene."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torchvision

ARTIFACT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step4_cpu_mesh_index_g1_v2_20260914")
PROTOCOL_ID = "proxygs-step4-g1-v2"
os.environ.setdefault("GDMGS_NATIVE_DIR", str(ARTIFACT / "native"))

from gdmgs.mesh_index import MeshIndex, MeshQueryResult
from online_proxy_depth import OnlineProxyDepthRasterizer
from render_gdmgs_backend import (
    BACKEND_ID,
    _environment_record,
    _file_identity,
    _frozen_camera_names,
    _load_cfg,
    _new_model,
    _ordered_views,
    _render_one,
)
from step4_runtime import (
    DEPTH_MARGIN,
    DEPTH_ORACLE_ATOL,
    DEPTH_ORACLE_MAX_COVERAGE_MISMATCH_FRACTION,
    DEPTH_ORACLE_MAX_OUTLIER_FRACTION,
    DEPTH_ORACLE_MEAN_LIMIT,
    DEPTH_ORACLE_NORMALIZED_MEAN_LIMIT,
    DEPTH_ORACLE_NORMALIZED_P99_LIMIT,
    DEPTH_ORACLE_P99_LIMIT,
    DEPTH_ORACLE_RTOL,
    INDEXED_DEPTH_ATOL,
    INDEXED_DEPTH_RTOL,
    atomic_json,
    camera_domain_from_view,
    camera_record,
    dense_anchor_filter_cpu,
    dense_anchor_filter_cuda_reference,
    file_identity,
    mean,
    parity_stats,
)


def ids_h2d(values: np.ndarray, device: torch.device) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = torch.tensor(values, dtype=torch.long, device=device)
    torch.cuda.synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


def save_id_payload(path: Path, candidate_ids: np.ndarray, selected_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, candidate_ids=candidate_ids, selected_ids=selected_ids)
    os.replace(temporary, path)


def render_scene(args: argparse.Namespace) -> Dict[str, Any]:
    if (args.formal or args.qualification) and args.max_views is not None:
        raise ValueError("formal and qualification Step 4 cannot reduce the camera denominator")
    if args.qualification and args.scene != "amsterdam":
        raise ValueError("the preregistered complete-scene qualification is Amsterdam")
    if args.formal and args.qualification:
        raise ValueError("formal and qualification modes are mutually exclusive")
    if not args.formal and not args.qualification and args.max_views is None:
        raise ValueError("development mode requires an explicit --max-views")
    if args.iteration != 40000 or args.width != 1600 or args.height != 900:
        raise ValueError("Step 4 freezes iteration=40000 and depth/render size=1600x900")
    if not np.isclose(args.margin, 0.3, rtol=0.0, atol=1.0e-12):
        raise ValueError("Step 4 freezes the depth safety margin at +0.3")

    output_dir = (args.output_root / args.scene / args.run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(map(str, sys.argv)) + "\n")
    atomic_json(output_dir / "status.json", {"state": "starting", "scene": args.scene})

    model_path = args.model_path.resolve()
    cfg = _load_cfg(model_path)
    if Path(cfg.source_path).resolve() != args.source_path.resolve():
        raise ValueError(f"source path mismatch: cfg={cfg.source_path} cli={args.source_path}")
    cfg.source_path = str(args.source_path.resolve())
    cfg.data_device = "cpu"
    model = _new_model(cfg)
    from scene import Scene

    scene = Scene(cfg, model, load_iteration=args.iteration, shuffle=False, resolution_scales=cfg.resolution_scales)
    model.eval()
    frozen_names = _frozen_camera_names(model_path)
    views = _ordered_views(scene, frozen_names)
    if len(views) != args.expected_views:
        raise ValueError(f"expected {args.expected_views} views, found {len(views)}")
    if args.max_views is not None:
        views = views[: args.max_views]
    if any((int(view.image_width), int(view.image_height)) != (args.width, args.height) for view in views):
        raise ValueError("every Step 4 view must be 1600x900")

    mesh_index = MeshIndex.load(args.mesh_index.resolve(), mesh_token=args.mesh_token)
    if mesh_index.build_settings != {"method": "binned_sah", "leaf_size": 8}:
        raise ValueError("formal Step 4 requires binned_sah leaf_size=8")
    rasterizer = OnlineProxyDepthRasterizer(mesh_index, device="cuda:0")
    all_triangle_ids = np.arange(len(mesh_index.triangles), dtype=np.int64)
    anchor_positions_cpu = np.ascontiguousarray(model.get_anchor.detach().cpu().numpy(), dtype=np.float32)
    background_values = [1.0, 1.0, 1.0] if cfg.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")
    pipeline = Namespace(compute_cov3D_python=args.compute_cov3D_python, debug=args.debug)
    lpips_model = None
    if args.lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="vgg").to(background.device).eval()

    inputs = {
        "cfg_args": _file_identity(model_path / "cfg_args"),
        "cameras_json": _file_identity(model_path / "cameras.json"),
        "point_cloud": _file_identity(model_path / "point_cloud" / "iteration_40000" / "point_cloud.ply"),
        "opacity_mlp": _file_identity(model_path / "point_cloud" / "iteration_40000" / "opacity_mlp.pt"),
        "cov_mlp": _file_identity(model_path / "point_cloud" / "iteration_40000" / "cov_mlp.pt"),
        "color_mlp": _file_identity(model_path / "point_cloud" / "iteration_40000" / "color_mlp.pt"),
        "mesh_index": file_identity(args.mesh_index.resolve()),
        "mesh_input": file_identity(args.mesh_input.resolve()),
        "renderer_settings": file_identity(args.renderer_settings.resolve()),
        "explicit_selection_contract": file_identity(args.explicit_selection_contract.resolve()),
    }
    contract = {
        "schema": "proxygs_step4_g1_scene_contract_v2",
        "protocol_id": PROTOCOL_ID,
        "state": "frozen_before_run",
        "scene": args.scene,
        "formal": args.formal,
        "qualification": args.qualification,
        "backend": BACKEND_ID,
        "iteration": args.iteration,
        "camera_count": len(views),
        "frozen_camera_count": len(frozen_names),
        "camera_names": [view.image_name for view in views],
        "image_size": [args.width, args.height],
        "mesh_token": args.mesh_token,
        "mesh_index_settings": mesh_index.build_settings,
        "depth_margin": args.margin,
        "depth_oracle_policy": {
            "role": "diagnostic_only",
            "hard_gate": False,
            "max_coverage_mismatch_fraction": DEPTH_ORACLE_MAX_COVERAGE_MISMATCH_FRACTION,
            "atol": DEPTH_ORACLE_ATOL,
            "rtol": DEPTH_ORACLE_RTOL,
            "normalized_mean_absolute_error_limit": DEPTH_ORACLE_NORMALIZED_MEAN_LIMIT,
            "normalized_p99_absolute_error_limit": DEPTH_ORACLE_NORMALIZED_P99_LIMIT,
            "local_edge_tie_radius_pixels": 1,
            "local_edge_explanation_role": "diagnostic_only",
            "selected_ids_role": "diagnostic_only",
            "rationale": "scale-normalized p99 and bounded coverage drift; formal selection reference is full-mesh online depth",
        },
        "indexed_depth_policy": {"coverage_exact": True, "atol": INDEXED_DEPTH_ATOL, "rtol": INDEXED_DEPTH_RTOL},
        "selected_id_policy": "exact_original_order",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "environment": _environment_record(),
        "inputs": inputs,
    }
    atomic_json(output_dir / "run_contract.json", contract)

    records: List[Dict[str, Any]] = []
    camera_records = []
    ref_id_manifest: Dict[str, Any] = {}
    g1_id_manifest: Dict[str, Any] = {}
    ref_render_dir = output_dir / "g1_ref" / "renders"
    g1_render_dir = output_dir / "g1" / "renders"
    if args.save_images:
        ref_render_dir.mkdir(parents=True, exist_ok=True)
        g1_render_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "status.json", {"state": "running", "scene": args.scene, "expected_views": len(views)})

    for index, view in enumerate(views):
        domain = camera_domain_from_view(view)
        camera_records.append(camera_record(view, domain, index))

        brute = mesh_index.query(domain, backend="brute_force")
        indexed = mesh_index.query(domain, backend="bvh")
        query_ids_equal = np.array_equal(brute.triangle_ids, indexed.triangle_ids)
        if not query_ids_equal:
            raise RuntimeError(f"{view.image_name}: BVH and brute-force triangle IDs differ")

        full_query = MeshQueryResult(
            triangle_ids=all_triangle_ids,
            mesh_token=args.mesh_token,
            camera_domain=domain,
            counters={"backend": "full_mesh", "mesh_triangles": len(all_triangle_ids)},
            elapsed_ms=0.0,
            complete=True,
        )
        del full_query  # The online rasterizer consumes the same validated sorted IDs directly.
        full_depth = rasterizer.render(all_triangle_ids, domain, (args.width, args.height))
        oracle = np.load(args.depth_oracle_root / args.scene / f"{view.image_name}.npy", allow_pickle=False)
        oracle = np.ascontiguousarray(oracle, dtype=np.float32)
        oracle_parity = parity_stats(
            full_depth.depth_cpu,
            oracle,
            atol=DEPTH_ORACLE_ATOL,
            rtol=DEPTH_ORACLE_RTOL,
            normalized_mean_limit=DEPTH_ORACLE_NORMALIZED_MEAN_LIMIT,
            normalized_p99_limit=DEPTH_ORACLE_NORMALIZED_P99_LIMIT,
            allow_local_edge_ties=True,
            max_coverage_mismatch_fraction=DEPTH_ORACLE_MAX_COVERAGE_MISMATCH_FRACTION,
        )
        oracle_parity["role"] = "diagnostic_only"
        oracle_parity["hard_gate"] = False

        indexed_depth = rasterizer.render(indexed.triangle_ids, domain, (args.width, args.height))
        indexed_parity = parity_stats(
            indexed_depth.depth_cpu,
            full_depth.depth_cpu,
            atol=INDEXED_DEPTH_ATOL,
            rtol=INDEXED_DEPTH_RTOL,
        )
        if not indexed_parity["pass"]:
            raise RuntimeError(f"{view.image_name}: indexed depth differs from full online depth: {indexed_parity}")

        torch.cuda.synchronize()
        candidate_start = time.perf_counter()
        model.set_anchor_mask(view.camera_center, args.iteration, view.resolution_scale)
        candidate_ids_gpu = torch.nonzero(model._anchor_mask, as_tuple=False).flatten()
        candidate_ids = candidate_ids_gpu.detach().cpu().numpy().copy()
        torch.cuda.synchronize()
        candidate_ms = (time.perf_counter() - candidate_start) * 1000.0

        world_view = np.ascontiguousarray(view.world_view_transform.detach().cpu().numpy(), dtype=np.float32)
        full_projection = np.ascontiguousarray(view.full_proj_transform.detach().cpu().numpy(), dtype=np.float32)
        selected_ref, dense_ref = dense_anchor_filter_cpu(
            candidate_ids,
            anchor_positions_cpu,
            world_view,
            full_projection,
            full_depth.depth_cpu,
            margin=args.margin,
        )
        selected_g1, dense_g1 = dense_anchor_filter_cpu(
            candidate_ids,
            anchor_positions_cpu,
            world_view,
            full_projection,
            indexed_depth.depth_cpu,
            margin=args.margin,
        )
        selected_oracle, dense_oracle = dense_anchor_filter_cpu(
            candidate_ids,
            anchor_positions_cpu,
            world_view,
            full_projection,
            oracle,
            margin=args.margin,
        )
        cuda_selected, cuda_dense_ms = dense_anchor_filter_cuda_reference(
            candidate_ids_gpu,
            model.get_anchor,
            view.world_view_transform,
            view.full_proj_transform,
            full_depth.depth_gpu,
            margin=args.margin,
        )
        cpu_cuda_equal = np.array_equal(selected_ref, cuda_selected)
        ref_g1_equal = np.array_equal(selected_ref, selected_g1)
        oracle_ref_equal = np.array_equal(selected_oracle, selected_ref)
        if not cpu_cuda_equal:
            cpu_only = np.setdiff1d(selected_ref, cuda_selected, assume_unique=True)
            cuda_only = np.setdiff1d(cuda_selected, selected_ref, assume_unique=True)
            diagnostic = {
                "camera": view.image_name,
                "candidate_count": int(len(candidate_ids)),
                "cpu_selected_count": int(len(selected_ref)),
                "cuda_selected_count": int(len(cuda_selected)),
                "cpu_only_count": int(len(cpu_only)),
                "cuda_only_count": int(len(cuda_only)),
                "cpu_only_first_100": cpu_only[:100].tolist(),
                "cuda_only_first_100": cuda_only[:100].tolist(),
            }
            atomic_json(output_dir / "diagnostics" / f"{view.image_name}_cpu_cuda_ids.json", diagnostic)
            raise RuntimeError(
                f"{view.image_name}: CPU dense IDs differ from CUDA reference: {diagnostic}"
            )
        if not ref_g1_equal:
            raise RuntimeError(f"{view.image_name}: G1-Ref and G1 selected IDs differ")
        oracle_cpu_only = np.setdiff1d(selected_ref, selected_oracle, assume_unique=True)
        oracle_saved_only = np.setdiff1d(selected_oracle, selected_ref, assume_unique=True)

        payload = output_dir / "id_payload" / f"{view.image_name}.npz"
        save_id_payload(payload, candidate_ids, selected_ref)
        payload_record = {
            "path": str(payload),
            "candidate_key": "candidate_ids",
            "selected_key": "selected_ids",
            "candidate_count": int(len(candidate_ids)),
            "selected_count": int(len(selected_ref)),
            "dtype": "int64",
            "order": "original_frozen_anchor_row_order",
        }
        ref_id_manifest[view.image_name] = payload_record
        g1_id_manifest[view.image_name] = payload_record

        selected_ref_gpu, ref_ids_h2d_ms = ids_h2d(selected_ref, model.get_anchor.device)
        selected_g1_gpu, g1_ids_h2d_ms = ids_h2d(selected_g1, model.get_anchor.device)
        ref_record, ref_image, _ = _render_one(
            view=view,
            model=model,
            pipeline=pipeline,
            background=background,
            anchor_ids=selected_ref_gpu,
            render_mode="RGB",
            warmup=args.warmup if index == 0 else 0,
            repeat=args.repeat,
            native_diagnostic=False,
            lpips_model=lpips_model,
            record_explicit_ids=False,
        )
        g1_record, g1_image, _ = _render_one(
            view=view,
            model=model,
            pipeline=pipeline,
            background=background,
            anchor_ids=selected_g1_gpu,
            render_mode="RGB",
            warmup=args.warmup if index == 0 else 0,
            repeat=args.repeat,
            native_diagnostic=False,
            lpips_model=lpips_model,
            record_explicit_ids=False,
        )
        ref_record["selection_contract"] = {"mode": "explicit", **payload_record}
        g1_record["selection_contract"] = {"mode": "explicit", **payload_record}

        ref_write_ms = g1_write_ms = 0.0
        if args.save_images:
            write_start = time.perf_counter()
            torchvision.utils.save_image(ref_image, ref_render_dir / f"{view.image_name}.png")
            ref_write_ms = (time.perf_counter() - write_start) * 1000.0
            write_start = time.perf_counter()
            torchvision.utils.save_image(g1_image, g1_render_dir / f"{view.image_name}.png")
            g1_write_ms = (time.perf_counter() - write_start) * 1000.0
        if args.save_depth:
            depth_dir = output_dir / "online_depth" / view.image_name
            depth_dir.mkdir(parents=True, exist_ok=True)
            np.save(depth_dir / "g1_ref_full.npy", full_depth.depth_cpu, allow_pickle=False)
            np.save(depth_dir / "g1_indexed.npy", indexed_depth.depth_cpu, allow_pickle=False)

        g1_ref_total_ms = (
            candidate_ms
            + full_depth.timings["depth_total_ms"]
            + dense_ref["elapsed_ms"]
            + ref_ids_h2d_ms
            + ref_record["decode_seconds"] * 1000.0
            + ref_record["render_seconds_mean"] * 1000.0
            + ref_write_ms
        )
        g1_total_ms = (
            candidate_ms
            + indexed.elapsed_ms
            + indexed_depth.timings["depth_total_ms"]
            + dense_g1["elapsed_ms"]
            + g1_ids_h2d_ms
            + g1_record["decode_seconds"] * 1000.0
            + g1_record["render_seconds_mean"] * 1000.0
            + g1_write_ms
        )
        record = {
            "index": index,
            "camera": view.image_name,
            "total_triangles": int(len(all_triangle_ids)),
            "bvh_returned_triangles": int(len(indexed.triangle_ids)),
            "visited_nodes": int(indexed.counters["visited_nodes"]),
            "tested_triangles": int(indexed.counters["tested_triangles"]),
            "fov_lod_candidate_anchors": int(len(candidate_ids)),
            "g1_selected_anchors": int(len(selected_g1)),
            "mesh_query_cpu_ms": float(indexed.elapsed_ms),
            "brute_query_cpu_ms": float(brute.elapsed_ms),
            "fov_lod_candidate_ms": candidate_ms,
            "g1_ref_depth": {"timings": full_depth.timings, "counters": full_depth.counters},
            "g1_depth": {"timings": indexed_depth.timings, "counters": indexed_depth.counters},
            "g1_ref_dense": dense_ref,
            "g1_dense": dense_g1,
            "step2_oracle_dense": dense_oracle,
            "step2_oracle_selection_diagnostic": {
                "equal": oracle_ref_equal,
                "online_only_count": int(len(oracle_cpu_only)),
                "saved_oracle_only_count": int(len(oracle_saved_only)),
                "online_only_first_100": oracle_cpu_only[:100].tolist(),
                "saved_oracle_only_first_100": oracle_saved_only[:100].tolist(),
            },
            "cuda_dense_reference_ms": cuda_dense_ms,
            "g1_ref_selected_ids_h2d_ms": ref_ids_h2d_ms,
            "g1_selected_ids_h2d_ms": g1_ids_h2d_ms,
            "g1_ref_render": ref_record,
            "g1_render": g1_record,
            "g1_ref_image_write_ms": ref_write_ms,
            "g1_image_write_ms": g1_write_ms,
            "g1_ref_frame_total_ms": g1_ref_total_ms,
            "frame_total_ms": g1_total_ms,
            "parity": {
                "brute_vs_bvh_triangle_ids": query_ids_equal,
                "full_online_vs_step2_oracle": oracle_parity,
                "indexed_vs_full_online_depth": indexed_parity,
                "cuda_vs_cpu_dense_ids": cpu_cuda_equal,
                "g1_ref_vs_g1_selected_ids": ref_g1_equal,
            },
            "id_payload": payload_record,
        }
        records.append(record)
        atomic_json(output_dir / "per_view.json", records)
        atomic_json(output_dir / "g1_ref_anchor_ids.json", ref_id_manifest)
        atomic_json(output_dir / "g1_anchor_ids.json", g1_id_manifest)
        atomic_json(
            output_dir / "status.json",
            {
                "state": "running",
                "scene": args.scene,
                "completed_views": len(records),
                "expected_views": len(views),
                "last_camera": view.image_name,
            },
        )
        del ref_image, g1_image, selected_ref_gpu, selected_g1_gpu, full_depth, indexed_depth
        torch.cuda.empty_cache()

    summary = {
        "schema": "proxygs_step4_g1_scene_summary_v2",
        "protocol_id": PROTOCOL_ID,
        "state": "complete",
        "backend": BACKEND_ID,
        "scene": args.scene,
        "formal": args.formal,
        "qualification": args.qualification,
        "view_count": len(records),
        "frozen_view_count": len(frozen_names),
        "correctness_failures": [],
        "g1_ref_metrics": {
            name: mean(record["g1_ref_render"]["metrics"][name] for record in records if record["g1_ref_render"]["metrics"][name] is not None)
            for name in ("psnr", "ssim", "lpips")
        },
        "g1_metrics": {
            name: mean(record["g1_render"]["metrics"][name] for record in records if record["g1_render"]["metrics"][name] is not None)
            for name in ("psnr", "ssim", "lpips")
        },
        "timing_means_ms": {
            key: mean(record[key] for record in records)
            for key in ("mesh_query_cpu_ms", "brute_query_cpu_ms", "fov_lod_candidate_ms", "g1_ref_frame_total_ms", "frame_total_ms")
        },
        "parity": {
            "brute_vs_bvh_triangle_ids_all": all(record["parity"]["brute_vs_bvh_triangle_ids"] for record in records),
            "indexed_vs_full_online_depth_all": all(record["parity"]["indexed_vs_full_online_depth"]["pass"] for record in records),
            "cuda_vs_cpu_dense_ids_all": all(record["parity"]["cuda_vs_cpu_dense_ids"] for record in records),
            "g1_ref_vs_g1_selected_ids_all": all(record["parity"]["g1_ref_vs_g1_selected_ids"] for record in records),
        },
        "step2_oracle_selection_diagnostic": {
            "mismatch_view_count": sum(not record["step2_oracle_selection_diagnostic"]["equal"] for record in records),
            "online_only_count": sum(record["step2_oracle_selection_diagnostic"]["online_only_count"] for record in records),
            "saved_oracle_only_count": sum(record["step2_oracle_selection_diagnostic"]["saved_oracle_only_count"] for record in records),
        },
        "step2_oracle_depth_diagnostic": {
            "role": "diagnostic_only",
            "hard_gate": False,
            "policy_exceeded_view_count": sum(not record["parity"]["full_online_vs_step2_oracle"]["pass"] for record in records),
            "coverage_mismatch_view_count": sum(not record["parity"]["full_online_vs_step2_oracle"]["coverage_match"] for record in records),
            "worst_normalized_p99": max(record["parity"]["full_online_vs_step2_oracle"]["normalized_p99_absolute_error"] for record in records),
            "worst_coverage_mismatch_fraction": max(record["parity"]["full_online_vs_step2_oracle"]["coverage_mismatch_fraction"] for record in records),
        },
        "input_identities_after": {name: file_identity(Path(item["path"])) for name, item in inputs.items()},
    }
    if summary["input_identities_after"] != inputs:
        raise RuntimeError("a frozen Step 1/2/3 input changed during Step 4")
    if not all(summary["parity"].values()):
        raise RuntimeError(f"scene parity aggregate failed: {summary['parity']}")
    atomic_json(output_dir / "camera_domain_records.json", camera_records)
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "status.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene", required=True)
    result.add_argument("--source-path", type=Path, required=True)
    result.add_argument("--model-path", type=Path, required=True)
    result.add_argument("--mesh-input", type=Path, required=True)
    result.add_argument("--mesh-index", type=Path, required=True)
    result.add_argument("--mesh-token", required=True)
    result.add_argument("--depth-oracle-root", type=Path, required=True)
    result.add_argument("--renderer-settings", type=Path, required=True)
    result.add_argument("--explicit-selection-contract", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--expected-views", type=int, required=True)
    result.add_argument("--iteration", type=int, default=40000)
    result.add_argument("--width", type=int, default=1600)
    result.add_argument("--height", type=int, default=900)
    result.add_argument("--margin", type=float, default=0.3)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--repeat", type=int, default=1)
    result.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--save-depth", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--formal", action="store_true")
    result.add_argument("--qualification", action="store_true")
    result.add_argument("--max-views", type=int)
    from arguments import PipelineParams

    PipelineParams(result)
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    try:
        report = render_scene(parsed)
    except Exception as error:
        failure = parsed.output_root / parsed.scene / parsed.run_id
        atomic_json(failure / "status.json", {"state": "failed", "scene": parsed.scene, "error": repr(error)})
        raise
    print(json.dumps(report, indent=2, sort_keys=True))

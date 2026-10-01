"""Replay one frozen Step 6 window and complete the G1/G2 chain matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torchvision

ARTIFACT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step6_high_overlap_cpu_index_20260915")
STEP5 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step5_cpu_anchor_index_g2_20260915")
STEP4 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step4_cpu_mesh_index_g1_v2_20260914")
PROTOCOL_ID = "proxygs-step6-j3-window-v2"
os.environ.setdefault("GDMGS_NATIVE_DIR", str(ARTIFACT / "native"))

from anchor_query_runtime import query_record
from gdmgs.anchor_index import AnchorPointIndex
from gdmgs.mesh_index import MeshIndex
from online_proxy_depth import OnlineProxyDepthRasterizer
from render_g1 import ids_h2d
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
    atomic_json,
    camera_domain_from_view,
    camera_record,
    file_identity,
    dense_anchor_filter_cpu,
)


def mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def percentile(values: Iterable[float], q: float) -> float | None:
    materialized = list(values)
    return float(np.percentile(materialized, q)) if materialized else None


def save_payload(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def pack_id_bitmap(ids: np.ndarray, universe_count: int) -> np.ndarray:
    if ids.dtype != np.int64 or ids.ndim != 1:
        raise TypeError("ID bitmap input must be rank-one int64")
    if ids.size and (
        ids[0] < 0
        or ids[-1] >= universe_count
        or np.any(ids[1:] <= ids[:-1])
    ):
        raise ValueError("ID bitmap input must be sorted, unique, and in range")
    bitmap = np.zeros(universe_count, dtype=np.uint8)
    bitmap[ids] = 1
    return np.packbits(bitmap, bitorder="little")


def unpack_id_bitmap(bitmap: np.ndarray, universe_count: int) -> np.ndarray:
    if bitmap.dtype != np.uint8 or bitmap.ndim != 1:
        raise TypeError("ID bitmap must be rank-one uint8")
    return np.flatnonzero(
        np.unpackbits(bitmap, count=universe_count, bitorder="little")
    ).astype(np.int64, copy=False)


def load_step5_ids(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "anchor_universe_count",
            "triangle_universe_count",
            "candidate_bitmap",
            "selected_bitmap",
            "triangle_bitmap",
        }
        if set(data.files) != required:
            raise ValueError("Step 5 ID payload schema drifted")
        anchor_count = int(data["anchor_universe_count"].item())
        triangle_count = int(data["triangle_universe_count"].item())
        candidates = unpack_id_bitmap(data["candidate_bitmap"], anchor_count)
        selected = unpack_id_bitmap(data["selected_bitmap"], anchor_count)
        triangles = unpack_id_bitmap(data["triangle_bitmap"], triangle_count)
    return candidates, selected, triangles, anchor_count, triangle_count


def render_selected(
    *,
    view: Any,
    model: Any,
    pipeline: Any,
    background: torch.Tensor,
    ids: np.ndarray,
    warmup: int,
    repeat: int,
    lpips_model: Any,
) -> tuple[dict[str, Any], torch.Tensor, float]:
    ids_gpu, h2d_ms = ids_h2d(ids, model.get_anchor.device)
    record, image, _ = _render_one(
        view=view,
        model=model,
        pipeline=pipeline,
        background=background,
        anchor_ids=ids_gpu,
        render_mode="RGB",
        warmup=warmup,
        repeat=repeat,
        native_diagnostic=False,
        lpips_model=lpips_model,
        record_explicit_ids=False,
    )
    record["selection_contract"] = {
        "mode": "explicit",
        "selected_count": int(len(ids)),
        "id_dtype": "int64",
        "order": "original_final_ply_row_order",
        "decoded_row_count": record["decoded_row_count"],
    }
    return record, image, h2d_ms


def tensor_contract_without_object_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Compare decoded tensor values/contracts without allocator-specific object IDs."""
    return {
        tensor: {key: value for key, value in fields.items() if key != "object_id"}
        for tensor, fields in record.items()
    }


def render_scene(args: argparse.Namespace) -> dict[str, Any]:
    if not args.formal or args.qualification or args.max_views is not None:
        raise ValueError("Step 6 window replay requires formal mode and the frozen 32-frame window")
    if args.iteration != 40000 or args.width != 1600 or args.height != 900:
        raise ValueError("Step 6 freezes iteration=40000 and image size=1600x900")
    if not np.isclose(args.margin, float(DEPTH_MARGIN), rtol=0.0, atol=1.0e-12):
        raise ValueError("Step 6 freezes the pointwise depth margin at float32 +0.3")

    output_dir = (args.output_root / args.scene / args.run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(map(str, sys.argv)) + "\n")
    atomic_json(output_dir / "status.json", {"state": "starting", "scene": args.scene})

    model_path = args.model_path.resolve()
    cfg = _load_cfg(model_path)
    if Path(cfg.source_path).resolve() != args.source_path.resolve():
        raise ValueError("model cfg source path differs from the frozen CLI path")
    cfg.source_path = str(args.source_path.resolve())
    cfg.data_device = "cpu"
    model = _new_model(cfg)
    from scene import Scene

    scene = Scene(
        cfg,
        model,
        load_iteration=args.iteration,
        shuffle=False,
        resolution_scales=cfg.resolution_scales,
    )
    model.eval()
    frozen_names = _frozen_camera_names(model_path)
    views = _ordered_views(scene, frozen_names)
    if len(views) != args.expected_views:
        raise ValueError(f"expected {args.expected_views} views, found {len(views)}")
    selection = json.loads(args.selection.read_text())
    if not (
        selection.get("status") == "frozen"
        and selection.get("selection_performed_once") is True
        and selection.get("window_count") == 8
        and selection.get("frame_count") == 256
        and selection.get("transition_count") == 248
    ):
        raise ValueError("Step 6 speed-selected windows are not frozen")
    matches = [record for record in selection["windows"] if record["scene"] == args.scene]
    if len(matches) != 1:
        raise ValueError("scene does not have exactly one frozen window")
    window = matches[0]
    start = int(window["start_index"])
    stop = int(window["end_index_inclusive"]) + 1
    views = views[start:stop]
    if len(views) != 32 or [view.image_name for view in views] != window["camera_ids"]:
        raise ValueError("selected camera IDs do not match the frozen camera window")
    if any((int(view.image_width), int(view.image_height)) != (args.width, args.height) for view in views):
        raise ValueError("every Step 6 camera must be 1600x900")
    mesh_window_profile = json.loads(args.mesh_window_profile.read_text())
    if not (
        mesh_window_profile.get("status") == "frozen_from_qualification"
        and mesh_window_profile.get("selection_changed_windows") is False
        and mesh_window_profile.get("window_count") == 8
        and mesh_window_profile.get("frame_count") == 256
        and mesh_window_profile.get("minimum_index_speedup_to_retain") == 1.2
    ):
        raise ValueError("Mesh window profile is not the conservative frozen qualification result")
    scene_profile = mesh_window_profile.get("scene_profiles", {}).get(args.scene)
    if not scene_profile:
        raise ValueError("Mesh window profile has no entry for this scene")
    window_backend = scene_profile.get("backend")
    window_workers = scene_profile.get("workers")
    if window_backend not in {"brute_force", "optimized_bvh"}:
        raise ValueError("Mesh window profile backend is unsupported")
    if type(window_workers) is not int or not 1 <= window_workers <= 32:
        raise ValueError("Mesh window profile workers must be in [1, 32]")

    mesh_index = MeshIndex.load(args.mesh_index.resolve(), mesh_token=args.mesh_token)
    if mesh_index.build_settings != {"method": "binned_sah", "leaf_size": 8}:
        raise ValueError("Step 6 must reuse Step 4 binned_sah leaf_size=8 Mesh Index")
    anchor_source = file_identity(
        model_path / "point_cloud" / "iteration_40000" / "point_cloud.ply"
    )
    anchor_source["iteration"] = 40000
    anchor_index = AnchorPointIndex.load(
        args.anchor_index.resolve(), scene=args.scene, source=anchor_source
    )
    if anchor_index.leaf_capacity != args.leaf_capacity or anchor_index.max_depth != args.max_depth:
        raise ValueError("loaded Anchor Index settings differ from the frozen CLI settings")
    anchor_positions = np.ascontiguousarray(
        model.get_anchor.detach().cpu().numpy(), dtype=np.float32
    )
    if not np.array_equal(anchor_positions, anchor_index.positions):
        raise ValueError("Anchor Index positions differ from loaded Step 2 final PLY rows")
    rasterizer = OnlineProxyDepthRasterizer(mesh_index, device="cuda:0")
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
        "anchor_index": file_identity(args.anchor_index.resolve()),
        "renderer_settings": file_identity(args.renderer_settings.resolve()),
        "explicit_selection_contract": file_identity(args.explicit_selection_contract.resolve()),
        "step4_summary": file_identity(args.step4_run.resolve() / "summary.json"),
        "step5_review": file_identity(STEP5 / "review" / "final_step5_review.json"),
        "window_selection": file_identity(args.selection.resolve()),
        "mesh_window_profile": file_identity(args.mesh_window_profile.resolve()),
    }
    contract = {
        "schema": "proxygs_step6_j3_window_scene_contract_v1",
        "protocol_id": PROTOCOL_ID,
        "state": "frozen_before_run",
        "scene": args.scene,
        "formal": args.formal,
        "qualification": False,
        "backend": BACKEND_ID,
        "camera_count": len(views),
        "frozen_camera_count": len(frozen_names),
        "camera_names": [view.image_name for view in views],
        "image_size": [args.width, args.height],
        "mesh_index_settings": mesh_index.build_settings,
        "anchor_index_settings": {
            "query_profile": "proxygs-pointwise-center-v1",
            "leaf_capacity": anchor_index.leaf_capacity,
            "max_depth": anchor_index.max_depth,
            "build_method": anchor_index.build_method,
            "anchor_count": anchor_index.anchor_count,
            "node_count": anchor_index.node_count,
        },
        "window": window,
        "g1_modes": ["G1-Ref", "G1-Brute", "G1-BVH", "G1-Retained"],
        "g2_modes": ["J0", "J1-Mesh", "J2-Anchor", "J3-Window"],
        "depth_margin": float(DEPTH_MARGIN),
        "candidate_order": "sorted_unique_original_final_ply_rows",
        "paired_order_rotation": "baseline-first-even-camera optimized-first-odd-camera",
        "cpu_threads": args.mesh_threads,
        "index_repeat": args.index_repeat,
        "optimized_mesh": "parallel fast-double prefilter with frozen long-double fallback",
        "optimized_anchor": "trusted audited inputs and persistent native scratch buffers",
        "window_specialization": "qualification-frozen exact backend plus worker profile; no temporal result reuse or restrictive hint",
        "window_mesh_batch": {
            "backend": window_backend,
            "workers": window_workers,
            "camera_count": 32,
            "inner_query_threads": 1,
            "qualification_profile": scene_profile,
            "timing": "paired whole-window wall elapsed, reported both total and amortized per frame",
        },
        "warmup": args.warmup,
        "repeat": args.repeat,
        "environment": _environment_record(),
        "inputs": inputs,
    }
    atomic_json(output_dir / "run_contract.json", contract)

    render_modes = ("g1_ref", "g1_brute", "g1_bvh", "g2_j0", "g2_j3_window")
    render_dirs = {mode: output_dir / mode / "renders" for mode in render_modes}
    if args.save_images:
        for directory in render_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    camera_records = []
    atomic_json(
        output_dir / "status.json",
        {"state": "running", "scene": args.scene, "expected_views": len(views)},
    )

    full_triangle_ids = np.arange(len(mesh_index.triangles), dtype=np.int64)
    window_domains = [camera_domain_from_view(view) for view in views]
    batch_mesh_results = []
    batch_mesh_elapsed_samples = []
    for _ in range(args.index_repeat):
        batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=window_workers) as workers:
            results = list(
                workers.map(
                    lambda domain: mesh_index.query(
                        domain, backend=window_backend, threads=1
                    ),
                    window_domains,
                )
            )
        batch_mesh_elapsed_samples.append((time.perf_counter() - batch_start) * 1000.0)
        batch_mesh_results = results
    batch_mesh_amortized_samples = [value / len(views) for value in batch_mesh_elapsed_samples]
    for view_index, view in enumerate(views):
        global_view_index = start + view_index
        domain = window_domains[view_index]
        camera_records.append(camera_record(view, domain, global_view_index))
        mesh_samples = {"brute": [], "j0": [], "j1": []}
        mesh_results = {}
        for sample in range(args.index_repeat):
            base_order = ("brute", "j0", "j1")
            shift = (global_view_index + sample) % len(base_order)
            order = base_order[shift:] + base_order[:shift]
            for mode in order:
                result = mesh_index.query(
                    domain,
                    backend={"brute": "brute_force", "j0": "bvh", "j1": "optimized_bvh"}[mode],
                    threads=1 if mode == "j0" else args.mesh_threads,
                )
                mesh_results[mode] = result
                mesh_samples[mode].append(float(result.elapsed_ms))
        j0_mesh = mesh_results["j0"]
        j1_mesh = mesh_results["j1"]
        brute_mesh = mesh_results["brute"]
        j3_window_mesh = batch_mesh_results[view_index]
        mesh_equal = (
            np.array_equal(brute_mesh.triangle_ids, j0_mesh.triangle_ids)
            and np.array_equal(j0_mesh.triangle_ids, j1_mesh.triangle_ids)
            and np.array_equal(j1_mesh.triangle_ids, j3_window_mesh.triangle_ids)
        )
        if not mesh_equal:
            raise RuntimeError(f"{view.image_name}: G1-Brute/J0/J1 Mesh IDs differ")
        full_depth = rasterizer.render(full_triangle_ids, domain, (args.width, args.height))
        brute_depth = rasterizer.render(brute_mesh.triangle_ids, domain, (args.width, args.height))
        depth = rasterizer.render(j1_mesh.triangle_ids, domain, (args.width, args.height))
        brute_bvh_depth_equal = np.array_equal(brute_depth.depth_cpu, depth.depth_cpu)
        full_bvh_depth_equal = np.array_equal(full_depth.depth_cpu, depth.depth_cpu)
        if not brute_bvh_depth_equal:
            raise RuntimeError(f"{view.image_name}: G1-Brute and G1-BVH depth differ")

        torch.cuda.synchronize()
        candidate_start = time.perf_counter()
        model.set_anchor_mask(view.camera_center, args.iteration, view.resolution_scale)
        candidate_ids = torch.nonzero(model._anchor_mask, as_tuple=False).flatten().detach().cpu().numpy().copy()
        torch.cuda.synchronize()
        candidate_ms = (time.perf_counter() - candidate_start) * 1000.0
        step5_payload = (
            args.step5_run / "id_payload" / f"{view.image_name}.npz"
        )
        saved_candidates, saved_selected, saved_triangles, anchor_count, triangle_count = (
            load_step5_ids(step5_payload)
        )
        if anchor_count != anchor_index.anchor_count or triangle_count != len(mesh_index.triangles):
            raise RuntimeError(f"{view.image_name}: Step 5 row universe drifted")
        if not np.array_equal(candidate_ids, saved_candidates):
            raise RuntimeError(f"{view.image_name}: Step 6 candidates drifted from Step 5")
        if not np.array_equal(j0_mesh.triangle_ids, saved_triangles):
            raise RuntimeError(f"{view.image_name}: Step 6 Mesh IDs drifted from Step 5")
        depth_values = depth.depth_cpu
        if np.any(np.isnan(depth_values)) or np.any(np.isneginf(depth_values)) or np.any(
            np.isfinite(depth_values) & (depth_values <= 0)
        ):
            raise RuntimeError(f"{view.image_name}: online depth violates the frozen predicate domain")

        world_view = np.ascontiguousarray(
            view.world_view_transform.detach().cpu().numpy(), dtype=np.float32
        )
        full_projection = np.ascontiguousarray(
            view.full_proj_transform.detach().cpu().numpy(), dtype=np.float32
        )
        anchor_samples = {"j0": [], "j2": []}
        anchor_results = {}
        for sample in range(args.index_repeat):
            order = ("j0", "j2") if (view_index + sample) % 2 == 0 else ("j2", "j0")
            for mode in order:
                result = anchor_index.query(
                    candidate_ids,
                    depth_values,
                    world_view,
                    full_projection,
                    mode="tree",
                    camera=str(view.image_name),
                    margin=float(DEPTH_MARGIN),
                    trusted_buffers=mode == "j2",
                    reuse_buffers=mode == "j2",
                )
                anchor_results[mode] = result
                anchor_samples[mode].append(
                    float(result.timings["anchor_index_total_ms"])
                )
        j0_anchor = anchor_results["j0"]
        j2_anchor = anchor_results["j2"]
        anchor_equal = np.array_equal(j0_anchor.selected_anchor_ids, j2_anchor.selected_anchor_ids)
        step5_equal = np.array_equal(j0_anchor.selected_anchor_ids, saved_selected)
        ranges_equal = np.array_equal(
            j2_anchor.expand_ranges(candidate_ids), j2_anchor.selected_anchor_ids
        )
        if not anchor_equal or not step5_equal or not ranges_equal:
            raise RuntimeError(f"{view.image_name}: J2 Anchor outputs differ from J0/Step 5")

        dense_ref_ids, dense_ref_record = dense_anchor_filter_cpu(
            candidate_ids, anchor_positions, world_view, full_projection, full_depth.depth_cpu,
            margin=float(DEPTH_MARGIN),
        )
        dense_brute_ids, dense_brute_record = dense_anchor_filter_cpu(
            candidate_ids, anchor_positions, world_view, full_projection, brute_depth.depth_cpu,
            margin=float(DEPTH_MARGIN),
        )
        dense_bvh_ids, dense_bvh_record = dense_anchor_filter_cpu(
            candidate_ids, anchor_positions, world_view, full_projection, depth.depth_cpu,
            margin=float(DEPTH_MARGIN),
        )
        dense_equal = (
            np.array_equal(dense_ref_ids, dense_brute_ids)
            and np.array_equal(dense_brute_ids, dense_bvh_ids)
            and np.array_equal(dense_bvh_ids, saved_selected)
        )
        if not dense_equal:
            raise RuntimeError(f"{view.image_name}: G1 Ref/Brute/BVH dense selected IDs differ")

        render_order = render_modes
        shift = view_index % len(render_order)
        render_order = render_order[shift:] + render_order[:shift]
        render_records: dict[str, dict[str, Any]] = {}
        images: dict[str, torch.Tensor] = {}
        h2d: dict[str, float] = {}
        selected_by_mode = {
            "g1_ref": dense_ref_ids,
            "g1_brute": dense_brute_ids,
            "g1_bvh": dense_bvh_ids,
            "g2_j0": j0_anchor.selected_anchor_ids,
            "g2_j3_window": j2_anchor.selected_anchor_ids,
        }
        for mode in render_order:
            record, image, h2d_ms = render_selected(
                view=view,
                model=model,
                pipeline=pipeline,
                background=background,
                ids=selected_by_mode[mode],
                warmup=args.warmup if view_index == 0 else 0,
                repeat=args.repeat,
                lpips_model=lpips_model,
            )
            render_records[mode] = record
            images[mode] = image
            h2d[mode] = h2d_ms
        image_exact = all(torch.equal(images["g1_ref"], images[mode]) for mode in render_order)
        maximum_image_delta = max(
            float(torch.max(torch.abs(images["g1_ref"] - images[mode]))) for mode in render_order
        )
        j0_tensor = tensor_contract_without_object_identity(render_records["g1_ref"]["tensor_identity"])
        ownership_equal = all(
            render_records["g1_ref"]["requested_anchor_count"] == render_records[mode]["requested_anchor_count"]
            and render_records["g1_ref"]["decoded_row_count"] == render_records[mode]["decoded_row_count"]
            and j0_tensor == tensor_contract_without_object_identity(render_records[mode]["tensor_identity"])
            for mode in render_order
        )
        metrics_equal = all(
            render_records["g1_ref"]["metrics"] == render_records[mode]["metrics"]
            for mode in render_order
        )
        if not image_exact or not ownership_equal or not metrics_equal:
            atomic_json(
                output_dir / "diagnostics" / f"{view.image_name}_render_parity.json",
                {
                    "camera": str(view.image_name),
                    "image_exact": image_exact,
                    "maximum_image_delta": maximum_image_delta,
                    "ownership_equal": ownership_equal,
                    "metrics_equal": metrics_equal,
                    "mode_records": render_records,
                },
            )
            raise RuntimeError(
                f"{view.image_name}: G1/G2 window render ownership or image parity failed"
            )

        write_ms = {mode: 0.0 for mode in render_order}
        if args.save_images:
            for mode in render_order:
                write_start = time.perf_counter()
                torchvision.utils.save_image(images[mode], render_dirs[mode] / f"{view.image_name}.png")
                write_ms[mode] = (time.perf_counter() - write_start) * 1000.0

        payload = output_dir / "id_payload" / f"{view.image_name}.npz"
        save_payload(
            payload,
            anchor_universe_count=np.asarray([anchor_index.anchor_count], dtype=np.int64),
            triangle_universe_count=np.asarray([len(mesh_index.triangles)], dtype=np.int64),
            candidate_bitmap=pack_id_bitmap(candidate_ids, anchor_index.anchor_count),
            selected_bitmap=pack_id_bitmap(j2_anchor.selected_anchor_ids, anchor_index.anchor_count),
            triangle_bitmap=pack_id_bitmap(
                np.ascontiguousarray(j1_mesh.triangle_ids, dtype=np.int64),
                len(mesh_index.triangles),
            ),
        )
        payload_record = {
            **file_identity(payload),
            "candidate_count": int(len(candidate_ids)),
            "selected_count": int(len(j2_anchor.selected_anchor_ids)),
            "triangle_count": int(len(j1_mesh.triangle_ids)),
            "id_dtype": "int64",
            "selected_order": "original_final_ply_row_order",
            "range_space": j2_anchor.range_space,
            "encoding": "numpy_packbits_original_row_bitmap",
            "bitorder": "little",
        }
        mesh_mean = {mode: mean(values) for mode, values in mesh_samples.items()}
        mesh_mean["j3_window"] = mean(batch_mesh_amortized_samples)
        anchor_mean = {mode: mean(values) for mode, values in anchor_samples.items()}
        selection_components = {
            "g1_ref": (0.0, full_depth.timings["depth_total_ms"], dense_ref_record["elapsed_ms"]),
            "g1_brute": (mesh_mean["brute"], brute_depth.timings["depth_total_ms"], dense_brute_record["elapsed_ms"]),
            "g1_bvh": (mesh_mean["j0"], depth.timings["depth_total_ms"], dense_bvh_record["elapsed_ms"]),
            "g2_j0": (mesh_mean["j0"], depth.timings["depth_total_ms"], anchor_mean["j0"]),
            "g2_j3_window": (mesh_mean["j3_window"], depth.timings["depth_total_ms"], anchor_mean["j2"]),
        }
        frame_totals = {
            mode: candidate_ms + mesh_time + depth_time + selection_time + h2d[mode]
            + render_records[mode]["decode_seconds"] * 1000.0
            + render_records[mode]["render_seconds_mean"] * 1000.0 + write_ms[mode]
            for mode, (mesh_time, depth_time, selection_time) in selection_components.items()
        }
        # The retained Mesh backend is ID-identical to G1-BVH, so it reuses the
        # exact same depth, dense selection, H2D, decode, render, and write
        # records.  Only the measured Mesh discovery component is substituted.
        frame_totals["g1_retained"] = (
            frame_totals["g1_bvh"] - mesh_mean["j0"] + mesh_mean["j3_window"]
        )
        record = {
            "index": global_view_index,
            "window_index": view_index,
            "camera": str(view.image_name),
            "mesh": {
                "brute": {"samples_ms": mesh_samples["brute"], "mean_ms": mesh_mean["brute"], "counters": brute_mesh.counters},
                "j0": {"samples_ms": mesh_samples["j0"], "mean_ms": mesh_mean["j0"], "counters": j0_mesh.counters},
                "j1": {"samples_ms": mesh_samples["j1"], "mean_ms": mesh_mean["j1"], "counters": j1_mesh.counters},
                "j3_window": {
                    "batch_elapsed_samples_ms": batch_mesh_elapsed_samples,
                    "amortized_samples_ms": batch_mesh_amortized_samples,
                    "mean_ms": mesh_mean["j3_window"],
                    "backend": window_backend,
                    "workers": window_workers,
                    "counters": j3_window_mesh.counters,
                },
                "returned_triangles": int(len(j1_mesh.triangle_ids)),
            },
            "depth": {
                "full": {"timings": full_depth.timings, "counters": full_depth.counters},
                "brute": {"timings": brute_depth.timings, "counters": brute_depth.counters},
                "bvh": {"timings": depth.timings, "counters": depth.counters},
            },
            "fov_lod_candidate_ms": candidate_ms,
            "candidate_anchor_count": int(len(candidate_ids)),
            "selected_anchor_count": int(len(j2_anchor.selected_anchor_ids)),
            "anchor": {
                "j0": {**query_record(j0_anchor), "samples_ms": anchor_samples["j0"], "mean_ms": anchor_mean["j0"]},
                "j2": {**query_record(j2_anchor), "samples_ms": anchor_samples["j2"], "mean_ms": anchor_mean["j2"]},
            },
            "dense_anchor": {
                "g1_ref": dense_ref_record,
                "g1_brute": dense_brute_record,
                "g1_bvh": dense_bvh_record,
            },
            "selected_ids_h2d_ms": h2d,
            "renders": render_records,
            "image_write_ms": write_ms,
            "frame_total_ms": frame_totals,
            "parity": {
                "step5_payload_reversible": True,
                "g1_brute_bvh_mesh_ids": mesh_equal,
                "g1_brute_bvh_depth": brute_bvh_depth_equal,
                "g1_ref_bvh_depth": full_bvh_depth_equal,
                "g1_dense_ids": dense_equal,
                "j2_anchor_ids": anchor_equal,
                "j2_ranges_expand": ranges_equal,
                "j0_vs_step5_ids": step5_equal,
                "decoded_ownership": ownership_equal,
                "render_exact": image_exact,
                "render_max_abs_delta": maximum_image_delta,
                "metrics_exact": metrics_equal,
            },
            "id_payload": payload_record,
        }
        records.append(record)
        atomic_json(output_dir / "per_view.json", records)
        atomic_json(
            output_dir / "status.json",
            {
                "state": "running",
                "scene": args.scene,
                "completed_views": len(records),
                "expected_views": len(views),
                "last_camera": str(view.image_name),
            },
        )
        del images, depth, brute_depth, full_depth
        torch.cuda.empty_cache()

    brute_mesh_times = [record["mesh"]["brute"]["mean_ms"] for record in records]
    j0_mesh_times = [record["mesh"]["j0"]["mean_ms"] for record in records]
    j1_mesh_times = [record["mesh"]["j1"]["mean_ms"] for record in records]
    j3_window_mesh_times = [record["mesh"]["j3_window"]["mean_ms"] for record in records]
    j0_anchor_times = [record["anchor"]["j0"]["mean_ms"] for record in records]
    j2_anchor_times = [record["anchor"]["j2"]["mean_ms"] for record in records]
    summary = {
        "schema": "proxygs_step6_j3_window_scene_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "state": "complete",
        "backend": BACKEND_ID,
        "scene": args.scene,
        "formal": args.formal,
        "qualification": False,
        "view_count": len(records),
        "frozen_view_count": len(frozen_names),
        "window": window,
        "correctness_failures": [],
        "parity": {
            key: all(record["parity"][key] for record in records)
            for key in (
                "step5_payload_reversible",
                "g1_brute_bvh_mesh_ids",
                "g1_brute_bvh_depth",
                "g1_ref_bvh_depth",
                "g1_dense_ids",
                "j2_anchor_ids",
                "j2_ranges_expand",
                "j0_vs_step5_ids",
                "decoded_ownership",
                "render_exact",
                "metrics_exact",
            )
        },
        "metrics": {mode: {name: mean(record["renders"][mode]["metrics"][name] for record in records if record["renders"][mode]["metrics"][name] is not None) for name in ("psnr", "ssim", "lpips")} for mode in render_modes},
        "timing_ms": {
            "g1_brute_mesh_sum": sum(brute_mesh_times),
            "j0_mesh_sum": sum(j0_mesh_times),
            "j1_mesh_sum": sum(j1_mesh_times),
            "mesh_speedup": sum(j0_mesh_times) / sum(j1_mesh_times),
            "j3_window_mesh_sum": sum(j3_window_mesh_times),
            "window_batch_mesh_speedup": sum(j0_mesh_times) / sum(j3_window_mesh_times),
            "j0_anchor_sum": sum(j0_anchor_times),
            "j2_anchor_sum": sum(j2_anchor_times),
            "anchor_speedup": sum(j0_anchor_times) / sum(j2_anchor_times),
            "j0_index_sum": sum(j0_mesh_times) + sum(j0_anchor_times),
            "j3_index_sum": sum(j1_mesh_times) + sum(j2_anchor_times),
            "joint_index_speedup": (sum(j0_mesh_times) + sum(j0_anchor_times)) / (sum(j1_mesh_times) + sum(j2_anchor_times)),
            "j3_window_index_sum": sum(j3_window_mesh_times) + sum(j2_anchor_times),
            "j3_window_joint_speedup": (sum(j0_mesh_times) + sum(j0_anchor_times)) / (sum(j3_window_mesh_times) + sum(j2_anchor_times)),
            "window_batch_mesh_elapsed_samples_ms": batch_mesh_elapsed_samples,
            "frame_sums": {
                mode: sum(record["frame_total_ms"][mode] for record in records)
                for mode in (*render_modes, "g1_retained")
            },
            "j0_mesh_p50": percentile(j0_mesh_times, 50),
            "j0_mesh_p95": percentile(j0_mesh_times, 95),
            "j0_mesh_p99": percentile(j0_mesh_times, 99),
            "j1_mesh_p50": percentile(j1_mesh_times, 50),
            "j1_mesh_p95": percentile(j1_mesh_times, 95),
            "j1_mesh_p99": percentile(j1_mesh_times, 99),
            "j0_anchor_p50": percentile(j0_anchor_times, 50),
            "j0_anchor_p95": percentile(j0_anchor_times, 95),
            "j0_anchor_p99": percentile(j0_anchor_times, 99),
            "j2_anchor_p50": percentile(j2_anchor_times, 50),
            "j2_anchor_p95": percentile(j2_anchor_times, 95),
            "j2_anchor_p99": percentile(j2_anchor_times, 99),
            "full_depth_mean": mean(record["depth"]["full"]["timings"]["depth_total_ms"] for record in records),
            "brute_depth_mean": mean(record["depth"]["brute"]["timings"]["depth_total_ms"] for record in records),
            "indexed_depth_mean": mean(record["depth"]["bvh"]["timings"]["depth_total_ms"] for record in records),
        },
        "input_identities_after": {name: file_identity(Path(value["path"])) for name, value in inputs.items()},
    }
    if summary["input_identities_after"] != inputs:
        raise RuntimeError("a frozen Step 2/3/4/5 input changed during the scene run")
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
    result.add_argument("--mesh-index", type=Path, required=True)
    result.add_argument("--mesh-token", required=True)
    result.add_argument("--anchor-index", type=Path, required=True)
    result.add_argument("--step4-run", type=Path, required=True)
    result.add_argument("--step5-run", type=Path, required=True)
    result.add_argument("--selection", type=Path, required=True)
    result.add_argument("--mesh-window-profile", type=Path, required=True)
    result.add_argument("--renderer-settings", type=Path, required=True)
    result.add_argument("--explicit-selection-contract", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--expected-views", type=int, required=True)
    result.add_argument("--iteration", type=int, default=40000)
    result.add_argument("--width", type=int, default=1600)
    result.add_argument("--height", type=int, default=900)
    result.add_argument("--margin", type=float, default=float(DEPTH_MARGIN))
    result.add_argument("--leaf-capacity", type=int, default=4096)
    result.add_argument("--max-depth", type=int, default=32)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--repeat", type=int, default=1)
    result.add_argument("--index-repeat", type=int, default=3)
    result.add_argument("--mesh-threads", type=int, default=16)
    result.add_argument("--window-workers", type=int, default=16)
    result.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=False)
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

"""Run Step 5 G2-Ref, CPU G2-Brute, and CPU G2-Index on one scene."""

from __future__ import annotations

import argparse
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

ARTIFACT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step5_cpu_anchor_index_g2_20260915")
STEP4 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step4_cpu_mesh_index_g1_v2_20260914")
PROTOCOL_ID = "proxygs-step5-g2-v1"
os.environ.setdefault("GDMGS_NATIVE_DIR", str(ARTIFACT / "native"))

from anchor_query_runtime import load_step4_payload, query_record, run_anchor_queries
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
    if args.formal and args.qualification:
        raise ValueError("formal and qualification modes are mutually exclusive")
    if (args.formal or args.qualification) and args.max_views is not None:
        raise ValueError("formal/qualification Step 5 cannot reduce the camera denominator")
    if args.qualification and args.scene != "amsterdam":
        raise ValueError("the preregistered complete-scene qualification is Amsterdam")
    if not args.formal and not args.qualification and args.max_views is None:
        raise ValueError("development mode requires --max-views")
    if args.iteration != 40000 or args.width != 1600 or args.height != 900:
        raise ValueError("Step 5 freezes iteration=40000 and image size=1600x900")
    if not np.isclose(args.margin, float(DEPTH_MARGIN), rtol=0.0, atol=1.0e-12):
        raise ValueError("Step 5 freezes the pointwise depth margin at float32 +0.3")

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
    if args.max_views is not None:
        views = views[: args.max_views]
    if any((int(view.image_width), int(view.image_height)) != (args.width, args.height) for view in views):
        raise ValueError("every Step 5 camera must be 1600x900")

    mesh_index = MeshIndex.load(args.mesh_index.resolve(), mesh_token=args.mesh_token)
    if mesh_index.build_settings != {"method": "binned_sah", "leaf_size": 8}:
        raise ValueError("Step 5 must reuse Step 4 binned_sah leaf_size=8 Mesh Index")
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
    }
    contract = {
        "schema": "proxygs_step5_g2_scene_contract_v1",
        "protocol_id": PROTOCOL_ID,
        "state": "frozen_before_run",
        "scene": args.scene,
        "formal": args.formal,
        "qualification": args.qualification,
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
        "three_modes": ["G2-Ref", "G2-Brute", "G2-Index"],
        "depth_margin": float(DEPTH_MARGIN),
        "candidate_order": "sorted_unique_original_final_ply_rows",
        "brute_index_order_rotation": "linear_first_even_camera_index_tree_first_odd",
        "cpu_threads": 1,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "environment": _environment_record(),
        "inputs": inputs,
    }
    atomic_json(output_dir / "run_contract.json", contract)

    brute_render_dir = output_dir / "g2_brute" / "renders"
    index_render_dir = output_dir / "g2_index" / "renders"
    if args.save_images:
        brute_render_dir.mkdir(parents=True, exist_ok=True)
        index_render_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    camera_records = []
    atomic_json(
        output_dir / "status.json",
        {"state": "running", "scene": args.scene, "expected_views": len(views)},
    )

    for view_index, view in enumerate(views):
        domain = camera_domain_from_view(view)
        camera_records.append(camera_record(view, domain, view_index))
        mesh = mesh_index.query(domain, backend="bvh")
        depth = rasterizer.render(mesh.triangle_ids, domain, (args.width, args.height))

        torch.cuda.synchronize()
        candidate_start = time.perf_counter()
        model.set_anchor_mask(view.camera_center, args.iteration, view.resolution_scale)
        candidate_ids = torch.nonzero(model._anchor_mask, as_tuple=False).flatten().detach().cpu().numpy().copy()
        torch.cuda.synchronize()
        candidate_ms = (time.perf_counter() - candidate_start) * 1000.0
        step4_candidates, step4_selected = load_step4_payload(
            args.step4_run / "id_payload" / f"{view.image_name}.npz"
        )
        if not np.array_equal(candidate_ids, step4_candidates):
            raise RuntimeError(f"{view.image_name}: Step 5 candidates drifted from Step 4")

        world_view = np.ascontiguousarray(
            view.world_view_transform.detach().cpu().numpy(), dtype=np.float32
        )
        full_projection = np.ascontiguousarray(
            view.full_proj_transform.detach().cpu().numpy(), dtype=np.float32
        )
        queries = run_anchor_queries(
            index=anchor_index,
            candidate_ids=candidate_ids,
            anchor_positions=anchor_positions,
            world_view_transform=world_view,
            full_proj_transform=full_projection,
            indexed_depth=depth.depth_cpu,
            camera=str(view.image_name),
            tree_first=bool(view_index % 2),
        )
        reference_ids = queries["reference_ids"]
        brute_ids = queries["brute"].selected_anchor_ids
        index_ids = queries["tree"].selected_anchor_ids
        step4_equal = np.array_equal(reference_ids, step4_selected)
        if not step4_equal:
            raise RuntimeError(f"{view.image_name}: G2-Ref differs from Step 4 G1-BVH IDs")

        render_order = ("index", "brute") if view_index % 2 else ("brute", "index")
        render_records: dict[str, dict[str, Any]] = {}
        images: dict[str, torch.Tensor] = {}
        h2d: dict[str, float] = {}
        selected_by_mode = {"brute": brute_ids, "index": index_ids}
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
        image_exact = bool(torch.equal(images["brute"], images["index"]))
        maximum_image_delta = float(torch.max(torch.abs(images["brute"] - images["index"])))
        ownership_equal = bool(
            render_records["brute"]["requested_anchor_count"]
            == render_records["index"]["requested_anchor_count"]
            and render_records["brute"]["decoded_row_count"]
            == render_records["index"]["decoded_row_count"]
            and tensor_contract_without_object_identity(
                render_records["brute"]["tensor_identity"]
            )
            == tensor_contract_without_object_identity(
                render_records["index"]["tensor_identity"]
            )
        )
        metrics_equal = render_records["brute"]["metrics"] == render_records["index"]["metrics"]
        if not image_exact or not ownership_equal or not metrics_equal:
            atomic_json(
                output_dir / "diagnostics" / f"{view.image_name}_render_parity.json",
                {
                    "camera": str(view.image_name),
                    "image_exact": image_exact,
                    "maximum_image_delta": maximum_image_delta,
                    "ownership_equal": ownership_equal,
                    "metrics_equal": metrics_equal,
                    "brute_requested": render_records["brute"]["requested_anchor_count"],
                    "index_requested": render_records["index"]["requested_anchor_count"],
                    "brute_decoded": render_records["brute"]["decoded_row_count"],
                    "index_decoded": render_records["index"]["decoded_row_count"],
                },
            )
            raise RuntimeError(
                f"{view.image_name}: brute/index render ownership or image parity failed"
            )

        brute_write_ms = index_write_ms = 0.0
        if args.save_images:
            write_start = time.perf_counter()
            torchvision.utils.save_image(images["brute"], brute_render_dir / f"{view.image_name}.png")
            brute_write_ms = (time.perf_counter() - write_start) * 1000.0
            write_start = time.perf_counter()
            torchvision.utils.save_image(images["index"], index_render_dir / f"{view.image_name}.png")
            index_write_ms = (time.perf_counter() - write_start) * 1000.0

        payload = output_dir / "id_payload" / f"{view.image_name}.npz"
        save_payload(
            payload,
            anchor_universe_count=np.asarray([anchor_index.anchor_count], dtype=np.int64),
            triangle_universe_count=np.asarray([len(mesh_index.triangles)], dtype=np.int64),
            candidate_bitmap=pack_id_bitmap(candidate_ids, anchor_index.anchor_count),
            selected_bitmap=pack_id_bitmap(index_ids, anchor_index.anchor_count),
            triangle_bitmap=pack_id_bitmap(
                np.ascontiguousarray(mesh.triangle_ids, dtype=np.int64),
                len(mesh_index.triangles),
            ),
        )
        payload_record = {
            **file_identity(payload),
            "candidate_count": int(len(candidate_ids)),
            "selected_count": int(len(index_ids)),
            "triangle_count": int(len(mesh.triangle_ids)),
            "id_dtype": "int64",
            "selected_order": "original_final_ply_row_order",
            "range_space": queries["tree"].range_space,
            "encoding": "numpy_packbits_original_row_bitmap",
            "bitorder": "little",
        }
        shared_ms = candidate_ms + float(mesh.elapsed_ms) + depth.timings["depth_total_ms"]
        brute_total_ms = (
            shared_ms
            + queries["brute"].timings["anchor_index_total_ms"]
            + h2d["brute"]
            + render_records["brute"]["decode_seconds"] * 1000.0
            + render_records["brute"]["render_seconds_mean"] * 1000.0
            + brute_write_ms
        )
        index_total_ms = (
            shared_ms
            + queries["tree"].timings["anchor_index_total_ms"]
            + h2d["index"]
            + render_records["index"]["decode_seconds"] * 1000.0
            + render_records["index"]["render_seconds_mean"] * 1000.0
            + index_write_ms
        )
        record = {
            "index": view_index,
            "camera": str(view.image_name),
            "mesh_query_cpu_ms": float(mesh.elapsed_ms),
            "mesh_returned_triangles": int(len(mesh.triangle_ids)),
            "mesh_query_counters": mesh.counters,
            "indexed_depth": {"timings": depth.timings, "counters": depth.counters},
            "fov_lod_candidate_ms": candidate_ms,
            "candidate_anchor_count": int(len(candidate_ids)),
            "selected_anchor_count": int(len(index_ids)),
            "g2_reference": {
                "selected_count": int(len(reference_ids)),
                "query": queries["reference_counters"],
            },
            "g2_brute": query_record(queries["brute"]),
            "g2_index": query_record(queries["tree"]),
            "selected_ids_h2d_ms": h2d,
            "g2_brute_render": render_records["brute"],
            "g2_index_render": render_records["index"],
            "g2_brute_image_write_ms": brute_write_ms,
            "g2_index_image_write_ms": index_write_ms,
            "g2_brute_frame_total_ms": brute_total_ms,
            "g2_index_frame_total_ms": index_total_ms,
            "parity": {
                "step4_g1_bvh_vs_g2_ref_ids": step4_equal,
                **queries["parity"],
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
        del images, depth
        torch.cuda.empty_cache()

    brute_times = [record["g2_brute"]["timings"]["anchor_index_total_ms"] for record in records]
    index_times = [record["g2_index"]["timings"]["anchor_index_total_ms"] for record in records]
    brute_frames = [record["g2_brute_frame_total_ms"] for record in records]
    index_frames = [record["g2_index_frame_total_ms"] for record in records]
    summary = {
        "schema": "proxygs_step5_g2_scene_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "state": "complete",
        "backend": BACKEND_ID,
        "scene": args.scene,
        "formal": args.formal,
        "qualification": args.qualification,
        "view_count": len(records),
        "frozen_view_count": len(frozen_names),
        "correctness_failures": [],
        "parity": {
            key: all(record["parity"][key] for record in records)
            for key in (
                "step4_g1_bvh_vs_g2_ref_ids",
                "reference_vs_brute_ids",
                "brute_vs_index_ids",
                "brute_ranges_expand",
                "index_ranges_expand",
                "original_order",
                "decoded_ownership",
                "render_exact",
                "metrics_exact",
            )
        },
        "g2_brute_metrics": {
            name: mean(
                record["g2_brute_render"]["metrics"][name]
                for record in records
                if record["g2_brute_render"]["metrics"][name] is not None
            )
            for name in ("psnr", "ssim", "lpips")
        },
        "g2_index_metrics": {
            name: mean(
                record["g2_index_render"]["metrics"][name]
                for record in records
                if record["g2_index_render"]["metrics"][name] is not None
            )
            for name in ("psnr", "ssim", "lpips")
        },
        "timing_ms": {
            "brute_anchor_query_sum": sum(brute_times),
            "anchor_index_query_sum": sum(index_times),
            "anchor_component_speedup": sum(brute_times) / sum(index_times),
            "g2_brute_frame_sum": sum(brute_frames),
            "g2_index_frame_sum": sum(index_frames),
            "complete_frame_speedup": sum(brute_frames) / sum(index_frames),
            "brute_anchor_query_mean": mean(brute_times),
            "anchor_index_query_mean": mean(index_times),
            "mesh_query_mean": mean(record["mesh_query_cpu_ms"] for record in records),
            "indexed_depth_mean": mean(record["indexed_depth"]["timings"]["depth_total_ms"] for record in records),
            "brute_anchor_query_p50": percentile(brute_times, 50),
            "brute_anchor_query_p95": percentile(brute_times, 95),
            "brute_anchor_query_p99": percentile(brute_times, 99),
            "anchor_index_query_p50": percentile(index_times, 50),
            "anchor_index_query_p95": percentile(index_times, 95),
            "anchor_index_query_p99": percentile(index_times, 99),
        },
        "index_efficiency": {
            "candidate_anchors": sum(record["candidate_anchor_count"] for record in records),
            "selected_anchors": sum(record["selected_anchor_count"] for record in records),
            "certified_nodes": sum(record["g2_index"]["counters"]["certified_nodes"] for record in records),
            "certified_anchors": sum(record["g2_index"]["counters"]["certified_anchors"] for record in records),
            "fallback_checks": sum(record["g2_index"]["counters"]["anchor_fallback_checks"] for record in records),
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
    result.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
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

"""Full-camera, query-only Anchor Index ablation with interleaved repeats."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step5_cpu_anchor_index_g2_20260915")
STEP4 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step4_cpu_mesh_index_g1_v2_20260914")
RUNTIME = ROOT / "runtime" / "Proxy-GS-eac937e8"
sys.path.insert(0, str(RUNTIME))
os.environ.setdefault("GDMGS_NATIVE_DIR", str(ROOT / "native"))

from anchor_query_runtime import load_step4_payload
from gdmgs.anchor_index import AnchorPointIndex
from gdmgs.mesh_index import MeshIndex
from online_proxy_depth import OnlineProxyDepthRasterizer
from render_gdmgs_backend import _frozen_camera_names, _load_cfg, _new_model, _ordered_views
from step4_runtime import DEPTH_MARGIN, MIN_CAMERA_Z, atomic_json, camera_domain_from_view, dense_anchor_filter_cpu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mesh-index", type=Path, required=True)
    parser.add_argument("--mesh-token", required=True)
    parser.add_argument("--anchor-index", type=Path, required=True)
    parser.add_argument("--step4-run", type=Path, required=True)
    parser.add_argument("--expected-views", type=int, required=True)
    parser.add_argument("--leaf-capacity", type=int, required=True)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "status.json", {"state": "starting", "scene": args.scene})

    cfg = _load_cfg(args.model_path.resolve())
    if Path(cfg.source_path).resolve() != args.source_path.resolve():
        raise ValueError("source path differs from frozen model cfg")
    cfg.source_path = str(args.source_path.resolve())
    cfg.data_device = "cpu"
    model = _new_model(cfg)
    from scene import Scene

    scene = Scene(cfg, model, load_iteration=40000, shuffle=False, resolution_scales=cfg.resolution_scales)
    model.eval()
    views = _ordered_views(scene, _frozen_camera_names(args.model_path.resolve()))
    if len(views) != args.expected_views:
        raise ValueError("full-camera ablation denominator differs from expected views")
    mesh = MeshIndex.load(args.mesh_index.resolve(), mesh_token=args.mesh_token)
    anchors = AnchorPointIndex.load(args.anchor_index.resolve(), scene=args.scene)
    if anchors.leaf_capacity != args.leaf_capacity or anchors.max_depth != args.max_depth:
        raise ValueError("Anchor Index settings differ from ablation CLI")
    anchor_positions = np.ascontiguousarray(model.get_anchor.detach().cpu().numpy(), dtype=np.float32)
    if not np.array_equal(anchor_positions, anchors.positions):
        raise ValueError("Anchor Index rows differ from the loaded final PLY")
    rasterizer = OnlineProxyDepthRasterizer(mesh, device="cuda:0")
    records = []
    for camera_index, view in enumerate(views):
        domain = camera_domain_from_view(view)
        mesh_query = mesh.query(domain, backend="bvh")
        depth = rasterizer.render(mesh_query.triangle_ids, domain, (1600, 900))
        model.set_anchor_mask(view.camera_center, 40000, view.resolution_scale)
        candidate_ids = torch.nonzero(model._anchor_mask, as_tuple=False).flatten().detach().cpu().numpy().copy()
        saved_candidates, saved_ids = load_step4_payload(
            args.step4_run / "id_payload" / f"{view.image_name}.npz"
        )
        if not np.array_equal(candidate_ids, saved_candidates):
            raise RuntimeError(f"{view.image_name}: candidate IDs drifted from Step 4")
        world_view = np.ascontiguousarray(view.world_view_transform.detach().cpu().numpy(), dtype=np.float32)
        projection = np.ascontiguousarray(view.full_proj_transform.detach().cpu().numpy(), dtype=np.float32)
        reference, _ = dense_anchor_filter_cpu(
            candidate_ids,
            anchor_positions,
            world_view,
            projection,
            depth.depth_cpu,
            margin=float(DEPTH_MARGIN),
        )
        if not np.array_equal(reference, saved_ids):
            raise RuntimeError(f"{view.image_name}: dense reference drifted from Step 4")
        timings = {"linear": [], "tree": []}
        counters = None
        for repeat in range(args.repeats):
            order = ("tree", "linear") if (camera_index + repeat) % 2 else ("linear", "tree")
            results = {}
            for mode in order:
                results[mode] = anchors.query(
                    candidate_ids,
                    depth.depth_cpu,
                    world_view,
                    projection,
                    mode=mode,
                    camera=str(view.image_name),
                    margin=float(DEPTH_MARGIN),
                    minimum_camera_z=float(MIN_CAMERA_Z),
                )
                timings[mode].append(results[mode].timings["anchor_index_total_ms"])
            if not (
                np.array_equal(results["linear"].selected_anchor_ids, reference)
                and np.array_equal(results["tree"].selected_anchor_ids, reference)
            ):
                raise RuntimeError(f"{view.image_name}: query ablation ID parity failed")
            counters = results["tree"].counters
        record = {
            "index": camera_index,
            "camera": str(view.image_name),
            "candidate_count": int(len(candidate_ids)),
            "selected_count": int(len(reference)),
            "linear_ms": timings["linear"],
            "tree_ms": timings["tree"],
            "linear_mean_ms": sum(timings["linear"]) / args.repeats,
            "tree_mean_ms": sum(timings["tree"]) / args.repeats,
            "tree_counters_last_repeat": counters,
            "parity": True,
        }
        records.append(record)
        atomic_json(output / "per_view.json", records)
        atomic_json(
            output / "status.json",
            {
                "state": "running",
                "scene": args.scene,
                "completed_views": len(records),
                "expected_views": len(views),
            },
        )
    linear_sum = sum(sum(record["linear_ms"]) for record in records)
    tree_sum = sum(sum(record["tree_ms"]) for record in records)
    summary = {
        "schema": "proxygs_step5_anchor_query_ablation_v1",
        "state": "complete",
        "scene": args.scene,
        "view_count": len(records),
        "repeats": args.repeats,
        "leaf_capacity": args.leaf_capacity,
        "max_depth": args.max_depth,
        "linear_sum_ms": linear_sum,
        "tree_sum_ms": tree_sum,
        "tree_speedup": linear_sum / tree_sum,
        "linear_mean_ms": linear_sum / (len(records) * args.repeats),
        "tree_mean_ms": tree_sum / (len(records) * args.repeats),
        "all_ids_exact": all(record["parity"] for record in records),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "status.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

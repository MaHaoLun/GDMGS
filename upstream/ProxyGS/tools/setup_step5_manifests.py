"""Freeze and validate the complete Step 4 to Step 5 handoff."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step5_cpu_anchor_index_g2_20260915")
RUNTIME = ROOT / "runtime" / "Proxy-GS-eac937e8"
STEP2 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step2_20260913")
STEP3 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step3_gdmgs_backend_20260914")
STEP4 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step4_cpu_mesh_index_g1_v2_20260914")
SOURCE = ROOT / "source" / "step5_implementation_bvh_bitmap"
sys.path.insert(0, str(RUNTIME))

from gdmgs.anchor_index import AnchorPointIndex
from gdmgs.anchor_index.point_index import file_identity, source_identity
from step4_runtime import atomic_json, source_equal


SCENES = {
    "amsterdam": 161,
    "barcelona": 160,
    "bilbao": 129,
    "chicago": 160,
    "hollywood": 125,
    "pompidou": 161,
    "quebec": 160,
    "rome": 158,
}
RUN_ID = "formal_step4_g1_v2_20260914"
IMPLEMENTATION_FILES = (
    "gdmgs/anchor_index/__init__.py",
    "gdmgs/anchor_index/point_index.py",
    "gdmgs/anchor_index/native/anchor_point_native.cpp",
    "gdmgs/anchor_index/native/CMakeLists.txt",
    "anchor_query_runtime.py",
    "render_g2.py",
    "tools/build_step5_anchor_indices.py",
    "tools/setup_step5_manifests.py",
    "tools/run_step5_scene.sh",
    "tools/final_step5_review.py",
    "tools/benchmark_step5_anchor_queries.py",
    "tests/test_step5_anchor_queries.py",
)


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    step4_review_path = STEP4 / "review" / "final_step4_v2_review.json"
    step4_review = load(step4_review_path)
    if not (
        step4_review.get("scene_count") == 8
        and step4_review.get("expected_view_count") == 1214
        and step4_review.get("attempted_view_count") == 1214
        and step4_review.get("failed_scene_count") == 0
        and not step4_review.get("failures")
    ):
        raise RuntimeError("Step 4 final review is not the complete passing prerequisite")

    bindings = []
    for scene, expected_views in SCENES.items():
        step4_run = STEP4 / "runs" / "formal" / scene / RUN_ID
        summary = load(step4_run / "summary.json")
        per_view = load(step4_run / "per_view.json")
        if not (
            summary.get("state") == "complete"
            and summary.get("view_count") == expected_views
            and len(per_view) == expected_views
            and all(summary.get("parity", {}).values())
        ):
            raise RuntimeError(f"{scene}: incomplete Step 4 formal handoff")
        camera_names = [record["camera"] for record in per_view]
        if len(set(camera_names)) != expected_views:
            raise RuntimeError(f"{scene}: Step 4 camera inventory has duplicates")
        payloads = [Path(record["id_payload"]["path"]) for record in per_view]
        if any(not path.is_file() for path in payloads):
            raise RuntimeError(f"{scene}: missing Step 4 ID payload")
        model = STEP2 / "runs" / "bungee" / scene / "formal_proxygs_native_40k_20260913"
        ply = model / "point_cloud" / "iteration_40000" / "point_cloud.ply"
        index_path = ROOT / "indices" / scene / "anchor_point_bvh.npz"
        source = source_identity(ply)
        index = AnchorPointIndex.load(index_path, scene=scene, source=source)
        if index.leaf_capacity != 4096 or index.max_depth != 32:
            raise RuntimeError(f"{scene}: formal Anchor Index settings are not frozen")
        bindings.append(
            {
                "scene": scene,
                "expected_views": expected_views,
                "camera_names": camera_names,
                "step4_run": str(step4_run),
                "step4_summary": file_identity(step4_run / "summary.json"),
                "step4_per_view": file_identity(step4_run / "per_view.json"),
                "step4_id_payload_count": len(payloads),
                "step4_mesh_index": file_identity(STEP4 / "indices" / scene / "mesh_bvh.npz"),
                "step2_final_ply": source,
                "step5_anchor_index": file_identity(index_path),
                "step5_anchor_count": index.anchor_count,
                "step5_node_count": index.node_count,
            }
        )

    atomic_json(
        ROOT / "manifests" / "step5_input_binding_manifest.json",
        {
            "schema": "proxygs_step5_input_binding_v1",
            "protocol_id": "proxygs-step5-g2-v1",
            "status": "pass",
            "hash_or_checksum_operations": False,
            "scene_count": len(bindings),
            "view_count": sum(item["expected_views"] for item in bindings),
            "step4_final_review": file_identity(step4_review_path),
            "step3_renderer_settings": file_identity(STEP3 / "manifests" / "renderer_settings.json"),
            "step3_explicit_selection_contract": file_identity(
                STEP3 / "manifests" / "explicit_selection_contract.json"
            ),
            "bindings": bindings,
        },
    )
    atomic_json(
        ROOT / "manifests" / "anchor_query_contract.json",
        {
            "schema": "proxygs_step5_anchor_query_contract_v1",
            "protocol_id": "proxygs-step5-g2-v1",
            "status": "frozen_before_qualification",
            "modes": {
                "G2-Ref": "Step4 dense_anchor_filter_cpu rerun on current indexed depth",
                "G2-Brute": "CPU native linear query over every FoV/LoD candidate",
                "G2-Index": "CPU binned-SAH point BVH with spatial terminal states and exact source-order pointwise fallback",
            },
            "query_profile": "proxygs-pointwise-center-v1",
            "minimum_camera_z": float(np.float32(0.0001)),
            "depth_margin": float(np.float32(0.3)),
            "out_of_image": "keep",
            "positive_infinity_depth": "keep",
            "finite_rule": "keep iff float32 camera-z <= pixel depth + float32 0.3",
            "pixel_mapping": "ProxyGS full_proj_transform and truncation-to-int",
            "output_order": "sorted unique original final PLY row order",
            "range_space": "candidate source ordinal half-open",
            "leaf_capacity": 4096,
            "max_depth": 32,
            "build_method": "binned_sah",
            "certificate_minimum_candidates": 256,
            "terminal_outside_keep": {
                "enabled": True,
                "rule": "all centers have positive camera-z and conservative projection envelope is entirely outside image",
                "effect": "exact predicate Keep; never changes selected IDs",
            },
            "terminal_nonpositive_cull": "node camera-z upper bound <= float32 minimum camera-z",
            "terminal_depth_keep": "disabled after negative ablation; extra minimum pyramid cost exceeded saved fallback work",
            "unknown_near_partial": "descend to children or exact pointwise leaf fallback",
            "depth_range_max": "disabled after negative ablation; depth culls use exact fused pointwise fallback",
            "normal_path_fallback": False,
            "formal_id_evidence": {
                "encoding": "numpy packbits over original global row universes",
                "bitorder": "little",
                "arrays": ["candidate_bitmap", "selected_bitmap", "triangle_bitmap"],
                "reason": "lossless Step6 handoff without duplicate multi-million-row ID arrays",
            },
        },
    )
    comparisons = []
    for relative in IMPLEMENTATION_FILES:
        frozen = SOURCE / relative
        runtime = RUNTIME / relative
        if not frozen.is_file() or not runtime.is_file():
            raise RuntimeError(f"missing frozen/runtime implementation file: {relative}")
        equal = source_equal(frozen, runtime)
        comparisons.append(
            {
                "relative_path": relative,
                "frozen": file_identity(frozen),
                "runtime": file_identity(runtime),
                "direct_bytes_equal": equal,
            }
        )
        if not equal:
            raise RuntimeError(f"Step 5 runtime source drift: {relative}")
    native_modules = sorted((ROOT / "native").glob("*.so"))
    if {path.name.split(".cpython")[0] for path in native_modules} != {
        "GDMGS_mesh_native",
        "ProxyGS_anchor_point_native",
    }:
        raise RuntimeError("Step 5 native module set is incomplete or ambiguous")
    atomic_json(
        ROOT / "manifests" / "step5_source_manifest.json",
        {
            "schema": "proxygs_step5_source_manifest_v1",
            "protocol_id": "proxygs-step5-g2-v1",
            "status": "pass",
            "hash_or_checksum_operations": False,
            "comparisons": comparisons,
            "native_modules": [file_identity(path) for path in native_modules],
            "step4_runtime_role": "copied immutable starting point; Step4 artifact remains unmodified",
        },
    )
    octree_qualification = load(
        ROOT / "review" / "ablation_octree_full_qualification_summary.json"
    )
    bvh_before_bitmap = load(
        ROOT
        / "runs"
        / "ablation"
        / "amsterdam_bvh_l4096_spatial_terminal_only"
        / "summary.json"
    )
    bvh_global_bitmap = load(
        ROOT
        / "runs"
        / "ablation"
        / "amsterdam_bvh_l4096_global_row_bitmap"
        / "summary.json"
    )
    atomic_json(
        ROOT / "manifests" / "anchor_index_tuning_manifest.json",
        {
            "schema": "proxygs_step5_anchor_index_tuning_v1",
            "status": "frozen_before_qualification",
            "development_camera": "amsterdam complete 161-view camera order",
            "ablation_lineage": {
                "octree_full_qualification": {
                    "view_count": octree_qualification["view_count"],
                    "tree_speedup": octree_qualification["timing_ms"]["anchor_component_speedup"],
                    "retained": False,
                },
                "binned_sah_rank_bitmap": {
                    "view_count": bvh_before_bitmap["view_count"],
                    "repeats": bvh_before_bitmap["repeats"],
                    "tree_speedup": bvh_before_bitmap["tree_speedup"],
                    "retained": False,
                },
                "binned_sah_global_row_bitmap": {
                    "view_count": bvh_global_bitmap["view_count"],
                    "repeats": bvh_global_bitmap["repeats"],
                    "tree_speedup": bvh_global_bitmap["tree_speedup"],
                    "all_ids_exact": bvh_global_bitmap["all_ids_exact"],
                    "retained": True,
                },
                "depth_min_keep": {
                    "retained": False,
                    "reason": "extra min-depth pyramid bandwidth exceeded saved pointwise fallback work",
                },
                "depth_max_node_cull": {
                    "retained": False,
                    "reason": "range hierarchy and rectangle-query cost exceeded rare certified occlusion nodes",
                },
            },
            "selected_leaf_capacity": 4096,
            "selected_max_depth": 32,
            "selected_build_method": "binned_sah",
            "selected_materialization": "global_row_terminal_state_bitmap_then_single_ordered_scan",
            "selection_reason": "complete Amsterdam 161-view 3-repeat exact-ID ablation retained only the faster candidate",
            "qualification_may_measure_but_not_retune": True,
        },
    )
    atomic_json(
        ROOT / "manifests" / "environment_manifest.json",
        {
            "schema": "proxygs_step5_environment_v1",
            "protocol_id": "proxygs-step5-g2-v1",
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cpu": command_output(["lscpu"]),
            "compiler": command_output(["g++", "--version"]).splitlines()[0],
            "cpu_threads_frozen": 1,
            "openblas_threads_frozen": 1,
            "omp_threads_frozen": 1,
            "mkl_threads_frozen": 1,
            "gpu_execution": "one complete scene per process; GPU selected by frozen lane command",
        },
    )
    print(json.dumps({"status": "pass", "scene_count": 8, "view_count": 1214}, indent=2))


if __name__ == "__main__":
    main()

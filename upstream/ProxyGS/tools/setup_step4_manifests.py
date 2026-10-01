"""Freeze and validate Step 4 inputs, source copies, and runtime protocols."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import nvdiffrast
import numpy as np
import torch

ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step4_cpu_mesh_index_g1_v2_20260914")
RUNTIME = ROOT / "runtime" / "Proxy-GS-eac937e8"
sys.path.insert(0, str(RUNTIME))

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
    file_identity,
    source_equal,
    validate_mesh_input,
)


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
STEP1 = Path("/ssddata/lun/gdmgs_artifacts/proxy_mesh_20260911/proxy_mesh/bungee")
STEP2 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step2_20260913")
STEP3 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step3_gdmgs_backend_20260914")
PROTOCOL_ID = "proxygs-step4-g1-v2"


def command_output(command):
    return subprocess.check_output(command, text=True).strip()


def main() -> None:
    step3_review = json.loads((STEP3 / "review" / "final_gdmgs_backend_review.json").read_text())
    if not (
        step3_review.get("status") == "pass"
        and step3_review.get("scene_count") == 8
        and step3_review.get("view_count") == 1214
        and not step3_review.get("failures")
    ):
        raise RuntimeError("Step 3 final review is not the complete passing prerequisite")
    bindings = []
    for scene, expected_views in SCENES.items():
        mesh_input = STEP1 / scene / "cpu_mesh_index_input.npz"
        mesh = validate_mesh_input(mesh_input, scene)
        model = STEP2 / "runs" / "bungee" / scene / "formal_proxygs_native_40k_20260913"
        camera_records = json.loads((model / "cameras.json").read_text())
        camera_names = [record["img_name"] for record in camera_records]
        if len(camera_names) != expected_views or len(set(camera_names)) != expected_views:
            raise RuntimeError(f"{scene}: incomplete or duplicate camera inventory")
        depth_dir = STEP2 / "depth" / "bungee" / scene
        depth_names = sorted(path.stem for path in depth_dir.glob("*.npy"))
        if depth_names != sorted(camera_names):
            raise RuntimeError(f"{scene}: Step 2 depth oracle inventory differs from cameras")
        first_depth = np.load(depth_dir / f"{camera_names[0]}.npy", allow_pickle=False, mmap_mode="r")
        if first_depth.shape != (900, 1600) or first_depth.dtype != np.float32:
            raise RuntimeError(f"{scene}: depth oracle shape/dtype is not frozen 1600x900 float32")
        iteration = model / "point_cloud" / "iteration_40000"
        bindings.append(
            {
                "scene": scene,
                "expected_views": expected_views,
                "camera_names": camera_names,
                "source_path": f"/ssddata/lun/data/bungeenerf/{scene}",
                "mesh": mesh,
                "mesh_index": file_identity(ROOT / "indices" / scene / "mesh_bvh.npz"),
                "depth_oracle_dir": str(depth_dir),
                "depth_oracle_count": len(depth_names),
                "model_path": str(model),
                "model_files": {
                    "cfg_args": file_identity(model / "cfg_args"),
                    "cameras_json": file_identity(model / "cameras.json"),
                    "point_cloud": file_identity(iteration / "point_cloud.ply"),
                    "opacity_mlp": file_identity(iteration / "opacity_mlp.pt"),
                    "cov_mlp": file_identity(iteration / "cov_mlp.pt"),
                    "color_mlp": file_identity(iteration / "color_mlp.pt"),
                },
            }
        )
    atomic_json(
        ROOT / "manifests" / "step4_input_binding_manifest.json",
        {
            "schema": "proxygs_step4_input_binding_v2",
            "protocol_id": PROTOCOL_ID,
            "status": "pass",
            "hash_or_checksum_operations": False,
            "scene_count": len(bindings),
            "view_count": sum(item["expected_views"] for item in bindings),
            "step3_final_review": file_identity(STEP3 / "review" / "final_gdmgs_backend_review.json"),
            "renderer_settings": file_identity(STEP3 / "manifests" / "renderer_settings.json"),
            "explicit_selection_contract": file_identity(STEP3 / "manifests" / "explicit_selection_contract.json"),
            "bindings": bindings,
        },
    )
    atomic_json(
        ROOT / "manifests" / "mesh_index_reuse_manifest.json",
        {
            "schema": "proxygs_step4_mesh_index_reuse_v2",
            "protocol_id": PROTOCOL_ID,
            "status": "pass",
            "role": "immutable input copied before v1 artifact cleanup",
            "rebuild_per_frame": False,
            "scene_count": len(bindings),
            "indices": [item["mesh_index"] for item in bindings],
        },
    )

    comparisons = []
    for relative in (
        "gdmgs/mesh_index/index.py",
        "gdmgs/mesh_index/gpu_index.py",
        "gdmgs/mesh_index/native/mesh_native.cpp",
        "gdmgs/mesh_index/native/CMakeLists.txt",
        "gdmgs/query/depth_pyramid.py",
    ):
        frozen = ROOT / "source" / "gdmgs_reference" / relative
        runtime = RUNTIME / relative
        equal = source_equal(frozen, runtime)
        comparisons.append({"relative_path": relative, "frozen": file_identity(frozen), "runtime": file_identity(runtime), "direct_bytes_equal": equal})
        if not equal:
            raise RuntimeError(f"runtime GDMGS reference drift: {relative}")
    implementation_comparisons = []
    for relative in (
        "step4_runtime.py",
        "online_proxy_depth.py",
        "render_g1.py",
        "tools/build_step4_mesh_indices.py",
        "tools/setup_step4_manifests.py",
        "tools/run_step4_scene.sh",
        "tools/final_step4_review.py",
        "tests/test_step4_contracts.py",
    ):
        frozen = ROOT / "source" / "step4_implementation" / relative
        runtime = RUNTIME / relative
        equal = source_equal(frozen, runtime)
        implementation_comparisons.append(
            {
                "relative_path": relative,
                "frozen": file_identity(frozen),
                "runtime": file_identity(runtime),
                "direct_bytes_equal": equal,
            }
        )
        if not equal:
            raise RuntimeError(f"runtime Step 4 implementation drift: {relative}")
    atomic_json(
        ROOT / "manifests" / "cpu_mesh_index_source_manifest.json",
        {
            "schema": "proxygs_step4_cpu_mesh_index_source_v2",
            "protocol_id": PROTOCOL_ID,
            "status": "pass",
            "hash_or_checksum_operations": False,
            "comparisons": comparisons,
            "implementation_comparisons": implementation_comparisons,
            "native_module": file_identity(next((ROOT / "native").glob("GDMGS_mesh_native*.so"))),
            "build_settings": {"method": "binned_sah", "leaf_size": 8},
        },
    )
    atomic_json(
        ROOT / "manifests" / "depth_protocol.json",
        {
            "schema": "proxygs_step4_depth_protocol_v2",
            "protocol_id": PROTOCOL_ID,
            "backend": "nvdiffrast_cuda_online",
            "width": 1600,
            "height": 900,
            "near": 0.01,
            "far": 100.0,
            "depth_semantics": "positive OpenCV camera-z; +inf background",
            "pixel_center": 0.5,
            "two_sided": True,
            "full_depth_d2h": True,
            "step2_oracle_policy": {
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
            "indexed_vs_full_policy": {
                "coverage_exact": True,
                "atol": INDEXED_DEPTH_ATOL,
                "rtol": INDEXED_DEPTH_RTOL,
            },
            "saved_depth_role": "diagnostic only; never substituted into formal online timing or counted as correctness failure",
        },
    )
    atomic_json(
        ROOT / "manifests" / "dense_anchor_predicate.json",
        {
            "schema": "proxygs_step4_dense_anchor_predicate_v2",
            "protocol_id": PROTOCOL_ID,
            "status": "frozen_before_v2_formal_matrix",
            "candidate_source": "ProxyGS set_anchor_mask at iteration 40000 and frozen resolution_scale",
            "minimum_camera_z": 0.0001,
            "out_of_image": "keep",
            "positive_infinity_depth": "keep",
            "finite_rule": "keep iff anchor camera-z <= pixel depth + 0.3",
            "margin": float(DEPTH_MARGIN),
            "pixel_mapping": "ProxyGS full_proj_transform and truncation-to-int",
            "order": "original FoV/LoD candidate row order",
            "output_dtype": "int64",
            "duplicates": "forbidden",
        },
    )
    atomic_json(
        ROOT / "manifests" / "environment_manifest.json",
        {
            "schema": "proxygs_step4_environment_v2",
            "protocol_id": PROTOCOL_ID,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvdiffrast": nvdiffrast.__version__,
            "cpu": command_output(["lscpu"]),
            "compiler": command_output(["g++", "--version"]).splitlines()[0],
            "cuda_home": os.environ.get("CUDA_HOME"),
            "cpu_threads_frozen": 1,
            "openblas_threads_frozen": 1,
            "omp_threads_frozen": 1,
            "mkl_threads_frozen": 1,
            "gpu_assignments": {"lane_a": "GPU 4", "lane_b": "GPU 5", "lane_c": "GPU 0", "lane_d": "GPU 1"},
        },
    )
    print(json.dumps({"status": "pass", "scene_count": len(bindings), "view_count": 1214}, indent=2))


if __name__ == "__main__":
    main()

"""Create no-checksum Step 3 input, source, environment, and contract manifests."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import gsplat
import torch
import torchvision


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


def identity(path: Path):
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    artifact = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step3_gdmgs_backend_20260914")
    step2 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step2_20260913")
    runtime = artifact / "runtime" / "Proxy-GS-eac937e8"
    bindings = []
    for scene, expected_views in SCENES.items():
        model = step2 / "runs" / "bungee" / scene / "formal_proxygs_native_40k_20260913"
        iteration = model / "point_cloud" / "iteration_40000"
        camera_records = json.loads((model / "cameras.json").read_text())
        if len(camera_records) != expected_views:
            raise ValueError(f"{scene}: expected {expected_views} cameras, found {len(camera_records)}")
        files = {
            "cfg_args": identity(model / "cfg_args"),
            "cameras_json": identity(model / "cameras.json"),
            "point_cloud": identity(iteration / "point_cloud.ply"),
            "opacity_mlp": identity(iteration / "opacity_mlp.pt"),
            "cov_mlp": identity(iteration / "cov_mlp.pt"),
            "color_mlp": identity(iteration / "color_mlp.pt"),
            "results_json": identity(model / "results.json"),
            "per_view_json": identity(model / "per_view.json"),
        }
        bindings.append(
            {
                "scene": scene,
                "expected_views": expected_views,
                "source_path": f"/ssddata/lun/data/bungeenerf/{scene}",
                "model_path": str(model),
                "camera_names": [record["img_name"] for record in camera_records],
                "files": files,
            }
        )
    write(
        artifact / "manifests" / "step3_input_binding_manifest.json",
        {
            "schema": "proxygs_step3_input_binding_v1",
            "hash_or_checksum_operations": False,
            "scene_count": len(bindings),
            "view_count": sum(item["expected_views"] for item in bindings),
            "bindings": bindings,
        },
    )

    source_files = [
        runtime / "gaussian_renderer" / "__init__.py",
        runtime / "gaussian_renderer" / "raster_batch.py",
        runtime / "gaussian_renderer" / "gdmgs_gsplat_backend.py",
        runtime / "gaussian_renderer" / "native_decoded_backend.py",
        runtime / "render_gdmgs_backend.py",
        runtime / "tools" / "validate_gdmgs_backend.py",
    ]
    write(
        artifact / "manifests" / "gdmgs_backend_source_manifest.json",
        {
            "schema": "proxygs_step3_gdmgs_backend_source_v1",
            "hash_or_checksum_operations": False,
            "gdmgs_reference": identity(artifact / "manifests" / "gdmgs_fvdb_native_renderer.source.py"),
            "runtime_files": [identity(path) for path in source_files],
            "backend_id": "gdmgs-gsplat-v1",
            "native_fallback": False,
        },
    )
    write(
        artifact / "manifests" / "environment_manifest.json",
        {
            "schema": "proxygs_step3_environment_v1",
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "gsplat": gsplat.__version__,
            "cuda_runtime": torch.version.cuda,
            "required_versions_match": {
                "torch_2_4_0_cu124": torch.__version__ == "2.4.0+cu124",
                "torchvision_0_19_0_cu124": torchvision.__version__ == "0.19.0+cu124",
                "gsplat_1_4_0_pt24cu124": gsplat.__version__ == "1.4.0+pt24cu124",
            },
        },
    )
    write(
        artifact / "manifests" / "renderer_settings.json",
        {
            "schema": "proxygs_step3_renderer_settings_v1",
            "backend_id": "gdmgs-gsplat-v1",
            "packed": False,
            "sh_degree": None,
            "colors": "precomputed RGB",
            "dtype": "float32",
            "contiguous": True,
            "viewmat": "world_view_transform.transpose(0, 1)",
            "K": "FoV-derived fx/fy with principal point width/2,height/2",
            "render_mode": "RGB",
            "background": [0.0, 0.0, 0.0],
            "width": 1600,
            "height": 900,
            "native_fallback": False,
        },
    )
    write(
        artifact / "manifests" / "explicit_selection_contract.json",
        {
            "schema": "proxygs_step3_explicit_selection_contract_v1",
            "status": "implemented_not_cpu_index",
            "all_mode": "ordered anchor IDs [0, anchor_count)",
            "explicit_mode": "ordered rank-one int64 IDs supplied per camera or globally",
            "rejects": ["duplicates", "negative IDs", "out-of-range IDs", "non-integer JSON", "missing camera binding"],
            "decoder_semantics": "gather preserves request order; opacity mask preserves row ownership",
            "renderer_secondary_selection": False,
        },
    )
    qualification = (
        artifact
        / "runs"
        / "gdmgs_backend_full"
        / "amsterdam"
        / "formal_g0_full_20260914"
        / "per_view.json"
    )
    if qualification.exists():
        record = json.loads(qualification.read_text())[0]
        write(
            artifact / "manifests" / "decoded_tensor_contract.json",
            {
                "schema": "proxygs_step3_decoded_tensor_contract_v1",
                "backend_id": "gdmgs-gsplat-v1",
                "qualification_scene": "amsterdam",
                "qualification_camera": record["camera"],
                "requested_anchor_count": record["requested_anchor_count"],
                "decoded_row_count": record["decoded_row_count"],
                "tensor_identity_at_handoff": record["tensor_identity"],
                "mapping": {
                    "xyz": "means",
                    "color": "precomputed RGB colors",
                    "opacity": "flattened opacities",
                    "scaling": "scales",
                    "rotation": "quaternions",
                    "selection_mask": "decoded-row identity and ownership filter",
                },
                "same_tensor_objects_for_native_diagnostic_and_gsplat": True,
                "silent_cast_before_contract_boundary": False,
                "row_order": "ordered anchor request, then increasing offset slot after opacity mask",
            },
        )
    patch_files = sorted(path for path in (artifact / "patches").iterdir() if path.is_file())
    write(
        artifact / "patches" / "patch_manifest.json",
        {
            "schema": "proxygs_step3_patch_manifest_v1",
            "hash_or_checksum_operations": False,
            "base_runtime": "/ssddata/lun/gdmgs_artifacts/proxygs_step2_20260913/runtime/Proxy-GS-eac937e8",
            "files": [identity(path) for path in patch_files if path.name != "patch_manifest.json"],
        },
    )


if __name__ == "__main__":
    main()

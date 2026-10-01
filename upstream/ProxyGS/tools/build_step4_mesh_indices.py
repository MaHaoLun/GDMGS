"""Build and reload the eight frozen Step 4 CPU mesh BVHs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step4_cpu_mesh_index_g1_v2_20260914")
RUNTIME = ROOT / "runtime" / "Proxy-GS-eac937e8"
sys.path.insert(0, str(RUNTIME))
os.environ.setdefault("GDMGS_NATIVE_DIR", str(ROOT / "native"))

from gdmgs.mesh_index import MeshIndex
from step4_runtime import atomic_json, file_identity, load_mesh_arrays, validate_mesh_input


SCENES = ("amsterdam", "barcelona", "bilbao", "chicago", "hollywood", "pompidou", "quebec", "rome")
STEP1 = Path("/ssddata/lun/gdmgs_artifacts/proxy_mesh_20260911/proxy_mesh/bungee")
SETTINGS = {"method": "binned_sah", "leaf_size": 8}


def main() -> None:
    reports = []
    for scene in SCENES:
        source = STEP1 / scene / "cpu_mesh_index_input.npz"
        validation = validate_mesh_input(source, scene)
        token = f"proxy_mesh_20260911:{scene}:{validation['input']['bytes']}:{validation['input']['mtime_ns']}"
        vertices, faces = load_mesh_arrays(source)
        index = MeshIndex(vertices, faces, mesh_token=token, **SETTINGS)
        output = ROOT / "indices" / scene / "mesh_bvh.npz"
        metadata = index.save(output)
        built_layout = index.inspect_layout()
        loaded = MeshIndex.load(output, mesh_token=token)
        loaded_layout = loaded.inspect_layout()
        checks = {
            "vertices_equal": bool((index.vertices == loaded.vertices).all()),
            "triangles_equal": bool((index.triangles == loaded.triangles).all()),
            "triangle_refs_equal": bool((built_layout["triangle_refs"] == loaded_layout["triangle_refs"]).all()),
            "nodes_equal": bool((built_layout["nodes"] == loaded_layout["nodes"]).all()),
            "bounds_equal": bool((built_layout["bounds"] == loaded_layout["bounds"]).all()),
            "every_face_once": bool(
                np.array_equal(
                    np.sort(loaded_layout["triangle_refs"]),
                    np.arange(len(faces), dtype=np.int64),
                )
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"{scene}: persisted BVH reload parity failed: {checks}")
        report = {
            **validation,
            "mesh_token": token,
            "settings": SETTINGS,
            "metadata": metadata,
            "index": file_identity(output),
            "load_ms": float(loaded.load_ms),
            "checks": checks,
            "status": "pass",
        }
        atomic_json(ROOT / "indices" / scene / "build_load_report.json", report)
        reports.append(report)
        del loaded, index, vertices, faces
    atomic_json(
        ROOT / "manifests" / "mesh_index_build_manifest.json",
        {
            "schema": "proxygs_step4_mesh_index_build_v2",
            "status": "pass",
            "scene_count": len(reports),
            "settings": SETTINGS,
            "reports": reports,
        },
    )
    print(json.dumps({"status": "pass", "scene_count": len(reports)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure persisted index load cost and owned-array memory outside formal queries."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from gdmgs.anchor_index import AnchorPointIndex
from gdmgs.mesh_index import MeshIndex


ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step6_high_overlap_cpu_index_20260915")
STEP2 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step2_20260913")
STEP4 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step4_cpu_mesh_index_g1_v2_20260914")
STEP5 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step5_cpu_anchor_index_g2_20260915")
SCENES = ("amsterdam", "barcelona", "bilbao", "chicago", "hollywood", "pompidou", "quebec", "rome")


def identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def array_bytes(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        return sum(data[name].nbytes for name in data.files if name != "metadata")


def main() -> None:
    records = []
    for scene in SCENES:
        mesh_path = STEP4 / "indices" / scene / "mesh_bvh.npz"
        anchor_path = STEP5 / "indices" / scene / "anchor_point_bvh.npz"
        with np.load(mesh_path, allow_pickle=False) as data:
            mesh_metadata = json.loads(str(data["metadata"].item()))
        mesh = MeshIndex.load(mesh_path, mesh_token=mesh_metadata["mesh_token"])
        point_cloud = STEP2 / "runs" / "bungee" / scene / "formal_proxygs_native_40k_20260913" / "point_cloud" / "iteration_40000" / "point_cloud.ply"
        anchor_source = identity(point_cloud)
        anchor_source["iteration"] = 40000
        anchor = AnchorPointIndex.load(anchor_path, scene=scene, source=anchor_source)
        records.append(
            {
                "scene": scene,
                "mesh": {
                    "file": identity(mesh_path),
                    "load_ms": mesh.load_ms,
                    "original_build_ms": mesh.build_ms,
                    "owned_array_bytes": array_bytes(mesh_path),
                    "triangle_count": len(mesh.triangles),
                },
                "anchor": {
                    "file": identity(anchor_path),
                    "load_ms": anchor.load_ms,
                    "original_build_ms": anchor.build_ms,
                    "owned_array_bytes": array_bytes(anchor_path),
                    "anchor_count": anchor.anchor_count,
                    "node_count": anchor.node_count,
                },
            }
        )
        del mesh, anchor
        gc.collect()
    report = {
        "schema": "proxygs_step6_index_load_memory_report_v1",
        "status": "pass",
        "accounting": "build/load and owned persisted arrays are excluded from per-frame formal query timing",
        "scene_count": len(records),
        "records": records,
    }
    atomic_json(ROOT / "review" / "index_load_memory_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

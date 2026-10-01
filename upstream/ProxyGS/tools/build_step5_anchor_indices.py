"""Build, persist, reload, and validate the eight full-PLY Step 5 indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData

ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step5_cpu_anchor_index_g2_20260915")
RUNTIME = ROOT / "runtime" / "Proxy-GS-eac937e8"
STEP2 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step2_20260913")
sys.path.insert(0, str(RUNTIME))

from gdmgs.anchor_index import AnchorPointIndex
from gdmgs.anchor_index.point_index import file_identity, source_identity
from step4_runtime import atomic_json


SCENES = (
    "amsterdam",
    "barcelona",
    "bilbao",
    "chicago",
    "hollywood",
    "pompidou",
    "quebec",
    "rome",
)


def load_positions(path: Path) -> np.ndarray:
    vertex = PlyData.read(path)["vertex"].data
    required = {"x", "y", "z"}
    if not required.issubset(vertex.dtype.names or ()):
        raise ValueError(f"{path}: final PLY has no x/y/z anchor rows")
    positions = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    if not len(positions) or not np.isfinite(positions).all():
        raise ValueError(f"{path}: anchor positions are empty or nonfinite")
    return positions


def validate_layout(index: AnchorPointIndex) -> dict:
    layout = index.layout()
    rows = np.arange(index.anchor_count, dtype=np.int64)
    checks = {
        "every_row_once": np.array_equal(np.sort(layout["dfs_to_row"]), rows),
        "dfs_rank_bijection": np.array_equal(
            layout["rank_of_row"][layout["dfs_to_row"]], rows
        ),
        "root_complete": bool(
            len(layout["intervals"])
            and np.array_equal(layout["intervals"][0], [0, index.anchor_count])
        ),
        "positions_exact": np.array_equal(layout["positions"], index.positions),
    }
    if not all(checks.values()):
        raise RuntimeError(f"anchor index topology validation failed: {checks}")
    return checks


def build_scene(scene: str, leaf_capacity: int, max_depth: int) -> dict:
    ply = (
        STEP2
        / "runs"
        / "bungee"
        / scene
        / "formal_proxygs_native_40k_20260913"
        / "point_cloud"
        / "iteration_40000"
        / "point_cloud.ply"
    )
    source = source_identity(ply)
    positions = load_positions(ply)
    index = AnchorPointIndex.build(
        positions,
        scene=scene,
        leaf_capacity=leaf_capacity,
        max_depth=max_depth,
        source=source,
    )
    built_checks = validate_layout(index)
    path = ROOT / "indices" / scene / "anchor_point_bvh.npz"
    metadata = index.save(path)
    loaded = AnchorPointIndex.load(path, scene=scene, source=source)
    loaded_checks = validate_layout(loaded)
    save_load_parity = all(
        np.array_equal(index.layout()[key], loaded.layout()[key])
        for key in index.layout()
    )
    if not save_load_parity or not np.array_equal(positions, loaded.positions):
        raise RuntimeError(f"{scene}: saved topology differs after load")
    record = {
        "schema": "proxygs_step5_anchor_index_build_v1",
        "status": "pass",
        "scene": scene,
        "source": source,
        "index": file_identity(path),
        "anchor_count": index.anchor_count,
        "node_count": index.node_count,
        "leaf_capacity": leaf_capacity,
        "max_depth": max_depth,
        "build_ms": index.build_ms,
        "load_ms": loaded.load_ms,
        "built_checks": built_checks,
        "loaded_checks": loaded_checks,
        "save_load_parity": save_load_parity,
        "metadata": metadata,
    }
    atomic_json(ROOT / "indices" / scene / "index_build.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=(*SCENES, "all"), default="all")
    parser.add_argument("--leaf-capacity", type=int, default=4096)
    parser.add_argument("--max-depth", type=int, default=32)
    args = parser.parse_args()
    scenes = SCENES if args.scene == "all" else (args.scene,)
    records = [build_scene(scene, args.leaf_capacity, args.max_depth) for scene in scenes]
    report = {
        "schema": "proxygs_step5_anchor_index_build_matrix_v1",
        "status": "pass",
        "scene_count": len(records),
        "anchor_count": sum(record["anchor_count"] for record in records),
        "leaf_capacity": args.leaf_capacity,
        "max_depth": args.max_depth,
        "records": records,
    }
    atomic_json(ROOT / "review" / "anchor_index_build_matrix.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

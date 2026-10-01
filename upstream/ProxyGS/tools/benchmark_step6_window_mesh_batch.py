#!/usr/bin/env python3
"""Tune fixed-window Mesh batch workers after, and without changing, selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np

from gdmgs.mesh_index import MeshIndex


ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step6_high_overlap_cpu_index_20260915")
STEP5 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step5_cpu_anchor_index_g2_20260915")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def unpack_triangles(payload: Path) -> np.ndarray:
    with np.load(payload, allow_pickle=False) as data:
        count = int(data["triangle_universe_count"].item())
        bitmap = np.ascontiguousarray(data["triangle_bitmap"], dtype=np.uint8)
    return np.flatnonzero(
        np.unpackbits(bitmap, count=count, bitorder="little")
    ).astype(np.int64, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="amsterdam")
    parser.add_argument("--mesh-index", type=Path, required=True)
    parser.add_argument("--mesh-token", required=True)
    parser.add_argument("--step5-run", type=Path, required=True)
    parser.add_argument("--camera-records", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    selection_path = ROOT / "review" / "speed_selected_high_overlap_windows.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "frozen" or selection.get("selection_performed_once") is not True:
        raise RuntimeError("windows must be frozen before window-only tuning")
    matches = [record for record in selection["windows"] if record["scene"] == args.scene]
    if len(matches) != 1:
        raise RuntimeError("development scene has no unique frozen window")
    window = matches[0]
    camera_records = json.loads(args.camera_records.read_text())
    by_camera = {record["camera"]: record for record in camera_records}
    selected_records = [by_camera[camera] for camera in window["camera_ids"]]
    domains = [
        {
            "w2c": np.ascontiguousarray(record["w2c"], dtype=np.float64),
            "angular_domain": tuple(record["angular_domain"]),
            "near": record["near"],
            "far": record["far"],
            "camera_id": record["camera"],
        }
        for record in selected_records
    ]
    oracle = [
        unpack_triangles(args.step5_run / "id_payload" / f"{camera}.npz")
        for camera in window["camera_ids"]
    ]
    index = MeshIndex.load(args.mesh_index, mesh_token=args.mesh_token)
    results = []
    for workers in args.workers:
        elapsed_samples = []
        exact = True
        for _ in range(args.repeat):
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                queries = list(
                    pool.map(
                        lambda domain: index.query(
                            domain, backend="optimized_bvh", threads=1
                        ),
                        domains,
                    )
                )
            elapsed_samples.append((time.perf_counter() - start) * 1000.0)
            exact = exact and all(
                np.array_equal(query.triangle_ids, expected)
                for query, expected in zip(queries, oracle)
            )
        results.append(
            {
                "workers": workers,
                "repeat": args.repeat,
                "elapsed_samples_ms": elapsed_samples,
                "elapsed_mean_ms": sum(elapsed_samples) / len(elapsed_samples),
                "amortized_mean_ms": sum(elapsed_samples) / len(elapsed_samples) / len(domains),
                "all_triangle_ids_exact": exact,
            }
        )
    passing = [record for record in results if record["all_triangle_ids_exact"]]
    if len(passing) != len(results):
        raise RuntimeError("a window batch candidate changed triangle IDs")
    winner = min(passing, key=lambda record: (record["elapsed_mean_ms"], record["workers"]))
    manifest = {
        "schema": "proxygs_step6_window_mesh_batch_tuning_v1",
        "status": "frozen_before_window_matrix",
        "selection_identity": {
            "path": str(selection_path),
            "bytes": selection_path.stat().st_size,
            "mtime_ns": selection_path.stat().st_mtime_ns,
        },
        "selection_changed": False,
        "development_scene": args.scene,
        "window_start_index": window["start_index"],
        "camera_count": len(domains),
        "inner_query_threads": 1,
        "results": results,
        "selected_workers": winner["workers"],
        "selected_elapsed_mean_ms": winner["elapsed_mean_ms"],
        "selected_amortized_mean_ms": winner["amortized_mean_ms"],
        "qualification_may_measure_but_not_retune": True,
    }
    atomic_json(ROOT / "manifests" / "window_mesh_batch_tuning_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

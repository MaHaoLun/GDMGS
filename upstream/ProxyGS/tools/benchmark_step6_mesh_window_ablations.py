#!/usr/bin/env python3
"""Benchmark exact Mesh query variants on the frozen Step 6 windows.

The benchmark keeps index build/load outside timed regions, rotates variant
order between repeats, and validates every result with a reversible original-
row bitmap.  Shorter temporal windows are centered, nested subsets of the
already-frozen 32-frame windows; they are never reselected from timing data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from gdmgs.mesh_index import CameraDomain, MeshIndex


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
VARIANTS = {
    "brute_batch": "brute_force",
    "legacy_bvh_batch": "bvh",
    "optimized_bvh_batch": "optimized_bvh",
}
WINDOW_RUN_ID = "formal_step6_j3_window_v1_20260915"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def centered_window(values: list[Any], length: int) -> list[Any]:
    if length <= 0 or length > len(values):
        raise ValueError("window length must be in [1, frozen length]")
    start = (len(values) - length) // 2
    return values[start : start + length]


def bitmap(ids: np.ndarray, count: int) -> np.ndarray:
    bits = np.zeros(count, dtype=np.uint8)
    bits[ids] = 1
    return np.packbits(bits, bitorder="little")


def run_batch(
    index: MeshIndex,
    domains: list[CameraDomain],
    backend: str,
    workers: int,
) -> tuple[list[Any], float]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(workers, len(domains))) as pool:
        results = list(
            pool.map(
                lambda domain: index.query(domain, backend=backend, threads=1),
                domains,
            )
        )
    return results, (time.perf_counter() - start) * 1000.0


def summarize_results(results: list[Any]) -> dict[str, Any]:
    return {
        "native_elapsed_sum_ms": sum(result.elapsed_ms for result in results),
        "tested_triangles": sum(result.counters["tested_triangles"] for result in results),
        "visited_nodes": sum(result.counters["visited_nodes"] for result in results),
        "returned_triangles": sum(result.counters["returned_triangles"] for result in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step6-root", type=Path, required=True)
    parser.add_argument("--step4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--lengths", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.repeat < 1 or args.warmup < 0:
        raise ValueError("workers/repeat must be positive and warmup nonnegative")
    lengths = sorted(set(args.lengths))
    if not lengths or lengths[0] < 1 or lengths[-1] > 32:
        raise ValueError("temporal lengths must be within the frozen 32-frame window")
    variants = {name: VARIANTS[name] for name in args.variants}

    report: dict[str, Any] = {
        "schema": "proxygs_step6_mesh_window_ablation_v1",
        "status": "running",
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "workers": args.workers,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "window_policy": (
            "centered nested subsets of each frozen 32-frame window; no timing-based reselection"
        ),
        "timing_scope": "paired whole-window wall elapsed; Mesh index build/load excluded",
        "scenes": [],
        "failures": [],
    }

    for scene_index, scene in enumerate(args.scenes):
        run = (
            args.step6_root
            / "runs"
            / "windows"
            / scene
            / WINDOW_RUN_ID
        )
        camera_path = run / "camera_domain_records.json"
        index_path = args.step4_root / "indices" / scene / "mesh_bvh.npz"
        raw_records = json.loads(camera_path.read_text())
        if len(raw_records) != 32:
            raise RuntimeError(f"{scene}: expected the frozen 32 camera records")
        domains = [
            CameraDomain.parse(
                {
                    "w2c": np.ascontiguousarray(record["w2c"], dtype=np.float64),
                    "angular_domain": tuple(record["angular_domain"]),
                    "near": record["near"],
                    "far": record["far"],
                    "camera_id": record["camera"],
                }
            )
            for record in raw_records
        ]
        index = MeshIndex.load(index_path)
        scene_report: dict[str, Any] = {
            "scene": scene,
            "mesh_index": identity(index_path),
            "camera_records": identity(camera_path),
            "mesh_triangles": len(index.triangles),
            "index_build_ms_excluded": index.build_ms,
            "index_load_ms_excluded": index.load_ms,
            "lengths": [],
        }
        for _ in range(args.warmup):
            for backend in variants.values():
                index.query(domains[len(domains) // 2], backend=backend, threads=1)

        for length_index, length in enumerate(lengths):
            selected = centered_window(domains, length)
            by_variant: dict[str, dict[str, Any]] = {
                name: {
                    "backend": backend,
                    "wall_elapsed_samples_ms": [],
                    "native_elapsed_sum_samples_ms": [],
                    "tested_triangles": None,
                    "visited_nodes": None,
                    "returned_triangles": None,
                    "exact_original_row_ids": True,
                }
                for name, backend in variants.items()
            }
            oracle_bitmaps: list[np.ndarray] | None = None
            oracle_counts: list[int] | None = None
            for repeat_index in range(args.repeat):
                names = list(variants)
                shift = (scene_index + length_index + repeat_index) % len(names)
                names = names[shift:] + names[:shift]
                for name in names:
                    results, wall_ms = run_batch(
                        index, selected, variants[name], args.workers
                    )
                    counters = summarize_results(results)
                    record = by_variant[name]
                    record["wall_elapsed_samples_ms"].append(wall_ms)
                    record["native_elapsed_sum_samples_ms"].append(
                        counters["native_elapsed_sum_ms"]
                    )
                    for key in ("tested_triangles", "visited_nodes", "returned_triangles"):
                        if record[key] is None:
                            record[key] = counters[key]
                        elif record[key] != counters[key]:
                            raise RuntimeError(f"{scene}/{length}/{name}: counter drift")
                    current_counts = [len(result.triangle_ids) for result in results]
                    current_bitmaps = [
                        bitmap(result.triangle_ids, len(index.triangles)) for result in results
                    ]
                    if oracle_bitmaps is None:
                        oracle_bitmaps = current_bitmaps
                        oracle_counts = current_counts
                    else:
                        exact = current_counts == oracle_counts and all(
                            np.array_equal(actual, expected)
                            for actual, expected in zip(current_bitmaps, oracle_bitmaps)
                        )
                        record["exact_original_row_ids"] &= exact
                        if not exact:
                            report["failures"].append(
                                f"{scene}/{length}/{name}: triangle IDs differ"
                            )
                    del results, current_bitmaps
                    gc.collect()
            for record in by_variant.values():
                samples = record["wall_elapsed_samples_ms"]
                record["wall_elapsed_mean_ms"] = sum(samples) / len(samples)
                record["amortized_wall_mean_ms"] = record["wall_elapsed_mean_ms"] / length
                record["tested_fraction"] = record["tested_triangles"] / (
                    len(index.triangles) * length
                )
                record["returned_fraction"] = record["returned_triangles"] / (
                    len(index.triangles) * length
                )
            scene_report["lengths"].append(
                {"frame_count": length, "variants": by_variant}
            )
        report["scenes"].append(scene_report)
        del index
        gc.collect()
        atomic_json(args.output, report)

    report["status"] = "pass" if not report["failures"] else "failed"
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failures"]:
        raise RuntimeError(report["failures"])


if __name__ == "__main__":
    main()

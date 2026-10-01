#!/usr/bin/env python3
"""Freeze the fastest exact Mesh window profile from qualification repeats."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from gdmgs.mesh_index import CameraDomain, MeshIndex


SCENES = (
    "amsterdam", "barcelona", "bilbao", "chicago",
    "hollywood", "pompidou", "quebec", "rome",
)
WINDOW_RUN_ID = "formal_step6_j3_window_v1_20260915"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_domains(path: Path) -> list[CameraDomain]:
    records = json.loads(path.read_text())
    if len(records) != 32:
        raise RuntimeError("profile tuning requires the complete frozen 32-frame window")
    return [
        CameraDomain.parse(
            {
                "w2c": np.ascontiguousarray(record["w2c"], dtype=np.float64),
                "angular_domain": tuple(record["angular_domain"]),
                "near": record["near"],
                "far": record["far"],
                "camera_id": record["camera"],
            }
        )
        for record in records
    ]


def run_batch(index: MeshIndex, domains: list[CameraDomain], backend: str, workers: int):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(
                lambda domain: index.query(domain, backend=backend, threads=1),
                domains,
            )
        )
    return results, (time.perf_counter() - start) * 1000.0


def packed_ids(results: list[object], triangle_count: int) -> list[np.ndarray]:
    packed = []
    for result in results:
        bits = np.zeros(triangle_count, dtype=np.uint8)
        bits[result.triangle_ids] = 1
        packed.append(np.packbits(bits, bitorder="little"))
    return packed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step6-root", type=Path, required=True)
    parser.add_argument("--step4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--minimum-index-speedup", type=float, default=1.20)
    args = parser.parse_args()
    workers = sorted(set(args.workers))
    if not workers or workers[0] < 1 or workers[-1] > 32:
        raise ValueError("worker candidates must be in [1, 32]")
    if args.minimum_index_speedup <= 1.0:
        raise ValueError("minimum index speedup must be greater than one")
    candidates = [
        {"name": f"{label}_w{worker}", "backend": backend, "workers": worker}
        for backend, label in (("brute_force", "brute"), ("optimized_bvh", "optimized_bvh"))
        for worker in workers
    ]
    report = {
        "schema": "proxygs_step6_mesh_window_profile_tuning_v1",
        "status": "running",
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "frame_count": 256,
        "window_count": 8,
        "candidate_order": "rotated by scene and repeat",
        "timing_scope": "whole-window wall elapsed; index build/load excluded",
        "minimum_index_speedup_to_retain": args.minimum_index_speedup,
        "candidates": candidates,
        "scenes": [],
        "failures": [],
    }
    retained = {}
    for scene_index, scene in enumerate(SCENES):
        run = args.step6_root / "runs" / "windows" / scene / WINDOW_RUN_ID
        domains = load_domains(run / "camera_domain_records.json")
        index = MeshIndex.load(args.step4_root / "indices" / scene / "mesh_bvh.npz")
        for _ in range(args.warmup):
            index.query(domains[16], backend="brute_force", threads=1)
            index.query(domains[16], backend="optimized_bvh", threads=1)
        records = {
            candidate["name"]: {
                **candidate,
                "wall_elapsed_samples_ms": [],
                "exact_original_row_ids": True,
            }
            for candidate in candidates
        }
        oracle = None
        oracle_counts = None
        for repeat_index in range(args.repeat):
            order = list(candidates)
            shift = (scene_index + repeat_index) % len(order)
            order = order[shift:] + order[:shift]
            for candidate in order:
                results, elapsed_ms = run_batch(
                    index, domains, candidate["backend"], candidate["workers"]
                )
                record = records[candidate["name"]]
                record["wall_elapsed_samples_ms"].append(elapsed_ms)
                counts = [len(result.triangle_ids) for result in results]
                packed = packed_ids(results, len(index.triangles))
                if oracle is None:
                    oracle = packed
                    oracle_counts = counts
                else:
                    exact = counts == oracle_counts and all(
                        np.array_equal(actual, expected)
                        for actual, expected in zip(packed, oracle)
                    )
                    record["exact_original_row_ids"] &= exact
                    if not exact:
                        report["failures"].append(
                            f"{scene}/{candidate['name']}: triangle IDs differ"
                        )
                del results, packed
                gc.collect()
        for record in records.values():
            samples = np.asarray(record["wall_elapsed_samples_ms"], dtype=np.float64)
            record["wall_elapsed_mean_ms"] = float(samples.mean())
            record["wall_elapsed_median_ms"] = float(np.median(samples))
            record["wall_elapsed_p95_ms"] = float(np.percentile(samples, 95))
            record["amortized_mean_ms"] = float(samples.mean() / 32)
        best_brute = min(
            (record for record in records.values() if record["backend"] == "brute_force"),
            key=lambda record: (record["wall_elapsed_mean_ms"], record["name"]),
        )
        best_index = min(
            (record for record in records.values() if record["backend"] == "optimized_bvh"),
            key=lambda record: (record["wall_elapsed_mean_ms"], record["name"]),
        )
        qualified_index_speedup = (
            best_brute["wall_elapsed_mean_ms"] / best_index["wall_elapsed_mean_ms"]
        )
        # A small timing win on these memory-heavy queries is not stable enough
        # to freeze.  Retain the index only when it clears the preregistered
        # margin; otherwise use the exact brute bypass for this scene window.
        winner = (
            best_index
            if qualified_index_speedup >= args.minimum_index_speedup
            else best_brute
        )
        retained[scene] = {
            "backend": winner["backend"],
            "workers": winner["workers"],
            "qualification_candidate": winner["name"],
            "qualification_mean_ms": winner["wall_elapsed_mean_ms"],
            "best_index_speedup_over_brute": qualified_index_speedup,
            "minimum_index_speedup_to_retain": args.minimum_index_speedup,
        }
        report["scenes"].append(
            {"scene": scene, "results": list(records.values()), "winner": retained[scene]}
        )
        del index
        gc.collect()
        atomic_json(args.output, report)

    report["status"] = "pass" if not report["failures"] else "failed"
    atomic_json(args.output, report)
    profile = {
        "schema": "proxygs_step6_retained_mesh_window_profile_v1",
        "status": "frozen_from_qualification" if not report["failures"] else "failed",
        "source_qualification": str(args.output.resolve()),
        "selection_changed_windows": False,
        "window_count": 8,
        "frame_count": 256,
        "index_build_load_in_timing": False,
        "minimum_index_speedup_to_retain": args.minimum_index_speedup,
        "scene_profiles": retained,
    }
    atomic_json(args.profile_output, profile)
    print(json.dumps({"report": report, "profile": profile}, indent=2, sort_keys=True))
    if report["failures"]:
        raise RuntimeError(report["failures"])


if __name__ == "__main__":
    main()

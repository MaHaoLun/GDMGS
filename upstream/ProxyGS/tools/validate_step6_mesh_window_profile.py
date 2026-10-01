#!/usr/bin/env python3
"""Validate a qualification-frozen Mesh window profile on fresh repeats."""

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
        raise RuntimeError("formal validation requires the complete frozen window")
    return [
        CameraDomain.parse(
            {
                "w2c": np.ascontiguousarray(record["w2c"], dtype=np.float64),
                "angular_domain": tuple(record["angular_domain"]),
                "near": record["near"], "far": record["far"],
                "camera_id": record["camera"],
            }
        )
        for record in records
    ]


def run_batch(index: MeshIndex, domains: list[CameraDomain], backend: str, workers: int):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(
            lambda domain: index.query(domain, backend=backend, threads=1), domains
        ))
    return results, (time.perf_counter() - start) * 1000.0


def exact(left: list[object], right: list[object]) -> bool:
    return all(
        np.array_equal(a.triangle_ids, b.triangle_ids) for a, b in zip(left, right)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step6-root", type=Path, required=True)
    parser.add_argument("--step4-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if not (
        profile.get("status") == "frozen_from_qualification"
        and profile.get("window_count") == 8
        and profile.get("frame_count") == 256
        and set(profile.get("scene_profiles", {})) == set(SCENES)
    ):
        raise RuntimeError("retained profile is not a complete frozen qualification result")
    report = {
        "schema": "proxygs_step6_mesh_window_profile_formal_v1",
        "status": "running",
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "window_count": 8,
        "frame_count": 256,
        "profile": str(args.profile.resolve()),
        "profile_reselected": False,
        "timing_scope": "paired whole-window wall elapsed; index build/load excluded",
        "scenes": [],
        "failures": [],
    }
    totals = {"retained": 0.0, "brute_w32": 0.0, "legacy_bvh_w32": 0.0}
    for scene_index, scene in enumerate(SCENES):
        chosen = profile["scene_profiles"][scene]
        run = args.step6_root / "runs" / "windows" / scene / WINDOW_RUN_ID
        domains = load_domains(run / "camera_domain_records.json")
        index = MeshIndex.load(args.step4_root / "indices" / scene / "mesh_bvh.npz")
        candidates = {
            "retained": (chosen["backend"], int(chosen["workers"])),
            "brute_w32": ("brute_force", 32),
            "legacy_bvh_w32": ("bvh", 32),
        }
        for _ in range(args.warmup):
            for backend in {value[0] for value in candidates.values()}:
                index.query(domains[16], backend=backend, threads=1)
        samples = {name: [] for name in candidates}
        exact_flags = {name: True for name in candidates}
        for repeat_index in range(args.repeat):
            names = list(candidates)
            shift = (scene_index + repeat_index) % len(names)
            names = names[shift:] + names[:shift]
            oracle = None
            for name in names:
                backend, workers = candidates[name]
                results, elapsed_ms = run_batch(index, domains, backend, workers)
                samples[name].append(elapsed_ms)
                if oracle is None:
                    oracle = results
                else:
                    exact_flags[name] &= exact(results, oracle)
                    if not exact_flags[name]:
                        report["failures"].append(f"{scene}/{name}: triangle IDs differ")
                if name != names[0]:
                    del results
                gc.collect()
            del oracle
        means = {name: float(np.mean(values)) for name, values in samples.items()}
        for name in totals:
            totals[name] += means[name]
        report["scenes"].append(
            {
                "scene": scene,
                "retained_profile": chosen,
                "wall_elapsed_samples_ms": samples,
                "wall_elapsed_mean_ms": means,
                "exact_original_row_ids": exact_flags,
            }
        )
        del index
        gc.collect()
        atomic_json(args.output, report)
    report["totals_ms"] = totals
    report["speedups"] = {
        "brute_w32_to_retained": totals["brute_w32"] / totals["retained"],
        "legacy_bvh_w32_to_retained": totals["legacy_bvh_w32"] / totals["retained"],
    }
    if report["speedups"]["brute_w32_to_retained"] <= 1.0:
        report["failures"].append("retained profile did not beat paired brute_w32")
    report["status"] = "pass" if not report["failures"] else "failed"
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failures"]:
        raise RuntimeError(report["failures"])


if __name__ == "__main__":
    main()

"""Step 5 CPU reference/brute/index query helpers and parity contracts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gdmgs.anchor_index import AnchorPointIndex, AnchorPointQueryResult
from step4_runtime import DEPTH_MARGIN, MIN_CAMERA_Z, dense_anchor_filter_cpu


QUERY_PROFILE = "proxygs-pointwise-center-v1"


def run_anchor_queries(
    *,
    index: AnchorPointIndex,
    candidate_ids: np.ndarray,
    anchor_positions: np.ndarray,
    world_view_transform: np.ndarray,
    full_proj_transform: np.ndarray,
    indexed_depth: np.ndarray,
    camera: str,
    tree_first: bool,
) -> dict[str, Any]:
    """Run all three CPU modes against the same array objects."""
    reference_ids, reference_counters = dense_anchor_filter_cpu(
        candidate_ids,
        anchor_positions,
        world_view_transform,
        full_proj_transform,
        indexed_depth,
        margin=float(DEPTH_MARGIN),
    )
    modes = ("tree", "linear") if tree_first else ("linear", "tree")
    results: dict[str, AnchorPointQueryResult] = {}
    for mode in modes:
        results[mode] = index.query(
            candidate_ids,
            indexed_depth,
            world_view_transform,
            full_proj_transform,
            mode=mode,
            camera=camera,
            margin=float(DEPTH_MARGIN),
            minimum_camera_z=float(MIN_CAMERA_Z),
        )
    brute = results["linear"]
    tree = results["tree"]
    reference_brute_equal = np.array_equal(reference_ids, brute.selected_anchor_ids)
    brute_tree_equal = np.array_equal(brute.selected_anchor_ids, tree.selected_anchor_ids)
    brute_ranges_equal = np.array_equal(
        brute.expand_ranges(candidate_ids), brute.selected_anchor_ids
    )
    tree_ranges_equal = np.array_equal(
        tree.expand_ranges(candidate_ids), tree.selected_anchor_ids
    )
    if not reference_brute_equal or not brute_tree_equal:
        reference_only = np.setdiff1d(reference_ids, brute.selected_anchor_ids, assume_unique=True)
        brute_only = np.setdiff1d(brute.selected_anchor_ids, reference_ids, assume_unique=True)
        brute_only_vs_tree = np.setdiff1d(
            brute.selected_anchor_ids, tree.selected_anchor_ids, assume_unique=True
        )
        tree_only = np.setdiff1d(
            tree.selected_anchor_ids, brute.selected_anchor_ids, assume_unique=True
        )
        raise RuntimeError(
            f"{camera}: anchor query parity failed: "
            f"ref_only={reference_only[:20].tolist()} "
            f"brute_only={brute_only[:20].tolist()} "
            f"brute_vs_tree={brute_only_vs_tree[:20].tolist()} "
            f"tree_only={tree_only[:20].tolist()}"
        )
    if not brute_ranges_equal or not tree_ranges_equal:
        raise RuntimeError(f"{camera}: query ranges do not expand to selected IDs")
    return {
        "reference_ids": reference_ids,
        "reference_counters": reference_counters,
        "brute": brute,
        "tree": tree,
        "parity": {
            "reference_vs_brute_ids": reference_brute_equal,
            "brute_vs_index_ids": brute_tree_equal,
            "brute_ranges_expand": brute_ranges_equal,
            "index_ranges_expand": tree_ranges_equal,
            "original_order": bool(
                np.all(brute.selected_anchor_ids[1:] > brute.selected_anchor_ids[:-1])
                if len(brute.selected_anchor_ids) > 1
                else True
            ),
        },
    }


def query_record(result: AnchorPointQueryResult) -> dict[str, Any]:
    return {
        "mode": result.mode,
        "query_profile": result.query_profile,
        "range_space": result.range_space,
        "selected_count": int(len(result.selected_anchor_ids)),
        "raw_range_count": int(len(result.raw_ranges)),
        "formal_range_count": int(len(result.formal_ranges)),
        "counters": result.counters,
        "timings": result.timings,
    }


def load_step4_payload(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as record:
        if set(record.files) != {"candidate_ids", "selected_ids"}:
            raise ValueError("Step 4 ID payload fields differ from the frozen schema")
        candidates = np.ascontiguousarray(record["candidate_ids"], dtype=np.int64)
        selected = np.ascontiguousarray(record["selected_ids"], dtype=np.int64)
    return candidates, selected

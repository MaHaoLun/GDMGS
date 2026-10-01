#!/usr/bin/env python3
"""Freeze the Step 6 speed-selected high-overlap continuous windows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step6_high_overlap_cpu_index_20260915")
STEP5 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step5_cpu_anchor_index_g2_20260915")
RUN_ID = "formal_step6_j3_full_v1_20260915"
EXPECTED = {
    "amsterdam": 161,
    "barcelona": 160,
    "bilbao": 129,
    "chicago": 160,
    "hollywood": 125,
    "pompidou": 161,
    "quebec": 160,
    "rome": 158,
}
WINDOW_LENGTH = 32


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def harmonic(first: float, second: float) -> float:
    denominator = first + second
    return 2.0 * first * second / denominator if denominator else 0.0


def winner_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -record["index_speedup"],
        record["j3_full_index_elapsed_sum_ms"],
        -record["joint_overlap"],
        -record["minimum_transition_joint_overlap"],
        record["start_index"],
        record["start_camera"],
    )


def main() -> None:
    policy_path = ROOT / "manifests" / "window_selection_policy.json"
    audit_path = ROOT / "review" / "step6_input_audit.json"
    trace_path = STEP5 / "review" / "full_view_overlap_trace_manifest.json"
    policy = load(policy_path)
    audit = load(audit_path)
    trace = load(trace_path)
    if policy.get("status") != "frozen_before_j3_full" or audit.get("status") != "pass":
        raise RuntimeError("Step 6 policy/input audit is not frozen and passing")
    if trace.get("status") != "pass" or trace.get("window_selection_performed") is not False:
        raise RuntimeError("Step 5 trace is not the untouched full-view handoff")

    trace_by_scene = {scene["scene"]: scene for scene in trace["scenes"]}
    all_candidates = []
    winners = []
    full_reviews = []
    total_views = 0
    for scene, expected in EXPECTED.items():
        run = ROOT / "runs" / "formal" / scene / RUN_ID
        summary_path = run / "summary.json"
        per_view_path = run / "per_view.json"
        status_path = run / "status.json"
        summary = load(summary_path)
        records = load(per_view_path)
        status = load(status_path)
        if not (
            summary.get("state") == "complete"
            and status.get("state") == "complete"
            and summary.get("formal") is True
            and summary.get("qualification") is False
            and summary.get("view_count") == expected
            and len(records) == expected
            and all(summary.get("parity", {}).values())
        ):
            raise RuntimeError(f"{scene}: J3-Full formal run is incomplete or failed")
        total_views += len(records)
        full_reviews.append(
            {
                "scene": scene,
                "view_count": len(records),
                "summary": identity(summary_path),
                "per_view": identity(per_view_path),
                "timing_ms": summary["timing_ms"],
                "parity": summary["parity"],
            }
        )
        scene_trace = trace_by_scene[scene]
        cameras = scene_trace["camera_order"]
        transitions = scene_trace["adjacent_transitions"]
        if [record["camera"] for record in records] != cameras:
            raise RuntimeError(f"{scene}: Step 6 camera order differs from Step 5")
        candidates = []
        for start in range(expected - WINDOW_LENGTH + 1):
            stop = start + WINDOW_LENGTH
            window_transitions = transitions[start : stop - 1]
            anchor_values = np.asarray(
                [item["selected_anchor"]["jaccard"] for item in window_transitions],
                dtype=np.float64,
            )
            mesh_values = np.asarray(
                [item["mesh"]["jaccard"] for item in window_transitions],
                dtype=np.float64,
            )
            transition_joint = np.asarray(
                [harmonic(float(a), float(m)) for a, m in zip(anchor_values, mesh_values)],
                dtype=np.float64,
            )
            anchor_median = float(np.median(anchor_values))
            mesh_median = float(np.median(mesh_values))
            joint_overlap = harmonic(anchor_median, mesh_median)
            window_records = records[start:stop]
            j0_sum = sum(
                record["mesh"]["j0"]["mean_ms"] + record["anchor"]["j0"]["mean_ms"]
                for record in window_records
            )
            j3_sum = sum(
                record["mesh"]["j1"]["mean_ms"] + record["anchor"]["j2"]["mean_ms"]
                for record in window_records
            )
            sample_speedups = []
            repeat_count = len(window_records[0]["mesh"]["j0"]["samples_ms"])
            for repeat in range(repeat_count):
                sample_j0 = sum(
                    record["mesh"]["j0"]["samples_ms"][repeat]
                    + record["anchor"]["j0"]["samples_ms"][repeat]
                    for record in window_records
                )
                sample_j3 = sum(
                    record["mesh"]["j1"]["samples_ms"][repeat]
                    + record["anchor"]["j2"]["samples_ms"][repeat]
                    for record in window_records
                )
                sample_speedups.append(sample_j0 / sample_j3)
            candidates.append(
                {
                    "scene": scene,
                    "start_index": start,
                    "end_index_inclusive": stop - 1,
                    "start_camera": cameras[start],
                    "end_camera": cameras[stop - 1],
                    "frame_count": WINDOW_LENGTH,
                    "transition_count": WINDOW_LENGTH - 1,
                    "anchor_median_jaccard": anchor_median,
                    "mesh_median_jaccard": mesh_median,
                    "joint_overlap": joint_overlap,
                    "minimum_transition_joint_overlap": float(np.min(transition_joint)),
                    "j0_index_elapsed_sum_ms": j0_sum,
                    "j3_full_index_elapsed_sum_ms": j3_sum,
                    "index_speedup": j0_sum / j3_sum,
                    "repeat_index_speedups": sample_speedups,
                    "camera_ids": cameras[start:stop],
                }
            )
        threshold = float(
            np.quantile(
                np.asarray([record["joint_overlap"] for record in candidates], dtype=np.float64),
                0.75,
                method="linear",
            )
        )
        for record in candidates:
            record["eligibility_threshold"] = threshold
            record["eligible"] = record["joint_overlap"] >= threshold
        for rank, record in enumerate(sorted(candidates, key=winner_key), start=1):
            record["all_candidate_rank"] = rank
        eligible = [record for record in candidates if record["eligible"]]
        if not eligible:
            raise RuntimeError(f"{scene}: inclusive top-quartile eligibility is empty")
        eligible.sort(key=winner_key)
        for rank, record in enumerate(eligible, start=1):
            record["eligible_rank"] = rank
        winner = dict(eligible[0])
        repeat_winners = []
        for repeat in range(len(winner["repeat_index_speedups"])):
            ranked = sorted(
                eligible,
                key=lambda record: (
                    -record["repeat_index_speedups"][repeat],
                    record["j3_full_index_elapsed_sum_ms"],
                    -record["joint_overlap"],
                    -record["minimum_transition_joint_overlap"],
                    record["start_index"],
                    record["start_camera"],
                ),
            )
            repeat_winners.append(ranked[0]["start_index"])
        winner["repeat_winner_start_indices"] = repeat_winners
        winner["mean_winner_repeat_stability"] = sum(
            start == winner["start_index"] for start in repeat_winners
        ) / len(repeat_winners)
        winner["selection_signal_boundary"] = "Step5 overlap plus frozen J0/J3-Full CPU index timing only"
        winners.append(winner)
        all_candidates.extend(candidates)

    if total_views != 1214 or len(winners) != 8:
        raise RuntimeError("Step 6 full-view denominator or winner count drifted")
    ranking = {
        "schema": "proxygs_step6_all_candidate_window_ranking_v1",
        "status": "pass",
        "policy": identity(policy_path),
        "source_trace": identity(trace_path),
        "candidate_count": len(all_candidates),
        "scene_count": 8,
        "window_length": WINDOW_LENGTH,
        "candidates": all_candidates,
    }
    ranking_path = ROOT / "review" / "all_candidate_window_ranking.json"
    atomic_json(ranking_path, ranking)
    selection = {
        "schema": "proxygs_step6_speed_selected_high_overlap_windows_v1",
        "status": "frozen",
        "label": "speed-selected high-reuse continuous-view workload",
        "selection_performed_once": True,
        "cache_signals_used": False,
        "scene_count": 8,
        "window_count": 8,
        "frame_count": 256,
        "transition_count": 248,
        "source_full_view_count": total_views,
        "policy": identity(policy_path),
        "all_candidate_ranking": identity(ranking_path),
        "j3_full_reviews": full_reviews,
        "windows": winners,
    }
    selection_path = ROOT / "review" / "speed_selected_high_overlap_windows.json"
    atomic_json(selection_path, selection)
    atomic_json(
        ROOT / "review" / "j3_full_review.json",
        {
            "schema": "proxygs_step6_j3_full_review_v1",
            "status": "pass",
            "scene_count": 8,
            "view_count": total_views,
            "correctness_failure_count": 0,
            "reduced_count": False,
            "normal_path_fallback": False,
            "scene_reviews": full_reviews,
            "window_selection": identity(selection_path),
        },
    )
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

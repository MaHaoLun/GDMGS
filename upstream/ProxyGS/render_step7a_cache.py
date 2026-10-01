"""Run Step 7A full-bundle cache exactness on one frozen Step 6 window."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from gaussian_renderer import generate_neural_gaussians
from gaussian_renderer.gdmgs_gsplat_backend import render_gdmgs_backend
from gaussian_renderer.raster_batch import NeuralGaussianBatch, batch_from_proxygs_decode
from gdmgs.cache import CacheIdentity, FullBundleCacheCore
from render_gdmgs_backend import (
    BACKEND_ID,
    _environment_record,
    _file_identity,
    _frozen_camera_names,
    _load_cfg,
    _new_model,
    _ordered_views,
)


PROTOCOL_ID = "proxygs-step7a-full-bundle-cache-v1"
STEP6_ROOT = Path(
    "/ssddata/lun/gdmgs_artifacts/proxygs_step6_high_overlap_cpu_index_20260915"
)
STEP6_DIAG_ROOT = Path(
    "/ssddata/lun/gdmgs_artifacts/proxygs_step6_bvh_diagnosis_20260915"
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def percentile(values: Iterable[float], q: float) -> float | None:
    materialized = list(values)
    return float(np.percentile(materialized, q)) if materialized else None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def validate_step6_inputs(args: argparse.Namespace) -> Tuple[dict, dict, dict, dict]:
    review = load_json(args.step6_review)
    diagnosis = load_json(args.step6_bvh_review)
    trace = load_json(args.cache_trace)
    retained = load_json(args.retained_profile)
    if not (
        review.get("status") == "pass"
        and review.get("scene_count") == 8
        and review.get("full_view_count") == 1214
        and review.get("window_count") == 8
        and review.get("window_frame_count") == 256
        and review.get("window_transition_count") == 248
        and review.get("correctness_failure_count") == 0
        and review.get("reduced_count") is False
        and review.get("window_reselection_after_j3_window") is False
    ):
        raise ValueError("Step 6 final review is not the frozen passing 8-window result")
    if not (
        diagnosis.get("status") == "pass"
        and diagnosis.get("window_count") == 8
        and diagnosis.get("frame_count") == 256
        and diagnosis.get("correctness_failure_count") == 0
        and diagnosis.get("reduced_count") is False
        and diagnosis.get("window_reselection") is False
    ):
        raise ValueError("Step 6 Retained-v2 diagnosis review is not passing")
    if not (
        trace.get("status") == "pass"
        and trace.get("window_count") == 8
        and trace.get("frame_count") == 256
        and trace.get("transition_count") == 248
    ):
        raise ValueError("Step 6 cache trace is not the frozen 256-frame workload")
    if not (
        retained.get("status") == "frozen_from_qualification"
        and retained.get("window_count") == 8
        and retained.get("frame_count") == 256
        and retained.get("selection_changed_windows") is False
        and retained.get("minimum_index_speedup_to_retain") == 1.2
    ):
        raise ValueError("Retained-v2 conservative profile is not frozen")
    if args.scene not in retained.get("scene_profiles", {}):
        raise ValueError("Retained-v2 profile has no entry for this scene")
    scene_traces = [item for item in trace["scenes"] if item.get("scene") == args.scene]
    if len(scene_traces) != 1:
        raise ValueError("cache trace must contain exactly one window for the scene")
    return review, diagnosis, scene_traces[0], retained


def unpack_selected_ids(payload_record: dict, anchor_count: int) -> np.ndarray:
    path = Path(payload_record["path"])
    current = _file_identity(path)
    if (
        current["bytes"] != payload_record["bytes"]
        or current["mtime_ns"] != payload_record["mtime_ns"]
    ):
        raise ValueError(f"Step 6 ID payload identity changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "anchor_universe_count",
            "triangle_universe_count",
            "candidate_bitmap",
            "selected_bitmap",
            "triangle_bitmap",
        }
        if set(data.files) != required:
            raise ValueError("Step 6 ID payload schema drifted")
        if int(data["anchor_universe_count"].item()) != anchor_count:
            raise ValueError("Step 6 anchor universe differs from the loaded model")
        selected = np.flatnonzero(
            np.unpackbits(
                data["selected_bitmap"], count=anchor_count, bitorder="little"
            )
        ).astype(np.int64, copy=False)
    if len(selected) != payload_record["selected_count"]:
        raise ValueError("Step 6 selected count differs from reversible bitmap")
    return selected


def decode_batch(
    view: Any,
    model: Any,
    anchor_ids: torch.Tensor,
    level_ids: torch.Tensor,
) -> NeuralGaussianBatch:
    decoded = generate_neural_gaussians(
        view,
        model,
        is_training=False,
        anchor_indices=anchor_ids,
    )
    return batch_from_proxygs_decode(
        anchor_ids=anchor_ids,
        decoded=decoded,
        n_offsets=model.n_offsets,
        request_level_ids=level_ids,
    )


def timed_decode(
    view: Any,
    model: Any,
    anchor_ids: torch.Tensor,
    level_ids: torch.Tensor,
) -> Tuple[NeuralGaussianBatch, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch = decode_batch(view, model, anchor_ids, level_ids)
    torch.cuda.synchronize()
    return batch, (time.perf_counter() - start) * 1000.0


def timed_render(
    view: Any,
    batch: NeuralGaussianBatch,
    background: torch.Tensor,
    *,
    warmup: int,
    repeat: int,
) -> Tuple[torch.Tensor, List[float]]:
    for _ in range(warmup):
        output = render_gdmgs_backend(view, batch, background, "RGB")
        del output
    torch.cuda.synchronize()
    samples = []
    output = None
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = render_gdmgs_backend(view, batch, background, "RGB")
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    assert output is not None
    return torch.clamp(output["render"], 0.0, 1.0), samples


def batch_exact(reference: NeuralGaussianBatch, candidate: NeuralGaussianBatch) -> dict:
    payload = {
        name: torch.equal(getattr(reference, name), getattr(candidate, name))
        for name in ("xyz", "color", "opacity", "scaling", "rotation")
    }
    metadata = {
        name: torch.equal(
            getattr(reference.bundle_metadata, name),
            getattr(candidate.bundle_metadata, name),
        )
        for name in (
            "request_anchor_ids",
            "request_level_ids",
            "row_owner_ids",
            "row_owner_levels",
            "row_offset_slots",
            "counts",
            "offsets",
        )
    }
    return {
        "payload": payload,
        "metadata": metadata,
        "payload_exact": all(payload.values()),
        "metadata_exact": all(metadata.values()),
    }


def capacity_sweep(rows_per_frame: List[int], n_offsets: int) -> List[dict]:
    points = []
    for q in (50, 75, 90, 95, 99, 100):
        rows = max(n_offsets, int(np.ceil(np.percentile(rows_per_frame, q))))
        points.append(
            {
                "label": f"p{q}",
                "capacity_rows": rows,
                "float32_payload_bytes": rows * 56,
                "frames_fitting": sum(value <= rows for value in rows_per_frame),
                "frame_count": len(rows_per_frame),
            }
        )
    deduplicated = []
    seen = set()
    for point in points:
        if point["capacity_rows"] not in seen:
            deduplicated.append(point)
            seen.add(point["capacity_rows"])
    return deduplicated


def render_scene(args: argparse.Namespace) -> dict:
    if not args.formal or args.max_views is not None:
        raise ValueError("formal Step 7A requires the complete frozen 32-frame window")
    if args.iteration != 40000 or args.width != 1600 or args.height != 900:
        raise ValueError("Step 7A freezes iteration=40000 and 1600x900 rendering")
    if args.repeat < 1 or args.lookup_repeat < 1 or args.warmup < 0:
        raise ValueError("repeat counts must be positive and warmup nonnegative")
    review, diagnosis, scene_trace, retained = validate_step6_inputs(args)

    output_dir = (args.output_root / args.scene / args.run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(map(str, sys.argv)) + "\n")
    atomic_json(output_dir / "status.json", {"state": "starting", "scene": args.scene})

    model_path = args.model_path.resolve()
    cfg = _load_cfg(model_path)
    if Path(cfg.source_path).resolve() != args.source_path.resolve():
        raise ValueError("model cfg source path differs from the frozen CLI path")
    if cfg.n_offsets != 10:
        raise ValueError(f"Step 7A expects the trained n_offsets=10, found {cfg.n_offsets}")
    cfg.source_path = str(args.source_path.resolve())
    cfg.data_device = "cpu"
    model = _new_model(cfg)
    from scene import Scene

    scene = Scene(
        cfg,
        model,
        load_iteration=args.iteration,
        shuffle=False,
        resolution_scales=cfg.resolution_scales,
    )
    model.eval()
    frozen_names = _frozen_camera_names(model_path)
    views = _ordered_views(scene, frozen_names)
    if len(views) != args.expected_views:
        raise ValueError(f"expected {args.expected_views} views, found {len(views)}")
    camera_ids = scene_trace["camera_ids"]
    by_name = {view.image_name: view for view in views}
    if len(camera_ids) != 32 or any(name not in by_name for name in camera_ids):
        raise ValueError("Step 6 cache trace camera IDs do not map to this model")
    views = [by_name[name] for name in camera_ids]
    if any((int(view.image_width), int(view.image_height)) != (args.width, args.height) for view in views):
        raise ValueError("every Step 7A camera must be 1600x900")
    if len(scene_trace["id_payloads"]) != len(views):
        raise ValueError("cache trace payload count does not match frozen cameras")

    anchor_levels = model.get_level.detach().view(-1).to(dtype=torch.long).contiguous()
    anchor_count = int(model.get_anchor.shape[0])
    if anchor_levels.shape != (anchor_count,):
        raise ValueError("model level table does not contain one LoD per anchor row")
    capacity_rows = args.capacity_rows or anchor_count * model.n_offsets
    cache_identity = CacheIdentity(
        scene=args.scene,
        model=f"{model_path}:iteration-{args.iteration}",
        backend=BACKEND_ID,
        anchor_table=str(model_path / "point_cloud" / "iteration_40000" / "point_cloud.ply"),
        trace=str(args.cache_trace.resolve()),
    )
    cache = FullBundleCacheCore(
        identity=cache_identity,
        anchor_levels=anchor_levels,
        capacity_rows=capacity_rows,
        n_offsets=model.n_offsets,
    )
    background_values = [1.0, 1.0, 1.0] if cfg.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")

    contract = {
        "schema": "proxygs_step7a_scene_contract_v1",
        "protocol_id": PROTOCOL_ID,
        "state": "frozen_before_run",
        "scene": args.scene,
        "formal": True,
        "camera_count": 32,
        "camera_ids": camera_ids,
        "image_size": [args.width, args.height],
        "iteration": args.iteration,
        "n_offsets": model.n_offsets,
        "capacity_rows": capacity_rows,
        "payload_dtype": "float32",
        "row_payload_bytes": 56,
        "cache_key": ["level_id", "final_ply_anchor_row_id"],
        "current_generation_read_only": True,
        "cross_pose_reuse": False,
        "future_residency": False,
        "schedule": False,
        "window_reselection": False,
        "mesh_profile_binding": retained["scene_profiles"][args.scene],
        "environment": _environment_record(),
        "inputs": {
            "step6_review": _file_identity(args.step6_review.resolve()),
            "step6_bvh_review": _file_identity(args.step6_bvh_review.resolve()),
            "cache_trace": _file_identity(args.cache_trace.resolve()),
            "retained_profile": _file_identity(args.retained_profile.resolve()),
            "cfg_args": _file_identity(model_path / "cfg_args"),
            "cameras_json": _file_identity(model_path / "cameras.json"),
            "point_cloud": _file_identity(
                model_path / "point_cloud" / "iteration_40000" / "point_cloud.ply"
            ),
            "opacity_mlp": _file_identity(
                model_path / "point_cloud" / "iteration_40000" / "opacity_mlp.pt"
            ),
            "cov_mlp": _file_identity(
                model_path / "point_cloud" / "iteration_40000" / "cov_mlp.pt"
            ),
            "color_mlp": _file_identity(
                model_path / "point_cloud" / "iteration_40000" / "color_mlp.pt"
            ),
        },
        "upstream_summary": {
            "step6_status": review["status"],
            "step6_diagnosis_status": diagnosis["status"],
            "window_count": 8,
            "formal_frame_count": 256,
            "transition_count": 248,
        },
    }
    atomic_json(output_dir / "run_contract.json", contract)
    atomic_json(
        output_dir / "status.json",
        {"state": "running", "scene": args.scene, "expected_views": 32},
    )

    records: List[Dict[str, Any]] = []
    for frame_index, (view, payload_record) in enumerate(
        zip(views, scene_trace["id_payloads"])
    ):
        if payload_record["path"].split("/")[-1] != f"{view.image_name}.npz":
            raise ValueError("cache trace payload order differs from frozen camera order")
        selected = unpack_selected_ids(payload_record, anchor_count)
        anchor_ids_cpu = torch.from_numpy(selected.copy())
        torch.cuda.synchronize()
        request_setup_start = time.perf_counter()
        anchor_ids = anchor_ids_cpu.to(device="cuda", dtype=torch.long)
        level_ids = anchor_levels.index_select(0, anchor_ids)
        torch.cuda.synchronize()
        request_setup_ms = (time.perf_counter() - request_setup_start) * 1000.0
        model.set_anchor_mask(view.camera_center, args.iteration, view.resolution_scale)

        torch.cuda.reset_peak_memory_stats()
        fresh_batch, fresh_decode_ms = timed_decode(view, model, anchor_ids, level_ids)
        fresh_image, fresh_render_samples = timed_render(
            view,
            fresh_batch,
            background,
            warmup=args.warmup if frame_index == 0 else 0,
            repeat=args.repeat,
        )
        fresh_peak = torch.cuda.max_memory_allocated()
        metadata = fresh_batch.bundle_metadata
        counts = metadata.counts
        if bool((counts > model.n_offsets).any()):
            raise RuntimeError("fresh decoder produced a bundle longer than n_offsets")

        cache.reset()
        torch.cuda.synchronize()
        generation_build_before_bytes = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        build_start = time.perf_counter()
        same_pose_generation = cache.build_generation(
            fresh_batch,
            request_level_ids=level_ids,
            source_camera_id=str(view.image_name),
        )
        torch.cuda.synchronize()
        generation_build_ms = (time.perf_counter() - build_start) * 1000.0
        generation_build_after_bytes = int(torch.cuda.memory_allocated())
        generation_build_peak_bytes = int(torch.cuda.max_memory_allocated())
        generation_build_scratch_bytes = max(
            0,
            generation_build_peak_bytes
            - max(generation_build_before_bytes, generation_build_after_bytes),
        )
        cache.publish_generation(same_pose_generation)
        same_pose_samples = []
        same_pose_memory_samples = []
        same_pose_result = None
        same_pose_decode_calls = []

        def decode_same_pose_zero_rows(
            miss_ids: torch.Tensor,
            miss_levels: torch.Tensor,
        ) -> NeuralGaussianBatch:
            same_pose_decode_calls.append(miss_ids.detach().cpu().tolist())
            return decode_batch(view, model, miss_ids, miss_levels)

        for _ in range(args.lookup_repeat):
            if same_pose_result is not None:
                del same_pose_result
                same_pose_result = None
            torch.cuda.synchronize()
            resolution_before_bytes = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            same_pose_result = cache.resolve(
                anchor_ids=anchor_ids,
                level_ids=level_ids,
                decode_misses=decode_same_pose_zero_rows,
                profile=True,
            )
            same_pose_samples.append(same_pose_result.timings_ms)
            resolution_after_bytes = int(torch.cuda.memory_allocated())
            resolution_peak_bytes = int(torch.cuda.max_memory_allocated())
            same_pose_memory_samples.append(
                {
                    "before_bytes": resolution_before_bytes,
                    "after_bytes": resolution_after_bytes,
                    "peak_bytes": resolution_peak_bytes,
                    "scratch_bytes": max(
                        0,
                        resolution_peak_bytes
                        - max(resolution_before_bytes, resolution_after_bytes),
                    ),
                }
            )
        assert same_pose_result is not None
        torch.cuda.synchronize()
        same_pose_render_before_bytes = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        same_pose_image, same_pose_render_samples = timed_render(
            view,
            same_pose_result.batch,
            background,
            warmup=0,
            repeat=args.repeat,
        )
        same_pose_render_after_bytes = int(torch.cuda.memory_allocated())
        same_pose_peak = int(torch.cuda.max_memory_allocated())
        same_pose_render_scratch_bytes = max(
            0,
            same_pose_peak
            - max(same_pose_render_before_bytes, same_pose_render_after_bytes),
        )
        same_pose_exact = batch_exact(fresh_batch, same_pose_result.batch)
        same_pose_render_exact = torch.equal(fresh_image, same_pose_image)
        same_pose_max_delta = float(torch.max(torch.abs(fresh_image - same_pose_image)))
        same_pose_expected_hit_mask = counts > 0
        same_pose_hit_mask_exact = torch.equal(
            same_pose_result.hit_mask,
            same_pose_expected_hit_mask,
        )
        expected_same_pose_decode_calls = (
            args.lookup_repeat if bool((counts == 0).any()) else 0
        )

        hit_request_mask = torch.arange(
            anchor_ids.numel(), device=anchor_ids.device
        ) % 2 == 0
        seed_ids = anchor_ids[hit_request_mask]
        seed_levels = level_ids[hit_request_mask]
        seed_batch, seed_decode_ms = timed_decode(view, model, seed_ids, seed_levels)
        cache.reset()
        mixed_generation = cache.build_generation(
            seed_batch,
            request_level_ids=seed_levels,
            source_camera_id=str(view.image_name),
        )
        cache.publish_generation(mixed_generation)

        decode_calls = []

        def decode_misses(
            miss_ids: torch.Tensor,
            miss_levels: torch.Tensor,
        ) -> NeuralGaussianBatch:
            decode_calls.append(miss_ids.detach().cpu().tolist())
            return decode_batch(view, model, miss_ids, miss_levels)

        mixed_result = cache.resolve(
            anchor_ids=anchor_ids,
            level_ids=level_ids,
            decode_misses=decode_misses,
            profile=True,
        )
        mixed_image, mixed_render_samples = timed_render(
            view,
            mixed_result.batch,
            background,
            warmup=0,
            repeat=args.repeat,
        )
        mixed_exact = batch_exact(fresh_batch, mixed_result.batch)
        mixed_render_exact = torch.equal(fresh_image, mixed_image)
        mixed_max_delta = float(torch.max(torch.abs(fresh_image - mixed_image)))
        expected_hit_mask = hit_request_mask & (counts > 0)
        controlled_hit_mask_exact = torch.equal(mixed_result.hit_mask, expected_hit_mask)

        frame_pass = (
            same_pose_exact["payload_exact"]
            and same_pose_exact["metadata_exact"]
            and same_pose_render_exact
            and same_pose_hit_mask_exact
            and len(same_pose_decode_calls) == expected_same_pose_decode_calls
            and mixed_exact["payload_exact"]
            and mixed_exact["metadata_exact"]
            and mixed_render_exact
            and controlled_hit_mask_exact
            and len(decode_calls) == (1 if bool(mixed_result.miss_mask.any()) else 0)
        )
        if not frame_pass:
            atomic_json(
                output_dir / "diagnostics" / f"{view.image_name}.json",
                {
                    "same_pose_exact": same_pose_exact,
                    "same_pose_render_exact": same_pose_render_exact,
                    "same_pose_max_delta": same_pose_max_delta,
                    "same_pose_hit_mask_exact": same_pose_hit_mask_exact,
                    "same_pose_decode_call_count": len(same_pose_decode_calls),
                    "mixed_exact": mixed_exact,
                    "mixed_render_exact": mixed_render_exact,
                    "mixed_max_delta": mixed_max_delta,
                    "controlled_hit_mask_exact": controlled_hit_mask_exact,
                    "decode_call_count": len(decode_calls),
                },
            )
            raise RuntimeError(f"{view.image_name}: Step 7A mechanical exactness failed")

        histogram = torch.bincount(counts, minlength=model.n_offsets + 1)
        record = {
            "window_index": frame_index,
            "camera": str(view.image_name),
            "selected_anchor_count": int(anchor_ids.numel()),
            "decoded_row_count": int(fresh_batch.xyz.shape[0]),
            "bundle_length_histogram": {
                str(index): int(value)
                for index, value in enumerate(histogram.detach().cpu().tolist())
            },
            "nonempty_bundle_count": int((counts > 0).sum()),
            "zero_bundle_count": int((counts == 0).sum()),
            "request_setup_ms": request_setup_ms,
            "fresh": {
                "decode_ms": fresh_decode_ms,
                "render_samples_ms": fresh_render_samples,
                "render_mean_ms": mean(fresh_render_samples),
                "peak_allocated_bytes": int(fresh_peak),
                "cache_stage_frame_ms": request_setup_ms
                + fresh_decode_ms
                + mean(fresh_render_samples),
            },
            "same_pose_seeded": {
                "generation_id": same_pose_result.generation_id,
                "generation_build_ms": generation_build_ms,
                "generation_build_memory": {
                    "before_bytes": generation_build_before_bytes,
                    "after_bytes": generation_build_after_bytes,
                    "peak_bytes": generation_build_peak_bytes,
                    "scratch_bytes": generation_build_scratch_bytes,
                },
                "generation": {
                    "descriptors": same_pose_generation.descriptor_count,
                    "rows": same_pose_generation.occupied_rows,
                    "capacity_rows": same_pose_generation.capacity_rows,
                    "raw_memory_bytes": same_pose_generation.memory_bytes(),
                },
                "hit_anchors": int(same_pose_result.hit_mask.sum()),
                "zero_row_miss_anchors": int(same_pose_result.miss_mask.sum()),
                "zero_row_decode_calls": len(same_pose_decode_calls),
                "hit_mask_exact": same_pose_hit_mask_exact,
                "resolution_samples_ms": same_pose_samples,
                "resolution_memory_samples": same_pose_memory_samples,
                "render_samples_ms": same_pose_render_samples,
                "render_mean_ms": mean(same_pose_render_samples),
                "peak_allocated_bytes": int(same_pose_peak),
                "render_memory": {
                    "before_bytes": same_pose_render_before_bytes,
                    "after_bytes": same_pose_render_after_bytes,
                    "peak_bytes": same_pose_peak,
                    "scratch_bytes": same_pose_render_scratch_bytes,
                },
                "payload_and_metadata": same_pose_exact,
                "render_exact": same_pose_render_exact,
                "render_max_abs_delta": same_pose_max_delta,
                "cache_stage_frame_samples_ms": [
                    request_setup_ms + sample["total"] + mean(same_pose_render_samples)
                    for sample in same_pose_samples
                ],
            },
            "controlled_mixed": {
                "seed_decode_ms": seed_decode_ms,
                "seed_descriptor_count": mixed_generation.descriptor_count,
                "seed_rows": mixed_generation.occupied_rows,
                "hit_anchors": int(mixed_result.hit_mask.sum()),
                "miss_anchors": int(mixed_result.miss_mask.sum()),
                "hit_rows": mixed_result.hit_rows,
                "fresh_rows": mixed_result.fresh_rows,
                "miss_decode_calls": len(decode_calls),
                "resolution_ms": mixed_result.timings_ms,
                "render_samples_ms": mixed_render_samples,
                "render_mean_ms": mean(mixed_render_samples),
                "payload_and_metadata": mixed_exact,
                "expected_hit_mask_exact": controlled_hit_mask_exact,
                "render_exact": mixed_render_exact,
                "render_max_abs_delta": mixed_max_delta,
                "cache_stage_frame_ms": request_setup_ms
                + mixed_result.timings_ms["total"]
                + mean(mixed_render_samples),
            },
            "pass": frame_pass,
        }
        records.append(record)
        atomic_json(output_dir / "per_view.json", records)
        atomic_json(
            output_dir / "status.json",
            {
                "state": "running",
                "scene": args.scene,
                "completed_views": len(records),
                "expected_views": 32,
                "last_camera": str(view.image_name),
            },
        )
        del (
            fresh_batch,
            fresh_image,
            same_pose_result,
            same_pose_image,
            seed_batch,
            mixed_result,
            mixed_image,
        )
        cache.reset()
        torch.cuda.empty_cache()

    rows_per_frame = [record["decoded_row_count"] for record in records]
    summary = {
        "schema": "proxygs_step7a_scene_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "state": "complete",
        "scene": args.scene,
        "formal": True,
        "view_count": len(records),
        "camera_ids": camera_ids,
        "n_offsets": model.n_offsets,
        "correctness_failures": [],
        "same_pose_payload_exact": all(
            record["same_pose_seeded"]["payload_and_metadata"]["payload_exact"]
            for record in records
        ),
        "same_pose_metadata_exact": all(
            record["same_pose_seeded"]["payload_and_metadata"]["metadata_exact"]
            for record in records
        ),
        "same_pose_render_exact": all(
            record["same_pose_seeded"]["render_exact"] for record in records
        ),
        "controlled_mixed_payload_exact": all(
            record["controlled_mixed"]["payload_and_metadata"]["payload_exact"]
            for record in records
        ),
        "controlled_mixed_metadata_exact": all(
            record["controlled_mixed"]["payload_and_metadata"]["metadata_exact"]
            for record in records
        ),
        "controlled_mixed_render_exact": all(
            record["controlled_mixed"]["render_exact"] for record in records
        ),
        "zero_row_not_admitted": all(
            record["same_pose_seeded"]["generation"]["descriptors"]
            == record["nonempty_bundle_count"]
            for record in records
        ),
        "bundle_statistics": {
            "selected_anchors": sum(record["selected_anchor_count"] for record in records),
            "decoded_rows": sum(rows_per_frame),
            "zero_bundles": sum(record["zero_bundle_count"] for record in records),
            "nonempty_bundles": sum(record["nonempty_bundle_count"] for record in records),
            "rows_per_frame_p50": percentile(rows_per_frame, 50),
            "rows_per_frame_p95": percentile(rows_per_frame, 95),
            "rows_per_frame_p99": percentile(rows_per_frame, 99),
            "rows_per_frame_max": max(rows_per_frame),
        },
        "capacity_sweep": capacity_sweep(rows_per_frame, model.n_offsets),
        "timing_ms": {
            "fresh_decode_mean": mean(record["fresh"]["decode_ms"] for record in records),
            "request_setup_mean": mean(record["request_setup_ms"] for record in records),
            "fresh_render_mean": mean(record["fresh"]["render_mean_ms"] for record in records),
            "fresh_cache_stage_frame_sum": sum(
                record["fresh"]["cache_stage_frame_ms"] for record in records
            ),
            "generation_build_mean": mean(
                record["same_pose_seeded"]["generation_build_ms"] for record in records
            ),
            "same_pose_resolution_mean": mean(
                sample["total"]
                for record in records
                for sample in record["same_pose_seeded"]["resolution_samples_ms"]
            ),
            "same_pose_render_mean": mean(
                record["same_pose_seeded"]["render_mean_ms"] for record in records
            ),
            "same_pose_cache_stage_frame_sum": sum(
                mean(record["same_pose_seeded"]["cache_stage_frame_samples_ms"])
                for record in records
            ),
            "mixed_resolution_mean": mean(
                record["controlled_mixed"]["resolution_ms"]["total"]
                for record in records
            ),
            "mixed_render_mean": mean(
                record["controlled_mixed"]["render_mean_ms"] for record in records
            ),
        },
        "memory": {
            "same_pose_generation_raw_bytes_max": max(
                record["same_pose_seeded"]["generation"]["raw_memory_bytes"]["total"]
                for record in records
            ),
            "fresh_peak_allocated_bytes_max": max(
                record["fresh"]["peak_allocated_bytes"] for record in records
            ),
            "same_pose_peak_allocated_bytes_max": max(
                record["same_pose_seeded"]["peak_allocated_bytes"] for record in records
            ),
        },
        "input_identities_after": {
            name: _file_identity(Path(value["path"]))
            for name, value in contract["inputs"].items()
        },
        "completion_scope": (
            "same-pose and controlled mixed full-bundle cache mechanical exactness only; "
            "no cross-pose validity, Future Residency, schedule, or end-to-end FPS claim"
        ),
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        output_dir / "status.json",
        {
            "state": "complete",
            "scene": args.scene,
            "completed_views": len(records),
            "expected_views": 32,
            "correctness_failures": [],
        },
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--scene", required=True)
    result.add_argument("--source-path", type=Path, required=True)
    result.add_argument("--model-path", type=Path, required=True)
    result.add_argument(
        "--step6-review",
        type=Path,
        default=STEP6_ROOT / "review" / "final_step6_review.json",
    )
    result.add_argument(
        "--step6-bvh-review",
        type=Path,
        default=STEP6_DIAG_ROOT / "review" / "final_bvh_diagnosis_review.json",
    )
    result.add_argument(
        "--cache-trace",
        type=Path,
        default=STEP6_ROOT / "review" / "cache_trace_manifest.json",
    )
    result.add_argument(
        "--retained-profile",
        type=Path,
        default=STEP6_DIAG_ROOT / "qualification" / "retained_profile_conservative.json",
    )
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--expected-views", type=int, required=True)
    result.add_argument("--iteration", type=int, default=40000)
    result.add_argument("--width", type=int, default=1600)
    result.add_argument("--height", type=int, default=900)
    result.add_argument("--capacity-rows", type=int, default=0)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--repeat", type=int, default=1)
    result.add_argument("--lookup-repeat", type=int, default=3)
    result.add_argument("--max-views", type=int)
    result.add_argument("--formal", action="store_true")
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    try:
        report = render_scene(parsed)
    except Exception as error:
        failure = parsed.output_root / parsed.scene / parsed.run_id
        atomic_json(
            failure / "status.json",
            {"state": "failed", "scene": parsed.scene, "error": repr(error)},
        )
        raise
    print(json.dumps(report, indent=2, sort_keys=True))

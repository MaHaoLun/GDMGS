"""Step 7B unconditional cross-pose bundle fidelity and cost calibration."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from gaussian_renderer import generate_neural_gaussians
from gaussian_renderer.gdmgs_gsplat_backend import render_gdmgs_backend
from gaussian_renderer.raster_batch import NeuralGaussianBatch, batch_from_proxygs_decode
from gdmgs.cache import CacheGeneration, CacheIdentity, FullBundleCacheCore
from render_gdmgs_backend import (
    BACKEND_ID,
    _environment_record,
    _file_identity,
    _frozen_camera_names,
    _load_cfg,
    _new_model,
    _ordered_views,
)
from utils.image_utils import psnr
from utils.loss_utils import ssim


PROTOCOL_ID = "proxygs-step7b-unconditional-cross-pose-v1"
STEP6 = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step6_high_overlap_cpu_index_20260915")
STEP6_DIAG = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step6_bvh_diagnosis_20260915")
STEP7A = Path("/ssddata/lun/gdmgs_artifacts/proxygs_step7a_full_bundle_cache_20260916")
FORMAL_CAPACITY_ROWS = 6_826_846
FORMAL_LAGS = (1, 2, 4, 8)
THRESHOLDS = {
    "direct_psnr_min_db": 40.0,
    "direct_ssim_min": 0.99,
    "direct_lpips_max": 0.01,
    "direct_rgb_mae_max": 0.005,
    "direct_rgb_p99_max": 0.02,
    "direct_rgb_max_max": 0.10,
    "gt_psnr_drop_max_db": 0.10,
    "gt_ssim_drop_max": 0.002,
    "gt_lpips_increase_max": 0.005,
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def mean(values: Iterable[float]) -> Optional[float]:
    items = list(values)
    return sum(items) / len(items) if items else None


def validate_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    step6_review = load(args.step6_review)
    step6_trace = load(args.cache_trace)
    retained = load(args.retained_profile)
    step7a_review = load(args.step7a_review)
    step7a_exact = load(args.step7a_exactness)
    step7a_capacity = load(args.step7a_capacity)
    if not (
        step6_review.get("status") == "pass"
        and step6_review.get("scene_count") == 8
        and step6_review.get("full_view_count") == 1214
        and step6_review.get("window_count") == 8
        and step6_review.get("window_frame_count") == 256
        and step6_review.get("window_transition_count") == 248
        and step6_review.get("correctness_failure_count") == 0
        and step6_review.get("reduced_count") is False
        and step6_review.get("window_reselection_after_j3_window") is False
    ):
        raise ValueError("Step 6 final review is not the frozen passing result")
    if not (
        step6_trace.get("status") == "pass"
        and step6_trace.get("window_count") == 8
        and step6_trace.get("frame_count") == 256
        and step6_trace.get("transition_count") == 248
    ):
        raise ValueError("Step 6 cache trace is not complete")
    if not (
        retained.get("status") == "frozen_from_qualification"
        and retained.get("selection_changed_windows") is False
        and retained.get("window_count") == 8
        and retained.get("frame_count") == 256
    ):
        raise ValueError("Retained-v2 profile is not frozen")
    if not (
        step7a_review.get("status") == "pass"
        and step7a_review.get("scene_count") == 8
        and step7a_review.get("frame_count") == 256
        and step7a_review.get("correctness_failure_count") == 0
        and step7a_review.get("failures") == []
    ):
        raise ValueError("Step 7A final review is not passing")
    if not (
        step7a_exact.get("status") == "pass"
        and step7a_exact.get("same_pose_payload_exact") is True
        and step7a_exact.get("same_pose_metadata_exact") is True
        and step7a_exact.get("same_pose_render_exact") is True
        and step7a_exact.get("maximum_render_abs_delta") == 0.0
    ):
        raise ValueError("Step 7A same-pose exactness is not passing")
    if int(step7a_capacity["rows_per_frame"]["maximum"]) != FORMAL_CAPACITY_ROWS:
        raise ValueError("Step 7A p100 row capacity differs from the preregistered Step 7B capacity")
    scenes = [item for item in step6_trace["scenes"] if item.get("scene") == args.scene]
    if len(scenes) != 1:
        raise ValueError("Step 6 cache trace must contain exactly one scene window")
    if args.scene not in retained["scene_profiles"]:
        raise ValueError("Retained-v2 profile has no scene entry")
    return scenes[0], retained["scene_profiles"][args.scene], step7a_review, step7a_capacity


def unpack_selected_ids(payload_record: dict, anchor_count: int) -> np.ndarray:
    path = Path(payload_record["path"])
    current = _file_identity(path)
    if current["bytes"] != payload_record["bytes"] or current["mtime_ns"] != payload_record["mtime_ns"]:
        raise ValueError(f"Step 6 payload identity changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "anchor_universe_count",
            "triangle_universe_count",
            "candidate_bitmap",
            "selected_bitmap",
            "triangle_bitmap",
        }
        if set(data.files) != required:
            raise ValueError("Step 6 payload schema drifted")
        if int(data["anchor_universe_count"].item()) != anchor_count:
            raise ValueError("Step 6 anchor universe differs from the loaded model")
        selected = np.flatnonzero(
            np.unpackbits(data["selected_bitmap"], count=anchor_count, bitorder="little")
        ).astype(np.int64, copy=False)
    if len(selected) != int(payload_record["selected_count"]):
        raise ValueError("Step 6 selected count differs from the reversible bitmap")
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
) -> tuple[NeuralGaussianBatch, float]:
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
) -> tuple[torch.Tensor, List[float]]:
    for _ in range(warmup):
        output = render_gdmgs_backend(view, batch, background, "RGB")
        del output
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


def image_metrics(
    reuse: torch.Tensor,
    fresh: torch.Tensor,
    gt: torch.Tensor,
    lpips_model: Any,
    fresh_gt: Optional[dict] = None,
) -> dict:
    direct_delta = torch.abs(reuse - fresh)
    direct_psnr = float(psnr(reuse, fresh).mean())
    if direct_delta.max() == 0 and not math.isfinite(direct_psnr):
        direct_psnr = 120.0
    direct = {
        "psnr": direct_psnr,
        "ssim": float(ssim(reuse.unsqueeze(0), fresh.unsqueeze(0))),
        "lpips": float(
            lpips_model(reuse.unsqueeze(0), fresh.unsqueeze(0), normalize=True).mean()
        ),
        "rgb_mae": float(direct_delta.mean()),
        "rgb_p99_abs": float(torch.quantile(direct_delta.flatten(), 0.99)),
        "rgb_max_abs": float(direct_delta.max()),
    }
    if fresh_gt is None:
        fresh_gt_psnr = float(psnr(fresh, gt).mean())
        if torch.equal(fresh, gt) and not math.isfinite(fresh_gt_psnr):
            fresh_gt_psnr = 120.0
        fresh_gt = {
            "psnr": fresh_gt_psnr,
            "ssim": float(ssim(fresh.unsqueeze(0), gt.unsqueeze(0))),
            "lpips": float(
                lpips_model(fresh.unsqueeze(0), gt.unsqueeze(0), normalize=True).mean()
            ),
        }
    reuse_gt_psnr = float(psnr(reuse, gt).mean())
    if torch.equal(reuse, gt) and not math.isfinite(reuse_gt_psnr):
        reuse_gt_psnr = 120.0
    reuse_gt = {
        "psnr": reuse_gt_psnr,
        "ssim": float(ssim(reuse.unsqueeze(0), gt.unsqueeze(0))),
        "lpips": float(
            lpips_model(reuse.unsqueeze(0), gt.unsqueeze(0), normalize=True).mean()
        ),
    }
    deltas = {
        "psnr": reuse_gt["psnr"] - fresh_gt["psnr"],
        "ssim": reuse_gt["ssim"] - fresh_gt["ssim"],
        "lpips": reuse_gt["lpips"] - fresh_gt["lpips"],
    }
    finite = all(
        math.isfinite(value)
        for group in (direct, fresh_gt, reuse_gt, deltas)
        for value in group.values()
    )
    gates = {
        "finite": finite,
        "direct_psnr": direct["psnr"] >= THRESHOLDS["direct_psnr_min_db"],
        "direct_ssim": direct["ssim"] >= THRESHOLDS["direct_ssim_min"],
        "direct_lpips": direct["lpips"] <= THRESHOLDS["direct_lpips_max"],
        "direct_rgb_mae": direct["rgb_mae"] <= THRESHOLDS["direct_rgb_mae_max"],
        "direct_rgb_p99": direct["rgb_p99_abs"] <= THRESHOLDS["direct_rgb_p99_max"],
        "direct_rgb_max": direct["rgb_max_abs"] <= THRESHOLDS["direct_rgb_max_max"],
        "gt_psnr": deltas["psnr"] >= -THRESHOLDS["gt_psnr_drop_max_db"],
        "gt_ssim": deltas["ssim"] >= -THRESHOLDS["gt_ssim_drop_max"],
        "gt_lpips": deltas["lpips"] <= THRESHOLDS["gt_lpips_increase_max"],
    }
    return {
        "direct": direct,
        "fresh_gt": fresh_gt,
        "reuse_gt": reuse_gt,
        "reuse_minus_fresh_gt": deltas,
        "gates": gates,
        "pass": all(gates.values()),
    }


def batch_exact(reference: NeuralGaussianBatch, candidate: NeuralGaussianBatch) -> bool:
    fields = ("xyz", "color", "opacity", "scaling", "rotation")
    if not all(torch.equal(getattr(reference, name), getattr(candidate, name)) for name in fields):
        return False
    metadata_fields = (
        "request_anchor_ids",
        "request_level_ids",
        "row_owner_ids",
        "row_owner_levels",
        "row_offset_slots",
        "counts",
        "offsets",
    )
    return all(
        torch.equal(
            getattr(reference.bundle_metadata, name),
            getattr(candidate.bundle_metadata, name),
        )
        for name in metadata_fields
    )


def pose_delta(source: Any, target: Any) -> dict:
    translation = float(torch.linalg.vector_norm(target.camera_center - source.camera_center))
    source_rotation = source.world_view_transform[:3, :3]
    target_rotation = target.world_view_transform[:3, :3]
    relative = source_rotation.transpose(0, 1) @ target_rotation
    cosine = torch.clamp((torch.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    angle_degrees = float(torch.rad2deg(torch.acos(cosine)))
    return {"translation": translation, "rotation_degrees": angle_degrees}


def generation_lookup(
    generation: CacheGeneration,
    anchor_ids: torch.Tensor,
    level_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = level_ids * generation.anchor_count + anchor_ids
    positions = torch.full_like(keys, -1)
    if not generation.packed_keys.numel():
        return positions, positions >= 0
    insertion = torch.searchsorted(generation.packed_keys, keys)
    valid = insertion < generation.packed_keys.numel()
    safe = insertion.clamp(max=generation.packed_keys.numel() - 1)
    hit = valid & (generation.packed_keys.index_select(0, safe) == keys)
    positions[hit] = safe[hit]
    return positions, hit


def request_diagnostics(
    generation: CacheGeneration,
    target: NeuralGaussianBatch,
    level_ids: torch.Tensor,
    evidence_path: Path,
) -> dict:
    metadata = target.bundle_metadata
    assert metadata is not None
    anchor_ids = target.anchor_indices
    positions, hit_mask = generation_lookup(generation, anchor_ids, level_ids)
    target_keys = level_ids * generation.anchor_count + anchor_ids
    source_lengths = torch.zeros_like(anchor_ids)
    if bool(hit_mask.any()):
        source_lengths[hit_mask] = generation.row_lengths.index_select(0, positions[hit_mask])
    target_lengths = metadata.counts

    source_row_keys = (
        generation.packed_keys.repeat_interleave(generation.row_lengths) * generation.n_offsets
        + generation.row_offset_slots
    )
    target_row_keys = (
        target_keys.repeat_interleave(target_lengths) * generation.n_offsets
        + metadata.row_offset_slots
    )
    if source_row_keys.numel():
        insertion = torch.searchsorted(source_row_keys, target_row_keys)
        valid = insertion < source_row_keys.numel()
        safe = insertion.clamp(max=source_row_keys.numel() - 1)
        matched_rows = valid & (source_row_keys.index_select(0, safe) == target_row_keys)
    else:
        safe = torch.zeros_like(target_row_keys)
        matched_rows = torch.zeros_like(target_row_keys, dtype=torch.bool)
    request_for_target_row = torch.arange(
        anchor_ids.numel(), dtype=torch.long, device=anchor_ids.device
    ).repeat_interleave(target_lengths)
    matched_counts = torch.zeros_like(anchor_ids)
    if bool(matched_rows.any()):
        matched_counts.scatter_add_(
            0,
            request_for_target_row[matched_rows],
            torch.ones_like(request_for_target_row[matched_rows]),
        )

    attr_max = []
    attr_mean = []
    for name in ("xyz", "color", "opacity", "scaling", "rotation"):
        target_values = getattr(target, name)
        source_values = getattr(generation, name)
        row_max = torch.zeros(target_row_keys.numel(), dtype=torch.float32, device=anchor_ids.device)
        row_mean = torch.zeros_like(row_max)
        if bool(matched_rows.any()):
            diff = torch.abs(
                target_values[matched_rows]
                - source_values.index_select(0, safe[matched_rows])
            ).reshape(int(matched_rows.sum()), -1)
            row_max[matched_rows] = diff.amax(dim=1)
            row_mean[matched_rows] = diff.mean(dim=1)
        request_max = torch.zeros(anchor_ids.numel(), dtype=torch.float32, device=anchor_ids.device)
        request_sum = torch.zeros_like(request_max)
        if bool(matched_rows.any()):
            request_max.scatter_reduce_(
                0,
                request_for_target_row[matched_rows],
                row_max[matched_rows],
                reduce="amax",
                include_self=True,
            )
            request_sum.scatter_add_(
                0,
                request_for_target_row[matched_rows],
                row_mean[matched_rows],
            )
        request_mean = request_sum / matched_counts.clamp_min(1).to(torch.float32)
        request_max[matched_counts == 0] = torch.nan
        request_mean[matched_counts == 0] = torch.nan
        attr_max.append(request_max)
        attr_mean.append(request_mean)

    hit_ids = torch.nonzero(hit_mask, as_tuple=False).flatten()
    max_matrix = torch.stack(attr_max, dim=1).index_select(0, hit_ids)
    mean_matrix = torch.stack(attr_mean, dim=1).index_select(0, hit_ids)
    source_hit_lengths = source_lengths.index_select(0, hit_ids)
    target_hit_lengths = target_lengths.index_select(0, hit_ids)
    matched_hit_counts = matched_counts.index_select(0, hit_ids)
    atomic_npz(
        evidence_path,
        anchor_ids=anchor_ids.index_select(0, hit_ids).detach().cpu().numpy().astype(np.int64),
        level_ids=level_ids.index_select(0, hit_ids).detach().cpu().numpy().astype(np.int16),
        source_lengths=source_hit_lengths.detach().cpu().numpy().astype(np.uint8),
        target_lengths=target_hit_lengths.detach().cpu().numpy().astype(np.uint8),
        matched_offset_slots=matched_hit_counts.detach().cpu().numpy().astype(np.uint8),
        source_only_slots=(source_hit_lengths - matched_hit_counts).detach().cpu().numpy().astype(np.uint8),
        target_only_slots=(target_hit_lengths - matched_hit_counts).detach().cpu().numpy().astype(np.uint8),
        attribute_max_abs=max_matrix.detach().cpu().numpy().astype(np.float16),
        attribute_mean_abs=mean_matrix.detach().cpu().numpy().astype(np.float16),
    )
    finite_max = max_matrix[torch.isfinite(max_matrix)]
    return {
        "request_evidence": _file_identity(evidence_path),
        "request_evidence_schema": {
            "attributes": ["xyz", "color", "opacity", "scaling", "rotation"],
            "difference_dtype": "float16 storage from float32 computation",
            "identity_dtype": "int64 anchor ID plus int16 level ID",
        },
        "hit_anchors": int(hit_mask.sum()),
        "miss_anchors": int((~hit_mask).sum()),
        "source_rows_for_hits": int(source_hit_lengths.sum()),
        "target_rows_for_hits": int(target_hit_lengths.sum()),
        "matched_offset_slots": int(matched_hit_counts.sum()),
        "length_mismatch_anchors": int((source_hit_lengths != target_hit_lengths).sum()),
        "source_nonempty_target_zero": int((target_hit_lengths == 0).sum()),
        "source_only_slots": int((source_hit_lengths - matched_hit_counts).sum()),
        "target_only_slots": int((target_hit_lengths - matched_hit_counts).sum()),
        "attribute_max_abs_worst": float(finite_max.max()) if finite_max.numel() else None,
    }


def capacity_ceiling(
    generation: CacheGeneration,
    anchor_ids: torch.Tensor,
    level_ids: torch.Tensor,
    capacities: list[int],
) -> list[dict]:
    positions, hit = generation_lookup(generation, anchor_ids, level_ids)
    lengths = generation.row_lengths.index_select(0, positions[hit])
    sorted_lengths, _ = torch.sort(lengths)
    cumulative = sorted_lengths.cumsum(0)
    result = []
    for capacity in capacities:
        hits = int(torch.searchsorted(cumulative, cumulative.new_tensor(capacity), right=True))
        result.append(
            {
                "capacity_rows": capacity,
                "oracle_hit_anchor_ceiling": hits,
                "available_overlap_anchors": int(lengths.numel()),
                "rows_used": int(cumulative[hits - 1]) if hits else 0,
            }
        )
    return result


def evaluate_mode(
    *,
    mode: str,
    lag: int,
    source_index: int,
    target_index: int,
    source_view: Any,
    target_view: Any,
    generation: CacheGeneration,
    cache: FullBundleCacheCore,
    target_batch: NeuralGaussianBatch,
    target_ids: torch.Tensor,
    target_levels: torch.Tensor,
    target_fresh_image: torch.Tensor,
    target_gt: torch.Tensor,
    fresh_gt_metrics: dict,
    model: Any,
    background: torch.Tensor,
    lpips_model: Any,
    repeat: int,
    output_dir: Path,
    capacities: list[int],
) -> tuple[dict, torch.Tensor]:
    cache.publish_generation(generation)
    decode_calls = []

    def decode_misses(miss_ids: torch.Tensor, miss_levels: torch.Tensor) -> NeuralGaussianBatch:
        decode_calls.append(int(miss_ids.numel()))
        return decode_batch(target_view, model, miss_ids, miss_levels)

    torch.cuda.synchronize()
    before_bytes = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    resolution = cache.resolve(
        anchor_ids=target_ids,
        level_ids=target_levels,
        decode_misses=decode_misses,
        profile=True,
    )
    after_resolution_bytes = int(torch.cuda.memory_allocated())
    resolution_peak_bytes = int(torch.cuda.max_memory_allocated())
    image, render_samples = timed_render(
        target_view,
        resolution.batch,
        background,
        warmup=0,
        repeat=repeat,
    )
    final_peak_bytes = int(torch.cuda.max_memory_allocated())
    metrics = image_metrics(
        image,
        target_fresh_image,
        target_gt,
        lpips_model,
        fresh_gt=fresh_gt_metrics,
    )
    evidence_path = (
        output_dir
        / "request_evidence"
        / mode
        / f"{source_view.image_name}__{target_view.image_name}.npz"
    )
    diagnostics = request_diagnostics(
        generation,
        target_batch,
        target_levels,
        evidence_path,
    )
    positions, hit = generation_lookup(generation, target_ids, target_levels)
    if not torch.equal(hit, resolution.hit_mask):
        raise RuntimeError("diagnostic lookup differs from cache resolution")
    record = {
        "mode": mode,
        "lag": lag,
        "source_index": source_index,
        "target_index": target_index,
        "source_camera": str(source_view.image_name),
        "target_camera": str(target_view.image_name),
        "pose_delta": pose_delta(source_view, target_view),
        "source_generation": {
            "generation_id": generation.generation_id,
            "rows": generation.occupied_rows,
            "descriptors": generation.descriptor_count,
            "raw_memory_bytes": generation.memory_bytes(),
        },
        "resolution": {
            "hit_anchors": int(resolution.hit_mask.sum()),
            "miss_anchors": int(resolution.miss_mask.sum()),
            "hit_rows": resolution.hit_rows,
            "fresh_rows": resolution.fresh_rows,
            "decode_calls": decode_calls,
            "timings_ms": resolution.timings_ms,
            "memory": {
                "before_bytes": before_bytes,
                "after_resolution_bytes": after_resolution_bytes,
                "resolution_peak_bytes": resolution_peak_bytes,
                "render_peak_bytes": final_peak_bytes,
            },
        },
        "render_samples_ms": render_samples,
        "render_mean_ms": mean(render_samples),
        "fidelity": metrics,
        "request_diagnostics": diagnostics,
        "capacity_hit_ceiling": capacity_ceiling(
            generation,
            target_ids,
            target_levels,
            capacities,
        ),
    }
    return record, image


def render_scene(args: argparse.Namespace) -> dict:
    if not args.formal or args.max_views is not None:
        raise ValueError("formal Step 7B requires the complete frozen 32-frame window")
    if args.iteration != 40000 or args.width != 1600 or args.height != 900:
        raise ValueError("Step 7B freezes iteration=40000 and 1600x900")
    if args.repeat < 1 or args.warmup < 0:
        raise ValueError("repeat must be positive and warmup nonnegative")
    scene_trace, retained_profile, _, capacity_report = validate_inputs(args)

    output_dir = (args.output_root / args.scene / args.run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(map(str, sys.argv)) + "\n")
    atomic_json(output_dir / "status.json", {"state": "starting", "scene": args.scene})

    model_path = args.model_path.resolve()
    cfg = _load_cfg(model_path)
    if Path(cfg.source_path).resolve() != args.source_path.resolve():
        raise ValueError("model cfg source path differs from the frozen CLI path")
    if cfg.n_offsets != 10:
        raise ValueError("Step 7B requires the trained n_offsets=10")
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
    all_views = _ordered_views(scene, frozen_names)
    if len(all_views) != args.expected_views:
        raise ValueError(f"expected {args.expected_views} views, found {len(all_views)}")
    by_name = {view.image_name: view for view in all_views}
    camera_ids = scene_trace["camera_ids"]
    if len(camera_ids) != 32 or any(name not in by_name for name in camera_ids):
        raise ValueError("frozen Step 6 cameras do not map to this scene")
    views = [by_name[name] for name in camera_ids]
    if len(scene_trace["id_payloads"]) != 32:
        raise ValueError("Step 6 scene trace does not contain 32 payloads")

    anchor_levels = model.get_level.detach().view(-1).to(dtype=torch.long).contiguous()
    anchor_count = int(model.get_anchor.shape[0])
    identity = CacheIdentity(
        scene=args.scene,
        model=f"{model_path}:iteration-{args.iteration}",
        backend=BACKEND_ID,
        anchor_table=str(model_path / "point_cloud" / "iteration_40000" / "point_cloud.ply"),
        trace=str(args.cache_trace.resolve()),
    )
    builder = FullBundleCacheCore(
        identity=identity,
        anchor_levels=anchor_levels,
        capacity_rows=FORMAL_CAPACITY_ROWS,
        n_offsets=model.n_offsets,
    )
    same_pose_cache = FullBundleCacheCore(
        identity=identity,
        anchor_levels=anchor_levels,
        capacity_rows=FORMAL_CAPACITY_ROWS,
        n_offsets=model.n_offsets,
    )
    lag_caches = {
        lag: FullBundleCacheCore(
            identity=identity,
            anchor_levels=anchor_levels,
            capacity_rows=FORMAL_CAPACITY_ROWS,
            n_offsets=model.n_offsets,
        )
        for lag in FORMAL_LAGS
    }
    adversarial_cache = FullBundleCacheCore(
        identity=identity,
        anchor_levels=anchor_levels,
        capacity_rows=FORMAL_CAPACITY_ROWS,
        n_offsets=model.n_offsets,
    )
    capacities = sorted(
        {
            int(item["capacity_rows"])
            for item in capacity_report["capacity_sweep"]
        }
        | {FORMAL_CAPACITY_ROWS}
    )
    background_values = [1.0, 1.0, 1.0] if cfg.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")
    import lpips

    lpips_model = lpips.LPIPS(net="vgg").to(background.device).eval()

    inputs = {
        "step6_review": _file_identity(args.step6_review.resolve()),
        "cache_trace": _file_identity(args.cache_trace.resolve()),
        "retained_profile": _file_identity(args.retained_profile.resolve()),
        "step7a_review": _file_identity(args.step7a_review.resolve()),
        "step7a_exactness": _file_identity(args.step7a_exactness.resolve()),
        "step7a_capacity": _file_identity(args.step7a_capacity.resolve()),
        "adr_0013": _file_identity(args.adr_0013.resolve()),
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
    }
    contract = {
        "schema": "proxygs_step7b_scene_contract_v1",
        "protocol_id": PROTOCOL_ID,
        "state": "frozen_before_run",
        "scene": args.scene,
        "formal": True,
        "camera_count": 32,
        "camera_ids": camera_ids,
        "lags": list(FORMAL_LAGS),
        "pair_denominators": {"lag1": 31, "lag2": 30, "lag4": 28, "lag8": 24},
        "adversarial_pairs": 1,
        "hit_validity": "unconditional-complete-bundle-residency-per-ADR-0006-and-0013",
        "diagnostic_bins_do_not_gate_hits": True,
        "payload_dtype": "float32",
        "capacity_rows": FORMAL_CAPACITY_ROWS,
        "capacity_sweep_rows": capacities,
        "thresholds": THRESHOLDS,
        "future_residency": False,
        "schedule": False,
        "window_reselection": False,
        "retained_profile_binding": retained_profile,
        "environment": _environment_record(),
        "inputs": inputs,
    }
    atomic_json(output_dir / "run_contract.json", contract)
    atomic_json(
        output_dir / "status.json",
        {"state": "running", "scene": args.scene, "expected_views": 32},
    )

    generations: Dict[int, CacheGeneration] = {}
    records: List[dict] = []
    formal_pairs = 0
    failed_formal_pairs = 0
    same_pose_failures = 0
    for target_index, (view, payload_record) in enumerate(zip(views, scene_trace["id_payloads"])):
        if Path(payload_record["path"]).name != f"{view.image_name}.npz":
            raise ValueError("payload order differs from camera order")
        selected = unpack_selected_ids(payload_record, anchor_count)
        ids_cpu = torch.from_numpy(selected.copy())
        torch.cuda.synchronize()
        request_setup_start = time.perf_counter()
        anchor_ids = ids_cpu.to(device="cuda", dtype=torch.long)
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
            warmup=args.warmup if target_index == 0 else 0,
            repeat=args.repeat,
        )
        gt = torch.clamp(view.original_image.to(device=fresh_image.device), 0.0, 1.0)
        fresh_gt_metrics = {
            "psnr": float(psnr(fresh_image, gt).mean()),
            "ssim": float(ssim(fresh_image.unsqueeze(0), gt.unsqueeze(0))),
            "lpips": float(
                lpips_model(fresh_image.unsqueeze(0), gt.unsqueeze(0), normalize=True).mean()
            ),
        }
        torch.cuda.synchronize()
        build_before = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        build_start = time.perf_counter()
        generation = builder.build_generation(
            fresh_batch,
            request_level_ids=level_ids,
            source_camera_id=str(view.image_name),
            generation_id=target_index + 1,
        )
        torch.cuda.synchronize()
        build_ms = (time.perf_counter() - build_start) * 1000.0
        build_after = int(torch.cuda.memory_allocated())
        build_peak = int(torch.cuda.max_memory_allocated())

        same_pose_cache.publish_generation(generation)
        same_pose_result = same_pose_cache.resolve(
            anchor_ids=anchor_ids,
            level_ids=level_ids,
            decode_misses=lambda miss_ids, miss_levels: decode_batch(
                view, model, miss_ids, miss_levels
            ),
            profile=True,
        )
        same_pose_image, same_pose_render_samples = timed_render(
            view,
            same_pose_result.batch,
            background,
            warmup=0,
            repeat=args.repeat,
        )
        same_pose_exact = batch_exact(fresh_batch, same_pose_result.batch)
        same_pose_render_exact = torch.equal(fresh_image, same_pose_image)
        if not same_pose_exact or not same_pose_render_exact:
            same_pose_failures += 1

        frame_modes = {}
        for lag in FORMAL_LAGS:
            if target_index < lag:
                frame_modes[f"lag{lag}"] = {
                    "available": False,
                    "reason": "window boundary has no source frame at this lag",
                }
                continue
            source_index = target_index - lag
            source_generation = generations[source_index]
            pair_record, reuse_image = evaluate_mode(
                mode=f"lag{lag}",
                lag=lag,
                source_index=source_index,
                target_index=target_index,
                source_view=views[source_index],
                target_view=view,
                generation=source_generation,
                cache=lag_caches[lag],
                target_batch=fresh_batch,
                target_ids=anchor_ids,
                target_levels=level_ids,
                target_fresh_image=fresh_image,
                target_gt=gt,
                fresh_gt_metrics=fresh_gt_metrics,
                model=model,
                background=background,
                lpips_model=lpips_model,
                repeat=args.repeat,
                output_dir=output_dir,
                capacities=capacities,
            )
            pair_record["available"] = True
            frame_modes[f"lag{lag}"] = pair_record
            formal_pairs += 1
            if not pair_record["fidelity"]["pass"]:
                failed_formal_pairs += 1
            del reuse_image

        adversarial = None
        if target_index == 31:
            source_generation = generations[0]
            adversarial, adversarial_image = evaluate_mode(
                mode="adversarial_first_last",
                lag=31,
                source_index=0,
                target_index=31,
                source_view=views[0],
                target_view=view,
                generation=source_generation,
                cache=adversarial_cache,
                target_batch=fresh_batch,
                target_ids=anchor_ids,
                target_levels=level_ids,
                target_fresh_image=fresh_image,
                target_gt=gt,
                fresh_gt_metrics=fresh_gt_metrics,
                model=model,
                background=background,
                lpips_model=lpips_model,
                repeat=args.repeat,
                output_dir=output_dir,
                capacities=capacities,
            )
            adversarial["available"] = True
            del adversarial_image

        generations[target_index] = generation
        for old_index in list(generations):
            if old_index != 0 and old_index < target_index - 7:
                del generations[old_index]

        counts = fresh_batch.bundle_metadata.counts
        record = {
            "target_index": target_index,
            "target_camera": str(view.image_name),
            "selected_anchor_count": int(anchor_ids.numel()),
            "decoded_row_count": int(fresh_batch.xyz.shape[0]),
            "zero_row_bundle_count": int((counts == 0).sum()),
            "request_setup_ms": request_setup_ms,
            "fresh": {
                "decode_ms": fresh_decode_ms,
                "render_samples_ms": fresh_render_samples,
                "render_mean_ms": mean(fresh_render_samples),
                "gt_metrics": fresh_gt_metrics,
            },
            "generation_build": {
                "ms": build_ms,
                "rows": generation.occupied_rows,
                "descriptors": generation.descriptor_count,
                "raw_memory_bytes": generation.memory_bytes(),
                "memory_before_bytes": build_before,
                "memory_after_bytes": build_after,
                "memory_peak_bytes": build_peak,
                "scratch_bytes": max(0, build_peak - max(build_before, build_after)),
            },
            "same_pose_regression": {
                "payload_metadata_exact": same_pose_exact,
                "render_exact": same_pose_render_exact,
                "render_max_abs_delta": float(torch.abs(fresh_image - same_pose_image).max()),
                "resolution_ms": same_pose_result.timings_ms,
                "render_samples_ms": same_pose_render_samples,
            },
            "cross_pose": frame_modes,
            "adversarial": adversarial,
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
                "formal_pairs": formal_pairs,
                "failed_formal_pairs": failed_formal_pairs,
            },
        )
        del (
            fresh_batch,
            fresh_image,
            fresh_gt_metrics,
            same_pose_result,
            same_pose_image,
            gt,
        )
        torch.cuda.empty_cache()

    expected_pairs = sum(32 - lag for lag in FORMAL_LAGS)
    if formal_pairs != expected_pairs:
        raise RuntimeError(f"expected {expected_pairs} formal pairs, observed {formal_pairs}")
    summary = {
        "schema": "proxygs_step7b_scene_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "state": "complete",
        "scene": args.scene,
        "view_count": 32,
        "formal_pair_count": formal_pairs,
        "expected_formal_pair_count": expected_pairs,
        "failed_formal_pair_count": failed_formal_pairs,
        "same_pose_failure_count": same_pose_failures,
        "thresholds": THRESHOLDS,
        "hit_validity": "unconditional-residency",
        "reuse_policy_result": (
            "candidate_pass" if failed_formal_pairs == 0 and same_pose_failures == 0
            else "blocked_under_adr_0006"
        ),
        "per_lag": {
            f"lag{lag}": {
                "pair_count": sum(
                    1
                    for item in records
                    if item["cross_pose"][f"lag{lag}"].get("available") is True
                ),
                "failed_pairs": sum(
                    1
                    for item in records
                    if item["cross_pose"][f"lag{lag}"].get("available") is True
                    and item["cross_pose"][f"lag{lag}"]["fidelity"]["pass"] is False
                ),
            }
            for lag in FORMAL_LAGS
        },
        "input_identities_after": {
            name: _file_identity(Path(value["path"])) for name, value in inputs.items()
        },
        "completion_scope": (
            "unconditional cross-pose fidelity and cache cost calibration only; "
            "no pose-gated fallback, Future Residency, compaction policy, or schedule"
        ),
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        output_dir / "status.json",
        {
            "state": "complete",
            "scene": args.scene,
            "completed_views": 32,
            "expected_views": 32,
            "formal_pairs": formal_pairs,
            "failed_formal_pairs": failed_formal_pairs,
            "same_pose_failures": same_pose_failures,
        },
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--scene", required=True)
    result.add_argument("--source-path", type=Path, required=True)
    result.add_argument("--model-path", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--expected-views", type=int, required=True)
    result.add_argument("--iteration", type=int, default=40000)
    result.add_argument("--width", type=int, default=1600)
    result.add_argument("--height", type=int, default=900)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--repeat", type=int, default=1)
    result.add_argument("--max-views", type=int)
    result.add_argument("--formal", action="store_true")
    result.add_argument("--step6-review", type=Path, default=STEP6 / "review" / "final_step6_review.json")
    result.add_argument("--cache-trace", type=Path, default=STEP6 / "review" / "cache_trace_manifest.json")
    result.add_argument(
        "--retained-profile",
        type=Path,
        default=STEP6_DIAG / "qualification" / "retained_profile_conservative.json",
    )
    result.add_argument("--step7a-review", type=Path, default=STEP7A / "review" / "final_step7a_review.json")
    result.add_argument(
        "--step7a-exactness",
        type=Path,
        default=STEP7A / "review" / "same_pose_exactness_report.json",
    )
    result.add_argument(
        "--step7a-capacity",
        type=Path,
        default=STEP7A / "review" / "bundle_size_capacity_report.json",
    )
    result.add_argument(
        "--adr-0013",
        type=Path,
        default=Path("/ssddata/lun/gdmgs_artifacts/proxygs_step7b_cross_pose_20260916/manifests/adr_0013.md"),
    )
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

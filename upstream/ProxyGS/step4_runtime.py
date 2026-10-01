"""Step 4 camera, input, parity, and dense-selection contracts.

The CPU predicate deliberately mirrors ProxyGS' active pointwise depth test.
It consumes FoV/LoD candidates in caller order and never sorts or deduplicates
the result.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch


DEPTH_MARGIN = np.float32(0.3)
MIN_CAMERA_Z = np.float32(0.0001)
DEPTH_ORACLE_ATOL = 1.0e-3
DEPTH_ORACLE_RTOL = 2.0e-4
DEPTH_ORACLE_MEAN_LIMIT = 1.0e-4
DEPTH_ORACLE_P99_LIMIT = 5.0e-4
DEPTH_ORACLE_MAX_OUTLIER_FRACTION = 1.0e-4
DEPTH_ORACLE_MAX_COVERAGE_MISMATCH_FRACTION = 1.0e-4
DEPTH_ORACLE_NORMALIZED_MEAN_LIMIT = None
DEPTH_ORACLE_NORMALIZED_P99_LIMIT = 1.0e-4
INDEXED_DEPTH_ATOL = 1.0e-6
INDEXED_DEPTH_RTOL = 1.0e-6


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_identity(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_equal(first: Path, second: Path) -> bool:
    """Direct byte comparison without introducing a hash/checksum operation."""
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            a = left.read(1024 * 1024)
            b = right.read(1024 * 1024)
            if a != b:
                return False
            if not a:
                return True


def validate_mesh_input(path: Path, scene: str) -> Dict[str, Any]:
    required = {
        "format_version",
        "vertices",
        "faces",
        "triangle_ids",
        "triangle_aabb_min",
        "triangle_aabb_max",
        "triangle_centroids",
        "vertex_count",
        "face_count",
    }
    with np.load(path, allow_pickle=False, mmap_mode="r") as record:
        if set(record.files) != required:
            raise ValueError(f"{scene}: mesh input fields differ from the frozen schema")
        vertices = record["vertices"]
        faces = record["faces"]
        triangle_ids = record["triangle_ids"]
        if record["format_version"].dtype != np.int32 or record["format_version"].tolist() != [1]:
            raise ValueError(f"{scene}: unsupported mesh input format")
        if vertices.dtype != np.float64 or vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"{scene}: vertices must be float64[V,3]")
        if faces.dtype != np.int64 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"{scene}: faces must be int64[F,3]")
        if triangle_ids.dtype != np.int64 or triangle_ids.shape != (len(faces),):
            raise ValueError(f"{scene}: triangle_ids must be int64[F]")
        if not np.array_equal(triangle_ids, np.arange(len(faces), dtype=np.int64)):
            raise ValueError(f"{scene}: triangle IDs are not the complete stable 0..F-1 table")
        if record["vertex_count"].tolist() != [len(vertices)] or record["face_count"].tolist() != [len(faces)]:
            raise ValueError(f"{scene}: declared mesh counts do not match arrays")
        if len(vertices) and (not np.isfinite(vertices).all()):
            raise ValueError(f"{scene}: vertices contain non-finite values")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError(f"{scene}: face vertex IDs are out of range")
        return {
            "scene": scene,
            "input": file_identity(path),
            "format_version": 1,
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "triangle_id_policy": "complete_sorted_rows_0_to_F_minus_1",
        }


def load_mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as record:
        vertices = np.ascontiguousarray(record["vertices"], dtype=np.float64)
        faces = np.ascontiguousarray(record["faces"], dtype=np.int64)
    return vertices, faces


def camera_domain_from_view(view: Any):
    from gdmgs.mesh_index import CameraDomain

    w2c = np.ascontiguousarray(
        view.world_view_transform.transpose(0, 1).detach().cpu().numpy(), dtype=np.float64
    )
    tx = math.tan(float(view.FoVx) * 0.5)
    ty = math.tan(float(view.FoVy) * 0.5)
    return CameraDomain.parse(
        {
            "w2c": w2c,
            "angular_domain": (-tx, tx, -ty, ty),
            "near": float(view.znear),
            "far": float(view.zfar),
            "camera_id": str(view.image_name),
        }
    )


def camera_record(view: Any, domain: Any, index: int) -> Dict[str, Any]:
    return {
        "index": index,
        "camera": str(view.image_name),
        "width": int(view.image_width),
        "height": int(view.image_height),
        "FoVx": float(view.FoVx),
        "FoVy": float(view.FoVy),
        "Fx": float(view.Fx),
        "Fy": float(view.Fy),
        "Cx": float(view.Cx),
        "Cy": float(view.Cy),
        "resolution_scale": float(view.resolution_scale),
        "near": float(domain.near),
        "far": float(domain.far),
        "angular_domain": list(domain.angular_domain),
        "w2c": domain.w2c.tolist(),
    }


def _validate_dense_inputs(
    candidate_ids: np.ndarray,
    anchor_positions: np.ndarray,
    world_view_transform: np.ndarray,
    full_proj_transform: np.ndarray,
    depth: np.ndarray,
) -> None:
    if candidate_ids.dtype != np.int64 or candidate_ids.ndim != 1:
        raise TypeError("candidate_ids must be rank-one int64")
    if candidate_ids.size and (
        candidate_ids[0] < 0
        or candidate_ids[-1] >= len(anchor_positions)
        or np.any(candidate_ids[1:] <= candidate_ids[:-1])
    ):
        raise ValueError("candidate IDs must be valid, sorted, unique original rows")
    if anchor_positions.dtype != np.float32 or anchor_positions.ndim != 2 or anchor_positions.shape[1] != 3:
        raise TypeError("anchor_positions must be float32[N,3]")
    if world_view_transform.shape != (4, 4) or full_proj_transform.shape != (4, 4):
        raise ValueError("camera transforms must be [4,4]")
    if depth.dtype != np.float32 or depth.ndim != 2:
        raise TypeError("depth must be float32[H,W]")
    if np.isnan(depth).any() or np.isneginf(depth).any() or (depth[np.isfinite(depth)] <= 0).any():
        raise ValueError("depth must use positive finite camera-z or +inf")


def _source_order_transform(points: torch.Tensor, matrix: torch.Tensor, rows: int) -> torch.Tensor:
    """Mirror ProxyGS auxiliary.h transformPoint4x3/4x4 operand order.

    The stored matrices are column-major/transposed tensors. Keeping every
    multiply and add as a separate eager operation avoids replacing the source
    expression with a BLAS matmul whose cross-device reduction order differs.
    """
    flat = matrix.reshape(-1)
    outputs = []
    for row in range(rows):
        value = flat[row] * points[:, 0]
        value = value + flat[4 + row] * points[:, 1]
        value = value + flat[8 + row] * points[:, 2]
        value = value + flat[12 + row]
        outputs.append(value)
    return torch.stack(outputs, dim=1)


def dense_anchor_filter_cpu(
    candidate_ids: np.ndarray,
    anchor_positions: np.ndarray,
    world_view_transform: np.ndarray,
    full_proj_transform: np.ndarray,
    depth: np.ndarray,
    *,
    margin: float = float(DEPTH_MARGIN),
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Literal CPU dense pointwise predicate in original candidate order."""
    _validate_dense_inputs(
        candidate_ids, anchor_positions, world_view_transform, full_proj_transform, depth
    )
    start = time.perf_counter()
    candidate_tensor = torch.from_numpy(candidate_ids)
    points = torch.from_numpy(anchor_positions).index_select(0, candidate_tensor)
    view = torch.from_numpy(np.ascontiguousarray(world_view_transform, dtype=np.float32))
    projection = torch.from_numpy(np.ascontiguousarray(full_proj_transform, dtype=np.float32))
    p_hom = _source_order_transform(points, projection, 4)
    reciprocal_w = 1.0 / (p_hom[:, 3] + 1.0e-7)
    p_proj = p_hom[:, :3] * reciprocal_w[:, None]
    p_view = _source_order_transform(points, view, 3)
    width, height = depth.shape[1], depth.shape[0]
    columns = ((p_proj[:, 0] + 1.0) * width / 2.0).long()
    rows = ((p_proj[:, 1] + 1.0) * height / 2.0).long()
    positive = p_view[:, 2] > float(MIN_CAMERA_Z)
    inside = positive & (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    keep = positive.clone()
    sampled = torch.full_like(p_proj[:, 0], float("inf"))
    depth_tensor = torch.from_numpy(depth)
    sampled[inside] = depth_tensor[rows[inside], columns[inside]]
    finite_inside = inside & torch.isfinite(sampled)
    keep[finite_inside] = p_view[finite_inside, 2] <= sampled[finite_inside] + float(margin)
    selected = candidate_tensor[keep].numpy().copy()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return selected, {
        "candidate_count": int(len(candidate_ids)),
        "selected_count": int(len(selected)),
        "culled_nonpositive_z": int((~positive).sum().item()),
        "kept_out_of_image": int((positive & ~inside).sum().item()),
        "kept_infinite_depth": int((inside & ~torch.isfinite(sampled)).sum().item()),
        "culled_finite_depth": int((finite_inside & ~keep).sum().item()),
        "margin": float(np.float32(margin)),
        "minimum_camera_z": float(MIN_CAMERA_Z),
        "elapsed_ms": elapsed_ms,
        "order": "input_FoV_LoD_candidate_row_order",
    }


def dense_anchor_filter_cuda_reference(
    candidate_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    world_view_transform: torch.Tensor,
    full_proj_transform: torch.Tensor,
    depth: torch.Tensor,
    *,
    margin: float = float(DEPTH_MARGIN),
) -> tuple[np.ndarray, float]:
    """CUDA reference for the same active ProxyGS pointwise predicate."""
    if candidate_ids.dtype != torch.long or candidate_ids.ndim != 1 or candidate_ids.device.type != "cuda":
        raise TypeError("CUDA candidate IDs must be rank-one int64 on CUDA")
    torch.cuda.synchronize(candidate_ids.device)
    start = time.perf_counter()
    points = anchor_positions.index_select(0, candidate_ids)
    p_hom = _source_order_transform(points, full_proj_transform, 4)
    p_w = 1.0 / (p_hom[:, 3] + 1.0e-7)
    p_proj = p_hom[:, :3] * p_w[:, None]
    p_view = _source_order_transform(points, world_view_transform, 3)
    height, width = depth.shape
    columns = ((p_proj[:, 0] + 1.0) * width / 2.0).long()
    rows = ((p_proj[:, 1] + 1.0) * height / 2.0).long()
    positive = p_view[:, 2] > float(MIN_CAMERA_Z)
    inside = positive & (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    keep = positive.clone()
    sampled = torch.full_like(p_proj[:, 0], float("inf"))
    sampled[inside] = depth[rows[inside], columns[inside]]
    finite_inside = inside & torch.isfinite(sampled)
    keep[finite_inside] = p_view[finite_inside, 2] <= sampled[finite_inside] + float(margin)
    selected = candidate_ids[keep]
    values = selected.detach().cpu().numpy().copy()
    torch.cuda.synchronize(candidate_ids.device)
    return values, (time.perf_counter() - start) * 1000.0


def parity_stats(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float,
    rtol: float,
    mean_limit: float | None = None,
    p99_limit: float | None = None,
    max_limit: float | None = None,
    allow_local_edge_ties: bool = False,
    max_outlier_fraction: float | None = None,
    max_coverage_mismatch_fraction: float | None = None,
    normalized_mean_limit: float | None = None,
    normalized_p99_limit: float | None = None,
) -> Dict[str, Any]:
    if actual.shape != reference.shape:
        return {"pass": False, "shape_match": False, "actual_shape": list(actual.shape), "reference_shape": list(reference.shape)}
    actual_finite = np.isfinite(actual)
    reference_finite = np.isfinite(reference)
    coverage_match = np.array_equal(actual_finite, reference_finite)
    coverage_mismatch_pixels = int(np.count_nonzero(actual_finite != reference_finite))
    coverage_mismatch_fraction = coverage_mismatch_pixels / max(actual.size, 1)
    coverage_within = coverage_match or (
        max_coverage_mismatch_fraction is not None
        and coverage_mismatch_fraction <= max_coverage_mismatch_fraction
    )
    shared = actual_finite & reference_finite
    error = np.abs(actual[shared].astype(np.float64) - reference[shared].astype(np.float64))
    allowed = atol + rtol * np.abs(reference[shared].astype(np.float64))
    within = bool(np.all(error <= allowed)) if error.size else True
    outlier_count = int(np.count_nonzero(error > allowed))
    unexplained_edge_outliers = outlier_count
    if allow_local_edge_ties and outlier_count:
        actual64 = actual.astype(np.float64)
        reference64 = reference.astype(np.float64)
        rows, columns = np.nonzero(shared & ~np.isclose(actual, reference, atol=atol, rtol=rtol))
        unexplained_edge_outliers = 0
        for row, column in zip(rows.tolist(), columns.tolist()):
            neighborhood = reference64[
                max(0, row - 1) : min(reference.shape[0], row + 2),
                max(0, column - 1) : min(reference.shape[1], column + 2),
            ]
            finite_neighbors = neighborhood[np.isfinite(neighborhood)]
            if not finite_neighbors.size or not np.any(
                np.abs(finite_neighbors - actual64[row, column])
                <= atol + rtol * np.abs(finite_neighbors)
            ):
                unexplained_edge_outliers += 1
    if error.size:
        quantiles = np.quantile(error, [0.95, 0.99])
        normalized = error / (np.abs(reference[shared].astype(np.float64)) + 1.0)
        normalized_quantiles = np.quantile(normalized, [0.95, 0.99])
        maximum = float(error.max())
        mean = float(error.mean())
        p95, p99 = map(float, quantiles)
        normalized_mean = float(normalized.mean())
        normalized_p95, normalized_p99 = map(float, normalized_quantiles)
    else:
        maximum = mean = p95 = p99 = 0.0
        normalized_mean = normalized_p95 = normalized_p99 = 0.0
    outlier_fraction = outlier_count / max(int(shared.sum()), 1)
    aggregate_policy = any(
        limit is not None
        for limit in (
            mean_limit,
            p99_limit,
            max_limit,
            max_outlier_fraction,
            normalized_mean_limit,
            normalized_p99_limit,
        )
    ) or allow_local_edge_ties
    aggregate_within = (
        (mean_limit is None or mean <= mean_limit)
        and (p99_limit is None or p99 <= p99_limit)
        and (max_limit is None or maximum <= max_limit)
        and (max_outlier_fraction is None or outlier_fraction <= max_outlier_fraction)
        and (normalized_mean_limit is None or normalized_mean <= normalized_mean_limit)
        and (normalized_p99_limit is None or normalized_p99 <= normalized_p99_limit)
    )
    return {
        "pass": bool(coverage_within and (aggregate_within if aggregate_policy else within)),
        "shape_match": True,
        "coverage_match": bool(coverage_match),
        "coverage_mismatch_pixels": coverage_mismatch_pixels,
        "coverage_mismatch_fraction": coverage_mismatch_fraction,
        "max_coverage_mismatch_fraction": max_coverage_mismatch_fraction,
        "actual_finite_pixels": int(actual_finite.sum()),
        "reference_finite_pixels": int(reference_finite.sum()),
        "finite_overlap_pixels": int(shared.sum()),
        "mean_absolute_error": mean,
        "p95_absolute_error": p95,
        "p99_absolute_error": p99,
        "max_absolute_error": maximum,
        "normalized_mean_absolute_error": normalized_mean,
        "normalized_p95_absolute_error": normalized_p95,
        "normalized_p99_absolute_error": normalized_p99,
        "all_pixels_within_atol_rtol": within,
        "atol_rtol_outlier_pixels": outlier_count,
        "atol_rtol_outlier_fraction": outlier_fraction,
        "local_edge_tie_radius_pixels": 1 if allow_local_edge_ties else None,
        "unexplained_edge_outlier_pixels": unexplained_edge_outliers if allow_local_edge_ties else None,
        "atol": float(atol),
        "rtol": float(rtol),
        "mean_limit": mean_limit,
        "p99_limit": p99_limit,
        "max_limit": max_limit,
        "max_outlier_fraction": max_outlier_fraction,
        "normalized_mean_limit": normalized_mean_limit,
        "normalized_p99_limit": normalized_p99_limit,
    }


def mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None

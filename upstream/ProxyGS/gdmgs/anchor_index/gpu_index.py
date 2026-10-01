"""GPU-resident point-anchor index for the frozen ProxyGS G2 predicate.

The index deliberately keeps the stable final-PLY row layout instead of
reordering anchors into a GPU tree.  The existing CPU tree only obtains useful
terminal decisions for a small subset of nodes, so the first GPU experiment
measures a row-preserving parallel query before adding divergent traversal.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch


PROFILE = "proxygs-pointwise-center-v1"
RANGE_SPACE = "candidate_source_ordinal_half_open"


def _native():
    native_dir = os.environ.get("GDMGS_GPU_NATIVE_DIR")
    if native_dir and str(Path(native_dir).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(native_dir).resolve()))
    try:
        return import_module("ProxyGS_gpu_anchor_point_native")
    except ImportError as error:
        raise RuntimeError(
            "Build gdmgs/anchor_index/gpu_native/build.py and set "
            "GDMGS_GPU_NATIVE_DIR to its output directory"
        ) from error


def _source_order_transform(points: torch.Tensor, matrix: torch.Tensor, rows: int) -> torch.Tensor:
    """Match ProxyGS auxiliary.h multiply/add order without BLAS reduction."""
    flat = matrix.reshape(-1)
    outputs = []
    for row in range(rows):
        value = flat[row] * points[:, 0]
        value = value + flat[4 + row] * points[:, 1]
        value = value + flat[8 + row] * points[:, 2]
        value = value + flat[12 + row]
        outputs.append(value)
    return torch.stack(outputs, dim=1)


def _eager_keep_mask(
    positions: torch.Tensor,
    candidate_ids: torch.Tensor,
    depth: torch.Tensor,
    world_view: torch.Tensor,
    full_projection: torch.Tensor,
    margin: float,
    minimum_camera_z: float,
) -> torch.Tensor:
    points = positions.index_select(0, candidate_ids)
    p_hom = _source_order_transform(points, full_projection, 4)
    reciprocal_w = 1.0 / (p_hom[:, 3] + 1.0e-7)
    p_proj = p_hom[:, :3] * reciprocal_w[:, None]
    p_view = _source_order_transform(points, world_view, 3)
    height, width = depth.shape
    columns = ((p_proj[:, 0] + 1.0) * width / 2.0).long()
    rows = ((p_proj[:, 1] + 1.0) * height / 2.0).long()
    positive = p_view[:, 2] > minimum_camera_z
    inside = positive & (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    keep = positive.clone()
    sampled = torch.full_like(p_proj[:, 0], float("inf"))
    sampled[inside] = depth[rows[inside], columns[inside]]
    finite_inside = inside & torch.isfinite(sampled)
    keep[finite_inside] = p_view[finite_inside, 2] <= sampled[finite_inside] + margin
    return keep


@dataclass(frozen=True)
class GPUAnchorQueryResult:
    selected_anchor_ids: torch.Tensor
    keep_mask: torch.Tensor
    raw_ranges: torch.Tensor
    formal_ranges: torch.Tensor
    mode: str
    camera: str
    query_profile: str
    range_space: str
    resident_bytes: int
    _events: dict[str, torch.cuda.Event]
    _completion_wall_ms: float

    def timings(self) -> dict[str, float]:
        self._events["finish"].synchronize()
        return {
            "predicate_gpu_ms": float(
                self._events["start"].elapsed_time(self._events["predicate"])
            ),
            "materialization_gpu_ms": float(
                self._events["predicate"].elapsed_time(self._events["finish"])
            ),
            "gpu_resident_total_ms": float(
                self._events["start"].elapsed_time(self._events["finish"])
            ),
            "completion_wall_ms": float(self._completion_wall_ms),
        }

    def copy_ids_to_cpu(self) -> tuple[Any, float, float]:
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record(torch.cuda.current_stream(self.selected_anchor_ids.device))
        values = self.selected_anchor_ids.detach().cpu().numpy().copy()
        finish.record(torch.cuda.current_stream(self.selected_anchor_ids.device))
        finish.synchronize()
        return values, float(start.elapsed_time(finish)), (time.perf_counter() - wall_start) * 1000.0


class GPUAnchorIndex:
    """Immutable final-PLY row index resident on one CUDA device."""

    def __init__(self, positions: torch.Tensor, *, scene: str) -> None:
        if not isinstance(positions, torch.Tensor):
            raise TypeError("positions must be a torch Tensor")
        if positions.device.type != "cuda" or positions.dtype != torch.float32:
            raise TypeError("positions must be CUDA float32")
        if positions.ndim != 2 or positions.shape[1] != 3 or not positions.is_contiguous():
            raise ValueError("positions must be contiguous [N,3]")
        self.positions = positions.detach()
        self.scene = str(scene)
        self.anchor_count = int(positions.shape[0])
        self.device = positions.device
        self.resident_bytes = int(positions.numel() * positions.element_size())
        self._guard = (
            self.positions.data_ptr(),
            self.positions._version,
            tuple(self.positions.shape),
            self.positions.dtype,
            self.positions.device,
        )

    def _validate_binding(self) -> None:
        current = (
            self.positions.data_ptr(),
            self.positions._version,
            tuple(self.positions.shape),
            self.positions.dtype,
            self.positions.device,
        )
        if current != self._guard:
            raise RuntimeError("GPU anchor row index storage was modified")

    def query(
        self,
        candidate_ids: torch.Tensor,
        depth: torch.Tensor,
        world_view_transform: torch.Tensor,
        full_proj_transform: torch.Tensor,
        *,
        mode: str,
        camera: str = "",
        margin: float = float(torch.tensor(0.3, dtype=torch.float32)),
        minimum_camera_z: float = float(torch.tensor(0.0001, dtype=torch.float32)),
    ) -> GPUAnchorQueryResult:
        self._validate_binding()
        if mode not in {"eager", "fused"}:
            raise ValueError("GPU mode must be eager or fused")
        if (
            not isinstance(candidate_ids, torch.Tensor)
            or candidate_ids.dtype != torch.int64
            or candidate_ids.ndim != 1
            or candidate_ids.device != self.device
            or not candidate_ids.is_contiguous()
        ):
            raise TypeError("candidate_ids must be contiguous CUDA int64 on the index device")
        if (
            not isinstance(depth, torch.Tensor)
            or depth.dtype != torch.float32
            or depth.ndim != 2
            or depth.device != self.device
            or not depth.is_contiguous()
            or depth.shape[0] < 1
            or depth.shape[1] < 1
        ):
            raise TypeError("depth must be nonempty contiguous CUDA float32 on the index device")
        matrices = (world_view_transform, full_proj_transform)
        if any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.shape != (4, 4)
            or value.device != self.device
            or not value.is_contiguous()
            for value in matrices
        ):
            raise TypeError("camera transforms must be contiguous CUDA float32 [4,4]")
        if not (margin >= 0.0 and minimum_camera_z >= 0.0):
            raise ValueError("margin and minimum_camera_z must be nonnegative")

        events = {
            name: torch.cuda.Event(enable_timing=True)
            for name in ("start", "predicate", "finish")
        }
        stream = torch.cuda.current_stream(self.device)
        wall_start = time.perf_counter()
        events["start"].record(stream)
        if mode == "eager":
            keep = _eager_keep_mask(
                self.positions,
                candidate_ids,
                depth,
                world_view_transform,
                full_proj_transform,
                float(margin),
                float(minimum_camera_z),
            )
        else:
            keep = _native().pointwise_keep(
                self.positions,
                candidate_ids,
                depth,
                world_view_transform,
                full_proj_transform,
                float(margin),
                float(minimum_camera_z),
            ).bool()
        events["predicate"].record(stream)
        selected = candidate_ids[keep]
        padded = torch.cat(
            (
                torch.zeros(1, dtype=torch.bool, device=self.device),
                keep,
                torch.zeros(1, dtype=torch.bool, device=self.device),
            )
        )
        starts = torch.nonzero(padded[1:] & ~padded[:-1], as_tuple=False).flatten()
        ends = torch.nonzero(padded[:-1] & ~padded[1:], as_tuple=False).flatten()
        ranges = torch.stack((starts, ends), dim=1)
        events["finish"].record(stream)
        events["finish"].synchronize()
        return GPUAnchorQueryResult(
            selected_anchor_ids=selected,
            keep_mask=keep,
            raw_ranges=ranges,
            formal_ranges=ranges,
            mode=mode,
            camera=str(camera),
            query_profile=PROFILE,
            range_space=RANGE_SPACE,
            resident_bytes=self.resident_bytes,
            _events=events,
            _completion_wall_ms=(time.perf_counter() - wall_start) * 1000.0,
        )

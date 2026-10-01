"""Exact GPU-resident query contracts for the frozen Step 5 predicate."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdmgs.anchor_index import GPUAnchorIndex
from step4_runtime import DEPTH_MARGIN, MIN_CAMERA_Z, dense_anchor_filter_cpu


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def matrices() -> tuple[np.ndarray, np.ndarray]:
    return np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)


def expand_ranges(ranges: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    pieces = [candidates[begin:end] for begin, end in ranges]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def paired(
    positions: np.ndarray,
    candidates: np.ndarray,
    depth: np.ndarray,
    *,
    view: np.ndarray | None = None,
    projection: np.ndarray | None = None,
) -> None:
    if view is None or projection is None:
        view, projection = matrices()
    reference, _ = dense_anchor_filter_cpu(
        candidates,
        positions,
        np.ascontiguousarray(view),
        np.ascontiguousarray(projection),
        depth,
        margin=float(DEPTH_MARGIN),
    )
    device = torch.device("cuda:0")
    index = GPUAnchorIndex(torch.tensor(positions, device=device), scene="fixture")
    gpu_candidates = torch.tensor(candidates, dtype=torch.int64, device=device)
    gpu_depth = torch.tensor(depth, dtype=torch.float32, device=device)
    gpu_view = torch.tensor(view, dtype=torch.float32, device=device).contiguous()
    gpu_projection = torch.tensor(projection, dtype=torch.float32, device=device).contiguous()
    for mode in ("eager", "fused"):
        previous = None
        for _ in range(3):
            result = index.query(
                gpu_candidates,
                gpu_depth,
                gpu_view,
                gpu_projection,
                mode=mode,
                margin=float(DEPTH_MARGIN),
                minimum_camera_z=float(MIN_CAMERA_Z),
            )
            actual = result.selected_anchor_ids.cpu().numpy()
            ranges = result.formal_ranges.cpu().numpy()
            np.testing.assert_array_equal(actual, reference)
            np.testing.assert_array_equal(expand_ranges(ranges, candidates), reference)
            assert result.selected_anchor_ids.device.type == "cuda"
            assert result.formal_ranges.device.type == "cuda"
            assert result.selected_anchor_ids.dtype == torch.int64
            assert result.formal_ranges.dtype == torch.int64
            assert all(value >= 0 for value in result.timings().values())
            current = (actual.copy(), ranges.copy())
            if previous is not None:
                np.testing.assert_array_equal(current[0], previous[0])
                np.testing.assert_array_equal(current[1], previous[1])
            previous = current


def test_empty_all_keep_all_cull_and_canonical_runs():
    positions = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 3.0],
            [3.0, 0.0, 2.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.1],
            [0.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    )
    depth = np.full((5, 7), 1.0, dtype=np.float32)
    paired(positions, np.empty(0, dtype=np.int64), depth)
    paired(positions, np.arange(len(positions), dtype=np.int64), depth)
    paired(positions, np.array([0, 2, 3, 4], dtype=np.int64), np.full((5, 7), np.inf, np.float32))


def test_exact_camera_depth_and_pixel_boundaries():
    below_z = np.nextafter(np.float32(MIN_CAMERA_Z), np.float32(-np.inf))
    equal_z = np.float32(MIN_CAMERA_Z)
    above_z = np.nextafter(np.float32(MIN_CAMERA_Z), np.float32(np.inf))
    depth_value = np.float32(1.0)
    threshold = np.float32(depth_value + np.float32(DEPTH_MARGIN))
    positions = np.array(
        [
            [0.0, 0.0, below_z],
            [0.0, 0.0, equal_z],
            [0.0, 0.0, above_z],
            [0.0, 0.0, np.nextafter(threshold, np.float32(-np.inf))],
            [0.0, 0.0, threshold],
            [0.0, 0.0, np.nextafter(threshold, np.float32(np.inf))],
            [-1.1, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    paired(positions, np.arange(len(positions), dtype=np.int64), np.full((5, 7), depth_value, np.float32))


@pytest.mark.parametrize("seed", range(8))
def test_random_exact_parity(seed: int):
    random = np.random.default_rng(seed)
    count = 4097 if seed % 2 else 4095
    positions = random.uniform([-4.0, -3.0, -1.0], [4.0, 3.0, 8.0], (count, 3)).astype(np.float32)
    candidates = np.sort(random.choice(count, size=count * 3 // 4, replace=False)).astype(np.int64)
    depth = random.uniform(0.05, 7.0, (73, 117)).astype(np.float32)
    depth[random.random(depth.shape) < 0.1] = np.inf
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = projection[1, 1] = projection[2, 2] = projection[2, 3] = 1.0
    paired(positions, candidates, depth, view=np.eye(4, dtype=np.float32), projection=projection)


def test_binding_guard_rejects_in_place_mutation():
    positions = torch.ones((4, 3), dtype=torch.float32, device="cuda:0")
    index = GPUAnchorIndex(positions, scene="fixture")
    index.positions.add_(1.0)
    with pytest.raises(RuntimeError, match="modified"):
        index.query(
            torch.arange(4, dtype=torch.int64, device="cuda:0"),
            torch.ones((2, 2), dtype=torch.float32, device="cuda:0"),
            torch.eye(4, dtype=torch.float32, device="cuda:0"),
            torch.eye(4, dtype=torch.float32, device="cuda:0"),
            mode="eager",
        )

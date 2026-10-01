"""PLY-row-preserving CPU index for ProxyGS' pointwise depth predicate.

The saved binned-SAH point BVH covers every finalized anchor row.  A query may
only certify complete node culls; every unresolved leaf candidate is evaluated
by the same float32 point predicate used by the linear mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


FORMAT = "proxygs-pointwise-center-bvh-v2"
PROFILE = "proxygs-pointwise-center-v1"
RANGE_SPACE = "candidate_source_ordinal_half_open"
BUILD_METHOD = "binned_sah"


def _array(value: Any, dtype: Any, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must be a numpy {np.dtype(dtype)} array")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise ValueError(f"{name} must have shape {shape}")
    return np.ascontiguousarray(value)


def _native():
    native_dir = os.environ.get("GDMGS_NATIVE_DIR")
    if native_dir and str(Path(native_dir).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(native_dir).resolve()))
    try:
        return import_module("ProxyGS_anchor_point_native")
    except ImportError as error:
        raise RuntimeError(
            "Build gdmgs/anchor_index/native with CMake and set "
            "GDMGS_NATIVE_DIR to the directory containing "
            "ProxyGS_anchor_point_native."
        ) from error


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@dataclass(frozen=True)
class AnchorPointQueryResult:
    selected_anchor_ids: np.ndarray
    raw_ranges: np.ndarray
    formal_ranges: np.ndarray
    range_space: str
    counters: dict[str, Any]
    timings: dict[str, float]
    mode: str
    query_profile: str
    scene: str
    camera: str

    def expand_ranges(self, candidate_ids: np.ndarray) -> np.ndarray:
        """Expand candidate-ordinal half-open ranges in original source order."""
        candidate_ids = _array(candidate_ids, np.int64, (None,), "candidate_ids")
        pieces = [candidate_ids[begin:end] for begin, end in self.formal_ranges]
        if not pieces:
            return np.empty(0, dtype=np.int64)
        return np.ascontiguousarray(np.concatenate(pieces), dtype=np.int64)


class AnchorPointIndex:
    """Immutable full-PLY point BVH with native linear/tree query modes."""

    def __init__(
        self,
        tree: Any,
        *,
        scene: str,
        leaf_capacity: int,
        max_depth: int,
        source: Mapping[str, Any] | None,
        build_ms: float,
    ) -> None:
        self._tree = tree
        self.scene = str(scene)
        self.leaf_capacity = int(leaf_capacity)
        self.max_depth = int(max_depth)
        self.build_method = BUILD_METHOD
        self.source = dict(source) if source is not None else None
        self.build_ms = float(build_ms)
        layout = tree.layout()
        self.positions = np.ascontiguousarray(layout["positions"], dtype=np.float32)
        self.dfs_to_row = np.ascontiguousarray(layout["dfs_to_row"], dtype=np.int64)
        self.rank_of_row = np.ascontiguousarray(layout["rank_of_row"], dtype=np.int64)
        self.intervals = np.ascontiguousarray(layout["intervals"], dtype=np.int64)
        for value in (self.positions, self.dfs_to_row, self.rank_of_row, self.intervals):
            value.flags.writeable = False
        self.anchor_count = len(self.positions)
        self.node_count = len(self.intervals)

    @classmethod
    def build(
        cls,
        positions: np.ndarray,
        *,
        scene: str,
        leaf_capacity: int = 64,
        max_depth: int = 32,
        source: Mapping[str, Any] | None = None,
    ) -> "AnchorPointIndex":
        points = _array(positions, np.float32, (None, 3), "positions")
        if not np.isfinite(points).all():
            raise ValueError("positions must be finite")
        if type(leaf_capacity) is not int or leaf_capacity < 1:
            raise ValueError("leaf_capacity must be a positive integer")
        if type(max_depth) is not int or not 1 <= max_depth <= 64:
            raise ValueError("max_depth must be in [1, 64]")
        start = time.perf_counter()
        tree = _native().AnchorPointTree(points, leaf_capacity, max_depth, BUILD_METHOD)
        build_ms = (time.perf_counter() - start) * 1000.0
        return cls(
            tree,
            scene=scene,
            leaf_capacity=leaf_capacity,
            max_depth=max_depth,
            source=source,
            build_ms=build_ms,
        )

    def layout(self) -> dict[str, np.ndarray]:
        return {
            name: np.ascontiguousarray(value)
            for name, value in self._tree.layout().items()
        }

    def save(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": FORMAT,
            "query_profile": PROFILE,
            "scene": self.scene,
            "anchor_count": self.anchor_count,
            "node_count": self.node_count,
            "leaf_capacity": self.leaf_capacity,
            "max_depth": self.max_depth,
            "build_method": BUILD_METHOD,
            "source": self.source,
            "build_ms": self.build_ms,
            "row_identity": "final_ply_original_row_order",
            "range_space": RANGE_SPACE,
            "query_policy": {
                "certificate_minimum_candidates": int(
                    _native().certificate_minimum_candidates
                ),
                "outside_image_terminal_keep": bool(
                    _native().outside_image_terminal_keep
                ),
                "nonpositive_camera_z_terminal_cull": True,
                "depth_minimum_terminal_keep": False,
                "depth_range_node_certificate": False,
                "unresolved_candidate_policy": "descend_or_exact_pointwise_leaf_fallback",
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, metadata=np.asarray(json.dumps(metadata)), **self.layout())
        os.replace(temporary, path)
        return metadata

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        scene: str | None = None,
        source: Mapping[str, Any] | None = None,
    ) -> "AnchorPointIndex":
        start = time.perf_counter()
        with np.load(path, allow_pickle=False) as record:
            required = {
                "metadata",
                "positions",
                "dfs_to_row",
                "rank_of_row",
                "intervals",
                "children",
                "node_bounds",
                "partition_bounds",
            }
            if set(record.files) != required:
                raise ValueError("saved anchor index fields differ from the frozen schema")
            metadata = json.loads(str(record["metadata"].item()))
            layout = {
                name: np.ascontiguousarray(record[name])
                for name in required
                if name != "metadata"
            }
        if metadata.get("format") != FORMAT or metadata.get("query_profile") != PROFILE:
            raise ValueError("unsupported anchor point index format/profile")
        if scene is not None and metadata.get("scene") != str(scene):
            raise ValueError("saved anchor index belongs to another scene")
        if source is not None and metadata.get("source") != dict(source):
            raise ValueError("saved anchor index source identity drifted")
        if metadata.get("build_method") != BUILD_METHOD:
            raise ValueError("saved Anchor Index build method drifted")
        tree = _native().AnchorPointTree.from_layout(
            layout,
            int(metadata["leaf_capacity"]),
            int(metadata["max_depth"]),
            BUILD_METHOD,
        )
        result = cls(
            tree,
            scene=metadata["scene"],
            leaf_capacity=int(metadata["leaf_capacity"]),
            max_depth=int(metadata["max_depth"]),
            source=metadata.get("source"),
            build_ms=float(metadata["build_ms"]),
        )
        if result.anchor_count != int(metadata["anchor_count"]):
            raise ValueError("saved anchor count does not match topology")
        if result.node_count != int(metadata["node_count"]):
            raise ValueError("saved node count does not match topology")
        result.load_ms = (time.perf_counter() - start) * 1000.0
        return result

    def query(
        self,
        candidate_ids: np.ndarray,
        depth: np.ndarray,
        world_view_transform: np.ndarray,
        full_proj_transform: np.ndarray,
        *,
        mode: str,
        camera: str = "",
        margin: float = float(np.float32(0.3)),
        minimum_camera_z: float = float(np.float32(0.0001)),
        trusted_buffers: bool = False,
        reuse_buffers: bool = False,
    ) -> AnchorPointQueryResult:
        ids = _array(candidate_ids, np.int64, (None,), "candidate_ids")
        depth = _array(depth, np.float32, (None, None), "depth")
        world_view = _array(world_view_transform, np.float32, (4, 4), "world_view_transform")
        full_projection = _array(full_proj_transform, np.float32, (4, 4), "full_proj_transform")
        if mode not in {"linear", "tree"}:
            raise ValueError("mode must be linear or tree")
        raw = self._tree.query(
            ids,
            depth,
            world_view,
            full_projection,
            mode,
            np.float32(margin),
            np.float32(minimum_camera_z),
            bool(trusted_buffers),
            bool(reuse_buffers),
        )
        result = AnchorPointQueryResult(
            selected_anchor_ids=np.ascontiguousarray(raw["selected_anchor_ids"], dtype=np.int64),
            raw_ranges=np.ascontiguousarray(raw["raw_ranges"], dtype=np.int64),
            formal_ranges=np.ascontiguousarray(raw["formal_ranges"], dtype=np.int64),
            range_space=str(raw["range_space"]),
            counters={name: int(value) for name, value in raw["counters"].items()},
            timings={name: float(value) for name, value in raw["timings"].items()},
            mode=mode,
            query_profile=PROFILE,
            scene=self.scene,
            camera=str(camera),
        )
        if result.range_space != RANGE_SPACE:
            raise RuntimeError("native range space differs from the Python contract")
        expanded = result.expand_ranges(ids)
        if not np.array_equal(expanded, result.selected_anchor_ids):
            raise RuntimeError("formal ranges do not expand to selected IDs in source order")
        return result


def source_identity(path: Path, *, iteration: int = 40000) -> dict[str, Any]:
    return {**file_identity(path), "iteration": int(iteration)}

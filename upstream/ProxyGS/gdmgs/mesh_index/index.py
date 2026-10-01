"""Validated Python boundary for owned native mesh/index buffers.

W2C multiplies column vectors. Angular coordinates are camera x/z and y/z.
Triangle retrieval is conservative: an accepted triangle may cross a frustum
corner without actually intersecting it, but no centroid test is used.
"""

from dataclasses import dataclass
from importlib import import_module
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping

import numpy as np


def _native():
    directory = os.environ.get("GDMGS_NATIVE_DIR")
    if directory and directory not in sys.path:
        sys.path.insert(0, directory)
    try:
        return import_module("GDMGS_mesh_native")
    except ImportError as exc:
        raise RuntimeError(
            "Build gdmgs/mesh_index/native with CMake and set GDMGS_NATIVE_DIR "
            "to the directory containing GDMGS_mesh_native."
        ) from exc


def _array(value, dtype, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must be a numpy {np.dtype(dtype)} array")
    if value.ndim != len(shape) or any(
        want is not None and got != want for got, want in zip(value.shape, shape)
    ):
        raise ValueError(f"{name} must have shape {shape}")
    return np.ascontiguousarray(value)


@dataclass(frozen=True)
class CameraDomain:
    w2c: np.ndarray
    angular_domain: tuple
    near: float = 0.01
    far: float = float("inf")
    camera_id: str = ""

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            fields = value.__dict__
        elif isinstance(value, Mapping):
            fields = value
        else:
            fields = vars(value)
        w2c = _array(fields["w2c"], np.float64, (4, 4), "w2c")
        domain = tuple(float(x) for x in fields["angular_domain"])
        near, far = float(fields.get("near", 0.01)), float(fields.get("far", float("inf")))
        if len(domain) != 4 or not np.isfinite(domain).all():
            raise ValueError("angular_domain must contain four finite coordinates")
        if not np.isfinite(w2c).all() or not np.array_equal(w2c[3], [0, 0, 0, 1]):
            raise ValueError("w2c must be a finite column-vector affine transform")
        if not (domain[0] < domain[1] and domain[2] < domain[3]):
            raise ValueError("angular_domain bounds must be increasing")
        if not np.isfinite(near) or near <= 0 or not far > near:
            raise ValueError("camera depth interval must satisfy 0 < near < far")
        w2c = w2c.copy()
        w2c.flags.writeable = False
        return cls(w2c, domain, near, far, str(fields.get("camera_id", "")))

    def equivalent(self, other):
        return (self.camera_id == other.camera_id and self.near == other.near
                and self.far == other.far and self.angular_domain == other.angular_domain
                and np.array_equal(self.w2c, other.w2c))


@dataclass(frozen=True)
class MeshQueryResult:
    triangle_ids: np.ndarray
    mesh_token: str
    camera_domain: CameraDomain
    counters: dict
    elapsed_ms: float
    complete: bool = True

    @property
    def camera_id(self):
        return self.camera_domain.camera_id


class MeshIndex:
    """Object-split index; all face rows appear exactly once in leaf references."""

    def __init__(self, vertices, triangles, *, method="median", leaf_size=8, mesh_token=""):
        self.vertices = _array(vertices, np.float64, (None, 3), "vertices").copy()
        self.triangles = _array(triangles, np.int64, (None, 3), "triangles").copy()
        if not np.isfinite(self.vertices).all():
            raise ValueError("vertices must be finite")
        if self.triangles.size and (
            self.triangles.min() < 0 or self.triangles.max() >= len(self.vertices)
        ):
            raise ValueError("triangle vertex IDs are out of bounds")
        if type(leaf_size) is not int or leaf_size <= 0:
            raise ValueError("leaf_size must be a positive integer")
        self.mesh_token = str(mesh_token)
        self.build_settings = {"method": method, "leaf_size": leaf_size}
        self._mesh = _native().MeshIndex(self.vertices, self.triangles, method, leaf_size)
        self.build_ms = float(self._mesh.build_ms)
        self.vertices.flags.writeable = False
        self.triangles.flags.writeable = False

    @classmethod
    def build(cls, triangle_table, build_settings=None):
        settings = dict(build_settings or {})
        if isinstance(triangle_table, Mapping):
            vertices = triangle_table["vertices"]
            triangles = triangle_table.get("triangles", triangle_table.get("faces"))
            token = triangle_table.get("mesh_token", "")
        else:
            vertices = triangle_table.vertices
            triangles = getattr(triangle_table, "triangles", None)
            if triangles is None:
                triangles = triangle_table.faces
            token = getattr(triangle_table, "mesh_token", "")
        settings.setdefault("mesh_token", token)
        return cls(vertices, triangles, **settings)

    def query(self, camera_domain, *, backend="bvh", threads=1):
        if backend not in {"bvh", "brute_force", "optimized_bvh"}:
            raise ValueError("backend must be bvh, optimized_bvh, or brute_force")
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        camera = CameraDomain.parse(camera_domain)
        if backend == "optimized_bvh":
            raw = self._mesh.query_optimized(
                camera.w2c, camera.angular_domain, camera.near, camera.far, threads
            )
        else:
            raw = self._mesh.query(camera.w2c, camera.angular_domain, camera.near,
                                   camera.far, backend == "brute_force")
        ids = raw.pop("triangle_ids")
        elapsed = float(raw.pop("elapsed_ms"))
        complete = bool(raw.pop("complete"))
        if not complete:
            raise RuntimeError("native mesh query did not complete")
        ids.flags.writeable = False
        raw["backend"] = backend
        raw["mesh_triangles"] = len(self.triangles)
        return MeshQueryResult(ids, self.mesh_token, camera, raw, elapsed, complete)

    def inspect_layout(self):
        """Owned copies for topology and reference auditing; never native pointers."""
        return self._mesh.layout()

    def save(self, path):
        """Persist actual BVH arrays and mesh values; no pickle or content hash."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        layout = self.inspect_layout()
        metadata = {
            "format": "gdmgs-mesh-bvh-v1", "mesh_token": self.mesh_token,
            "build_settings": self.build_settings, "build_ms": self.build_ms,
            "vertex_count": len(self.vertices), "triangle_count": len(self.triangles),
            "node_count": len(layout["nodes"]),
        }
        with path.open("wb") as stream:
            np.savez(stream, vertices=self.vertices, triangles=self.triangles,
                     triangle_refs=layout["triangle_refs"], nodes=layout["nodes"],
                     bounds=layout["bounds"], metadata=np.array(json.dumps(metadata)))
        return metadata

    @classmethod
    def load(cls, path, *, mesh_token=None):
        """Load saved topology, validating geometry bounds and every reference.

        No BVH rebuilding occurs here. load_ms and original build_ms are distinct.
        """
        start = perf_counter()
        with np.load(path, allow_pickle=False) as record:
            required = {"vertices", "triangles", "triangle_refs", "nodes", "bounds", "metadata"}
            if set(record.files) != required:
                raise ValueError("persisted mesh index has incorrect fields")
            metadata = json.loads(str(record["metadata"].item()))
            if metadata["format"] != "gdmgs-mesh-bvh-v1":
                raise ValueError("unsupported mesh index format")
            if mesh_token is not None and metadata["mesh_token"] != str(mesh_token):
                raise ValueError("persisted index belongs to a different mesh")
            obj = cls.__new__(cls)
            obj.vertices = _array(record["vertices"], np.float64, (None, 3), "vertices")
            obj.triangles = _array(record["triangles"], np.int64, (None, 3), "triangles")
            refs = _array(record["triangle_refs"], np.int64, (None,), "triangle_refs")
            nodes = _array(record["nodes"], np.int64, (None, 4), "nodes")
            bounds = _array(record["bounds"], np.float64, (None, 2, 3), "bounds")
        if (metadata["vertex_count"] != len(obj.vertices)
                or metadata["triangle_count"] != len(obj.triangles)
                or metadata["node_count"] != len(nodes)):
            raise ValueError("persisted mesh index count mismatch")
        obj.mesh_token = str(metadata["mesh_token"])
        obj.build_settings = metadata["build_settings"]
        obj._mesh = _native().MeshIndex.from_layout(
            obj.vertices, obj.triangles, refs, nodes, bounds,
            obj.build_settings["method"], obj.build_settings["leaf_size"])
        obj.vertices.flags.writeable = False
        obj.triangles.flags.writeable = False
        obj.build_ms = float(metadata["build_ms"])
        obj.load_ms = (perf_counter() - start) * 1000
        return obj

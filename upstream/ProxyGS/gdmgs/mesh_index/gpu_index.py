"""GPU parallel triangle scan and BVH leaf-cluster relevance queries.

The indexed backend scans every leaf AABB in parallel. It does not traverse
from the BVH root. Only triangles in surviving leaf clusters run the common
complete-triangle predicate. Results remain sorted CUDA int64 face-row IDs.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from time import perf_counter, time_ns

import numpy as np

from .index import CameraDomain, MeshIndex


def camera_planes(camera_domain):
    """World halfspaces plus outward coefficient uncertainty, float64[P,8]."""
    camera = CameraDomain.parse(camera_domain)
    w = camera.w2c
    xmin, xmax, ymin, ymax = camera.angular_domain
    terms = [(w[0], -xmin * w[2]), (xmax * w[2], -w[0]),
             (w[1], -ymin * w[2]), (ymax * w[2], -w[1])]
    near = np.zeros(4, dtype=np.float64)
    near[3] = -camera.near
    terms.append((w[2], near))
    if np.isfinite(camera.far):
        far = np.zeros(4, dtype=np.float64)
        far[3] = camera.far
        terms.append((-w[2], far))
    result = np.empty((len(terms), 8), dtype=np.float64)
    for plane, (first, second) in enumerate(terms):
        result[plane, :4] = first + second
        result[plane, 4:] = np.nextafter(
            16 * np.finfo(np.float64).eps * (np.abs(first) + np.abs(second) + 1), np.inf)
    if not np.isfinite(result).all():
        raise ValueError("Camera halfspace coefficients exceed finite float64 capacity")
    return result


def leaf_layout(mesh_index):
    """Derive a complete face-to-leaf mapping from the validated saved topology."""
    layout = mesh_index.inspect_layout()
    nodes, bounds, refs = layout["nodes"], layout["bounds"], layout["triangle_refs"]
    leaf_nodes = np.flatnonzero(nodes[:, 0] < 0)
    face_leaf = np.full(len(mesh_index.triangles), -1, dtype=np.int64)
    counts = np.empty(len(leaf_nodes), dtype=np.int64)
    for leaf, node in enumerate(leaf_nodes):
        begin, end = nodes[node, 2:]
        ids = refs[begin:end]
        if np.any(face_leaf[ids] != -1):
            raise ValueError("A face occurs in more than one BVH leaf")
        face_leaf[ids] = leaf
        counts[leaf] = end - begin
    if np.any(face_leaf < 0):
        raise ValueError("A face is missing from the BVH leaf partition")
    return np.ascontiguousarray(bounds[leaf_nodes]), face_leaf, counts


def _native():
    from torch.utils.cpp_extension import load
    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        raise RuntimeError("Set a scoped TORCH_EXTENSIONS_DIR before building GPU mesh queries")
    return load(name="_gdmgs_mesh_gpu_native",
                sources=[str(Path(__file__).with_name("native") / "gpu_query.cu")],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--fmad=false"],
                with_cuda=True, verbose=False)


@dataclass
class GPUMeshQueryResult:
    triangle_ids: object
    mesh_token: str
    camera_domain: CameraDomain
    index_token: str
    timings: dict
    _counts: dict
    _active_leaves: object = None
    _leaf_counts: object = None
    _events: object = None
    complete: bool = True
    _ids_version: int = field(init=False, repr=False)
    _ids_identity: int = field(init=False, repr=False)

    def __post_init__(self):
        self._ids_version = self.triangle_ids._version
        self._ids_identity = id(self.triangle_ids)

    @property
    def camera_id(self):
        return self.camera_domain.camera_id

    def validate(self):
        if (not self.complete or id(self.triangle_ids) != self._ids_identity
                or self.triangle_ids._version != self._ids_version):
            raise ValueError("GPU triangle query was incomplete or modified")

    def collect_counters(self):
        """Resolve small logging statistics after the enclosing measured frame."""
        self.validate()
        if self._events is not None:
            self._events[1].synchronize()
            self.timings["gpu_elapsed_ms"] = self._events[0].elapsed_time(self._events[1])
            self._events = None
        if self._active_leaves is not None:
            import torch
            values = torch.stack((self._active_leaves.sum(),
                                  self._leaf_counts[self._active_leaves].sum())).cpu().tolist()
            self._counts.update(active_leaf_clusters=int(values[0]), tested_triangles=int(values[1]))
            self._active_leaves = None
            self._leaf_counts = None
        return self._counts

    @property
    def counters(self):
        return self.collect_counters()

    @property
    def elapsed_ms(self):
        self.collect_counters()
        return self.timings["gpu_elapsed_ms"]


class GPUMeshIndex:
    """GPU full triangle scan / GPU BVH leaf-cluster scan, same geometric test."""

    def __init__(self, mesh_index, *, device="cuda:0"):
        import torch
        if not isinstance(mesh_index, MeshIndex):
            raise TypeError("GPUMeshIndex requires the validated CPU MeshIndex triangle table")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GPUMeshIndex requires CUDA")
        start = perf_counter()
        bounds, face_leaf, counts = leaf_layout(mesh_index)
        self.cpu_index = mesh_index
        self.mesh_token = mesh_index.mesh_token
        self.index_token = f"{self.mesh_token}:gpu-leaf-clusters:{os.getpid()}:{time_ns()}"
        self._torch = torch
        self._native = _native()
        with torch.cuda.device(self.device), torch.no_grad():
            self._vertices = torch.tensor(mesh_index.vertices.copy(), dtype=torch.float64, device=self.device)
            self._triangles = torch.tensor(mesh_index.triangles.copy(), dtype=torch.int64, device=self.device)
            self._leaf_bounds = torch.tensor(bounds, dtype=torch.float64, device=self.device)
            self._face_leaf = torch.tensor(face_leaf, dtype=torch.int64, device=self.device)
            self._leaf_counts = torch.tensor(counts, dtype=torch.int64, device=self.device)
            self._all_leaves = torch.ones(len(counts), dtype=torch.bool, device=self.device)
            torch.cuda.synchronize(self.device)
        self.initialization_ms = (perf_counter() - start) * 1000
        self.resident_bytes = sum(value.numel() * value.element_size() for value in
                                  (self._vertices, self._triangles, self._leaf_bounds,
                                   self._face_leaf, self._leaf_counts, self._all_leaves))

    def query(self, camera_domain, *, backend="bvh"):
        if backend not in {"bvh", "brute_force"}:
            raise ValueError("GPU mesh backend must be bvh or brute_force")
        start = perf_counter()
        torch = self._torch
        camera = CameraDomain.parse(camera_domain)
        planes = camera_planes(camera)
        with torch.cuda.device(self.device), torch.no_grad():
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
            gpu_planes = torch.tensor(planes, dtype=torch.float64, device=self.device)
            active = (self._native.leaf_mask(self._leaf_bounds, gpu_planes)
                      if backend == "bvh" else self._all_leaves)
            mask = self._native.triangle_mask(self._vertices, self._triangles, gpu_planes,
                                               active, self._face_leaf, backend == "bvh")
            # Compaction may perform the small shape synchronization required
            # by torch.nonzero. No full triangle ID array leaves the device.
            ids = torch.nonzero(mask, as_tuple=False).flatten()
            events[1].record()
        counts = {"backend": backend, "actual_device": str(self.device),
                  "mesh_token": self.mesh_token, "index_token": self.index_token,
                  "camera_id": camera.camera_id,
                  "triangle_table_identity": {"mesh_token": self.mesh_token,
                                              "vertices": len(self.cpu_index.vertices),
                                              "triangles": len(self.cpu_index.triangles),
                                              "face_id_policy": "stable_triangle_table_row"},
                  "index_structure": "gpu_bvh_leaf_cluster_scan" if backend == "bvh" else "gpu_parallel_triangle_scan",
                  "mesh_triangles": len(self.cpu_index.triangles),
                  "returned_triangles": ids.numel(), "visited_nodes": 0,
                  "tested_leaf_clusters": len(self._leaf_counts) if backend == "bvh" else 0,
                  "active_leaf_clusters": None if backend == "bvh" else len(self._leaf_counts),
                  "tested_triangles": None if backend == "bvh" else len(self.cpu_index.triangles),
                  "triangle_thread_slots": len(self.cpu_index.triangles),
                  "bvh_builder": self.cpu_index.build_settings["method"],
                  "leaf_size": self.cpu_index.build_settings["leaf_size"],
                  "camera_upload_bytes": planes.nbytes, "triangle_id_download_bytes": 0,
                  "algorithm_shape_sync": "torch.nonzero output-size synchronization",
                  "resident_index_bytes": self.resident_bytes,
                  "logging_download_bytes": 16 if backend == "bvh" else 0}
        return GPUMeshQueryResult(ids, self.mesh_token, camera, self.index_token,
                                   {"host_enqueue_ms": (perf_counter() - start) * 1000,
                                    "gpu_elapsed_ms": None}, counts,
                                   active if backend == "bvh" else None,
                                   self._leaf_counts if backend == "bvh" else None, events)

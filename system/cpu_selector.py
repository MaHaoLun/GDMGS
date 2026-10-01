"""All-CPU frame Selection for the capped-cell anchor candidate."""
import ctypes
import os
import time
from pathlib import Path

import numpy as np
import torch

from fast_geometry import fast_hole_planes, visible_leaf_boxes
from anchor_frustum.gpu_construction import GPUThreeTrees, camera_tensor
from gdmgs.mesh_index.gpu_index import camera_planes


class CPUSelector:
    def __init__(self, experiment, cells_path, native_path, threads_per_frame):
        self.experiment = experiment
        self.threads_per_frame = int(threads_per_frame)
        self.native = ctypes.CDLL(str(native_path))
        self.native.cpu_select.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] + \
            [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.native.cpu_select.restype = ctypes.c_int
        self.native.cpu_select_tree.argtypes = [ctypes.c_void_p] * 7 + \
            [ctypes.c_int64, ctypes.c_int64] + [ctypes.c_void_p] * 2 + \
            [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.native.cpu_select_tree.restype = ctypes.c_int
        self.native.cpu_select_tree_keep.argtypes = [ctypes.c_void_p] * 10 + \
            [ctypes.c_int64, ctypes.c_int64] + [ctypes.c_void_p] * 2 + \
            [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.native.cpu_select_tree_keep.restype = ctypes.c_int
        self.native.cpu_candidates.argtypes = [ctypes.c_void_p] * 4 + \
            [ctypes.c_int64] + [ctypes.c_float] * 3 + \
            [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.native.cpu_candidates.restype = ctypes.c_int
        self.native.cpu_build_holes.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_int, ctypes.c_int,
                                                ctypes.c_void_p, ctypes.c_void_p]
        self.native.cpu_build_holes.restype = ctypes.c_int
        self.backend = os.environ.get('CPU_SELECTOR_BACKEND', 'tree')
        assert self.backend in ('dense', 'tree', 'tree_keep')
        self.hole_backend = os.environ.get('CPU_HOLE_BACKEND', 'native')
        assert self.hole_backend in ('native', 'scalar')
        self.candidate_backend = os.environ.get('CPU_CANDIDATE_BACKEND', 'native')
        assert self.candidate_backend in ('native', 'numpy')
        model = experiment.rt.model
        index = GPUThreeTrees(model)
        self.bounds = np.ascontiguousarray(index.bounds.detach().cpu().numpy(), dtype=np.float64)
        self.nodes = np.ascontiguousarray(index.nodes.detach().cpu().numpy(), dtype=np.float64)
        self.left = np.ascontiguousarray(index.left.detach().cpu().numpy(), dtype=np.int32)
        self.right = np.ascontiguousarray(index.right.detach().cpu().numpy(), dtype=np.int32)
        self.order = np.ascontiguousarray(index.sorted_ids.detach().cpu().numpy(), dtype=np.int64)
        self.leaves = (len(self.bounds)+31)//32
        internal = self.leaves - 1
        self.range_begin = np.full(len(self.nodes), -1, dtype=np.int64)
        self.range_end = np.full(len(self.nodes), -1, dtype=np.int64)
        def fill_range(node):
            if self.range_begin[node] >= 0:
                return int(self.range_begin[node]), int(self.range_end[node])
            if node >= internal:
                leaf = node - internal
                begin, end = leaf*32, min(len(self.bounds), (leaf+1)*32)
            else:
                a = fill_range(int(self.left[node]))
                b = fill_range(int(self.right[node]))
                begin, end = min(a[0], b[0]), max(a[1], b[1])
            self.range_begin[node], self.range_end[node] = begin, end
            return begin, end
        assert fill_range(0) == (0, len(self.bounds))
        del index
        self.anchors = np.ascontiguousarray(model._anchor.detach().cpu().numpy(), dtype=np.float32)
        self.levels = np.ascontiguousarray(model._level.detach().cpu().numpy().reshape(-1), dtype=np.int32)
        self.extra = np.ascontiguousarray(model._extra_level.detach().cpu().numpy().reshape(-1), dtype=np.float32)
        self.standard_dist = np.float32(model.standard_dist.detach().cpu().item())
        self.voxel_size = np.float32(model.voxel_size)
        self.fork = np.float32(model.fork)
        self.max_level = int(model.levels) - 1
        assert model.dist2level == 'round' and len(self.bounds) == len(self.anchors)
        shift = np.float32(self.voxel_size / 2) / np.power(self.fork, self.levels, dtype=np.float32)
        self.lod_position = np.ascontiguousarray(self.anchors + shift[:, None], dtype=np.float32)
        packed = np.load(cells_path)
        nodes = packed['nodes']
        scale = (packed['hi'] - packed['lo']) / np.power(2., nodes[:, 0, None])
        lower = packed['lo'] + nodes[:, 1:] * scale
        self.boxes = np.ascontiguousarray(np.c_[lower, lower + scale], dtype=np.float64)
        self.cameras = []
        for view, domain in zip(experiment.views, experiment.rt.domains):
            self.cameras.append(dict(
                camera=np.ascontiguousarray(camera_tensor(view).detach().cpu().numpy(), dtype=np.float64),
                center=np.ascontiguousarray(view.camera_center.detach().cpu().numpy(), dtype=np.float32),
                planes=np.ascontiguousarray(camera_planes(domain), dtype=np.float64),
                w2c=np.ascontiguousarray(domain.w2c, dtype=np.float64),
                near=float(domain.near),
                resolution_scale=np.float32(view.resolution_scale),
            ))
        torch.cuda.synchronize()

    def candidates(self, frame):
        c = self.cameras[frame]
        if hasattr(self, 'canonical_radius2'):
            output = np.empty(len(self.bounds), dtype=np.uint8)
            eye = np.ascontiguousarray(c['center'], dtype=np.float64)
            rc = self.native.cpu_lod_threshold(self.canonical_position.ctypes.data,
                self.canonical_radius2.ctypes.data, self.levels.ctypes.data, eye.ctypes.data,
                float(c['resolution_scale']), len(self.bounds), self.threads_per_frame, output.ctypes.data)
            if rc: raise RuntimeError(f'canonical LoD failed: {rc}')
            return output
        if self.candidate_backend == 'native':
            output = np.empty(len(self.bounds), dtype=np.uint8)
            rc = self.native.cpu_candidates(
                self.lod_position.ctypes.data, self.levels.ctypes.data,
                self.extra.ctypes.data, c['center'].ctypes.data,
                len(self.bounds), self.standard_dist, self.fork,
                c['resolution_scale'], self.max_level, self.threads_per_frame,
                output.ctypes.data)
            if rc:
                raise RuntimeError(f'CPU LoD selector returned {rc}')
            return output
        with np.errstate(divide='ignore', invalid='ignore'):
            delta = self.lod_position - c['center']
            distance = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float32), dtype=np.float32) * c['resolution_scale']
            pred = np.log2(self.standard_dist / distance, dtype=np.float32) / np.log2(self.fork, dtype=np.float32) + self.extra
        int_level = np.clip(np.rint(pred), 0, self.max_level)
        return np.ascontiguousarray((self.levels <= int_level).astype(np.uint8))

    def visible_cells(self, frame):
        c = self.cameras[frame]
        if getattr(self, "occluder", None) is not None:
            return self.occluder.query(c["planes"][:, :4], c["w2c"], c["near"])
        return visible_leaf_boxes(self.boxes, c["planes"], c["w2c"], c["near"])

    def select(self, frame):
        start = time.perf_counter()
        c = self.cameras[frame]
        candidates = self.candidates(frame)
        candidate_end = time.perf_counter()
        visible = self.visible_cells(frame)
        eye = np.linalg.solve(c['w2c'][:3, :3], -c['w2c'][:3, 3])
        if self.hole_backend == 'native' and len(visible):
            visible = np.ascontiguousarray(visible, dtype=np.float64)
            eye = np.ascontiguousarray(eye, dtype=np.float64)
            planes = np.empty((32*len(visible), 4), dtype=np.float64)
            starts = np.zeros(len(visible)+1, dtype=np.int32)
            rc = self.native.cpu_build_holes(
                visible.ctypes.data, eye.ctypes.data, len(visible),
                self.threads_per_frame, planes.ctypes.data, starts.ctypes.data)
            if rc:
                raise RuntimeError(f'CPU hole builder returned {rc}')
            planes = np.ascontiguousarray(planes[:starts[-1]])
        else:
            chunks = [fast_hole_planes(box, eye) for box in visible]
            starts = np.zeros(len(chunks) + 1, dtype=np.int32)
            for j, chunk in enumerate(chunks):
                starts[j + 1] = starts[j] + len(chunk)
            planes = np.ascontiguousarray(np.concatenate(chunks) if chunks else np.zeros((1, 4)), dtype=np.float64)
        holes_end = time.perf_counter()
        output = np.empty(len(self.bounds), dtype=np.uint8)
        counters = np.zeros(3, dtype=np.int64)
        if self.backend == 'tree_keep':
            rc = self.native.cpu_select_tree_keep(
                self.bounds.ctypes.data, self.nodes.ctypes.data,
                self.left.ctypes.data, self.right.ctypes.data, self.order.ctypes.data,
                self.range_begin.ctypes.data, self.range_end.ctypes.data,
                c['camera'].ctypes.data, c['planes'].ctypes.data,
                candidates.ctypes.data, len(self.bounds), self.leaves,
                planes.ctypes.data, starts.ctypes.data, len(visible),
                self.threads_per_frame, output.ctypes.data, counters.ctypes.data)
        elif self.backend == 'tree':
            rc = self.native.cpu_select_tree(
                self.bounds.ctypes.data, self.nodes.ctypes.data,
                self.left.ctypes.data, self.right.ctypes.data, self.order.ctypes.data,
                c['camera'].ctypes.data, candidates.ctypes.data,
                len(self.bounds), self.leaves, planes.ctypes.data, starts.ctypes.data,
                len(visible), self.threads_per_frame, output.ctypes.data)
        else:
            rc = self.native.cpu_select(
                self.bounds.ctypes.data, c['camera'].ctypes.data, candidates.ctypes.data,
                len(self.bounds), planes.ctypes.data, starts.ctypes.data,
                len(visible), self.threads_per_frame, output.ctypes.data)
        if rc:
            raise RuntimeError(f'CPU selector returned {rc}')
        native_end = time.perf_counter()
        ids = np.ascontiguousarray(np.flatnonzero(output), dtype=np.int64)
        return ids, dict(frame=frame, selected=len(ids), candidates=int(candidates.sum()),
                         visible_cells=len(visible), wall_ms=(time.perf_counter() - start) * 1000,
                         candidate_ms=(candidate_end-start)*1000,
                         hole_setup_ms=(holes_end-candidate_end)*1000,
                         native_ms=(native_end-holes_end)*1000,
                         visited_nodes=int(counters[0]),refined_anchors=int(counters[1]),
                         kept_nodes=int(counters[2]))

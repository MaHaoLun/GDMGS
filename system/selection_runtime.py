"""Real C++/OpenMP and CUDA anchor queries, sharing one canonical LoD policy."""
import os
import ctypes
import subprocess
import threading
from pathlib import Path
import numpy as np
import torch
from bootstrap import configure
from cpu_selector import CPUSelector
from fast_geometry import visible_leaf_boxes
from fused_filter import native as hole_native, filter_ids
from gdmgs.anchor_frustum.gpu_construction import GPUThreeTrees
from holed_index import OccluderIndex
from gpu_occluders import GPUOccluders


def build_cpu():
    root = Path(__file__).resolve().parent
    output = configure() / 'cpu_select.so'
    sources = [root / 'cpu_select.cpp', root / 'hole_planes_native.cpp']
    if not output.exists() or any(p.stat().st_mtime_ns > output.stat().st_mtime_ns for p in sources):
        temporary = output.with_suffix('.building.so')
        subprocess.run([os.environ.get('CXX', 'g++'), '-O3', '-std=c++17', '-shared', '-fPIC',
                        '-fopenmp', '-ffp-contract=off', *map(str, sources), '-o', str(temporary)], check=True)
        temporary.replace(output)
    return output


class Selection:
    def __init__(self, experiment, cells, cpu_threads=1, occlusion=True, grid_level=None):
        self.e = experiment
        self.cpu = CPUSelector(experiment, cells, build_cpu(), cpu_threads)
        self.cpu.backend = 'tree_keep'
        self.cpu.hole_backend = self.cpu.candidate_backend = 'native'
        if not occlusion:
            self.cpu.boxes = np.empty((0, 6), dtype=np.float64)
        packed = np.load(cells, allow_pickle=False)
        level = int(packed['level']) if 'level' in packed else grid_level
        if level is None:
            raise ValueError('legacy cell files need an explicit occluder_level')
        nodes = packed['nodes'] if occlusion else np.empty((0,4), dtype=np.int64)
        self.cpu.occluder = OccluderIndex(packed['lo'], packed['hi'], level, nodes)
        self.occluders = GPUOccluders(packed['lo'], packed['hi'], level, nodes)
        self.gpu = GPUThreeTrees(experiment.rt.model)
        self.holes = hole_native()
        self.cpu.native.cpu_lod_threshold.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_double, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
        self.cpu.native.cpu_lod_threshold.restype = ctypes.c_int
        levels = self.cpu.levels.astype(np.float64)
        if float(self.cpu.standard_dist) <= 0 or float(self.cpu.fork) <= 1 or np.any(levels < 0) or np.any(levels > self.cpu.max_level):
            raise ValueError('invalid LoD model parameters')
        radius = float(self.cpu.standard_dist) / np.power(float(self.cpu.fork), levels-.5-self.cpu.extra.astype(np.float64))
        self.cpu.canonical_position = np.ascontiguousarray(self.cpu.lod_position, dtype=np.float64)
        self.cpu.canonical_radius2 = np.ascontiguousarray(radius*radius)
        if not np.isfinite(self.cpu.canonical_radius2).all() or np.any(self.cpu.canonical_radius2 <= 0):
            raise ValueError('LoD thresholds exceed double precision range')
        self.lod_position = torch.tensor(self.cpu.canonical_position, device=self.gpu.device)
        self.lod_radius2 = torch.tensor(self.cpu.canonical_radius2, device=self.gpu.device)
        self.lod_levels = torch.tensor(self.cpu.levels, device=self.gpu.device)
        self.lod_eyes = [torch.tensor(c['center'].astype(np.float64), device=self.gpu.device) for c in self.cpu.cameras]
        self.hole_eyes = [torch.tensor(np.linalg.solve(c['w2c'][:3,:3], -c['w2c'][:3,3]), device=self.gpu.device) for c in self.cpu.cameras]
        self.local = threading.local()
        torch.cuda.synchronize()

    def select_cpu(self, frame):
        # Native LoD + holes + indexed traversal, all query scratch private.
        return self.cpu.select(frame)[0]

    def select_gpu(self, frame):
        # Identical algebraic LoD thresholds; this predicate runs on CUDA.
        mask = self.holes.lod_mask(self.lod_position, self.lod_radius2, self.lod_levels,
                                   self.lod_eyes[frame], float(self.cpu.cameras[frame]['resolution_scale']))
        ids = torch.nonzero(mask).flatten().contiguous()
        ids = self.gpu.query(self.e.views[frame], ids, mode='gpu_bvh')[0]
        c = self.cpu.cameras[frame]
        cells = self.occluders.query(c['planes'], c['w2c'], c['near'])
        planes, counts = self.occluders.holes(cells, self.hole_eyes[frame])
        return ids[self.holes.hole_keep(self.gpu.bounds, ids, planes, counts)] if len(cells) else ids

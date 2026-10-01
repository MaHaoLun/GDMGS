"""Online 1600x900 positive-camera-z rasterization for Step 4."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict

import numpy as np
import torch

from gdmgs.mesh_index import CameraDomain, MeshIndex
from gdmgs.query.depth_pyramid import pixel_grid


@dataclass
class OnlineDepthResult:
    depth_cpu: np.ndarray | None
    depth_gpu: torch.Tensor
    coverage_cpu: np.ndarray | None
    triangle_ids: np.ndarray
    timings: Dict[str, float]
    counters: Dict[str, Any]


class OnlineProxyDepthRasterizer:
    """Same nvdiffrast projection/depth semantics as the frozen GDMGS reference.

    Full pixel depth is an explicit output. Raster completion and the D2H copy
    are separately synchronized and measured.
    """

    def __init__(self, mesh_index: MeshIndex, *, device: str = "cuda:0") -> None:
        try:
            import nvdiffrast.torch as dr
        except ImportError as exc:
            raise RuntimeError("Step 4 requires the frozen nvdiffrast dependency") from exc
        if not isinstance(mesh_index, MeshIndex):
            raise TypeError("mesh_index must be a validated MeshIndex")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("online depth requires an explicit CUDA device")
        self.mesh_index = mesh_index
        self._dr = dr
        with torch.cuda.device(self.device), torch.no_grad():
            vertices = np.column_stack((mesh_index.vertices, np.ones(len(mesh_index.vertices))))
            self._world_vertices = torch.tensor(vertices, dtype=torch.float64, device=self.device)
            self._triangles = torch.tensor(
                mesh_index.triangles.astype(np.int32), dtype=torch.int32, device=self.device
            )
            self._context = dr.RasterizeCudaContext(device=self.device)
            torch.cuda.synchronize(self.device)
        self.resident_bytes = (
            self._world_vertices.numel() * self._world_vertices.element_size()
            + self._triangles.numel() * self._triangles.element_size()
        )

    def render(
        self,
        triangle_ids: np.ndarray,
        camera_domain: CameraDomain,
        image_size: tuple[int, int],
        *,
        copy_to_cpu: bool = True,
    ) -> OnlineDepthResult:
        is_gpu = isinstance(triangle_ids, torch.Tensor)
        if is_gpu:
            if triangle_ids.dtype != torch.int64 or triangle_ids.ndim != 1 or triangle_ids.device != self.device:
                raise ValueError("GPU IDs must be CUDA int64 rows on the raster device")
        else:
            if triangle_ids.dtype != np.int64 or triangle_ids.ndim != 1:
                raise TypeError("CPU IDs must be int64 rows")
            if len(triangle_ids) and (triangle_ids[0] < 0 or triangle_ids[-1] >= len(self.mesh_index.triangles) or np.any(triangle_ids[1:] <= triangle_ids[:-1])):
                raise ValueError("CPU IDs must be sorted unique valid rows")
        camera = CameraDomain.parse(camera_domain)
        if image_size != (1600, 900):
            raise ValueError("formal Step 4 depth is frozen to 1600x900")
        padded_camera, padded_size, projection = pixel_grid(camera, image_size, tile_size=8)
        padded_width, padded_height = padded_size
        width, height = image_size
        matrix = np.ascontiguousarray(projection @ camera.w2c)
        camera_z_row = np.ascontiguousarray(camera.w2c[2])
        names = ("begin", "ids", "project", "raster", "copy_begin", "copy_end")
        events = {name: torch.cuda.Event(enable_timing=True) for name in names}
        frame_start = time.perf_counter()
        with torch.cuda.device(self.device), torch.no_grad():
            torch.cuda.synchronize(self.device)
            events["begin"].record()
            gpu_ids = triangle_ids if is_gpu else torch.tensor(triangle_ids, dtype=torch.int64, device=self.device)
            selected_faces = self._triangles.index_select(0, gpu_ids).contiguous()
            gpu_matrix = torch.tensor(matrix, dtype=torch.float64, device=self.device)
            gpu_z_row = torch.tensor(camera_z_row, dtype=torch.float64, device=self.device)
            events["ids"].record()
            if len(triangle_ids):
                clip = (self._world_vertices @ gpu_matrix.T).float().unsqueeze(0).contiguous()
                camera_z = (self._world_vertices @ gpu_z_row).float()[None, :, None].contiguous()
                if not bool(torch.isfinite(clip).all()):
                    raise ValueError("mesh clip coordinates exceed finite raster capacity")
                events["project"].record()
                raster, _ = self._dr.rasterize(
                    self._context,
                    clip,
                    selected_faces,
                    (padded_height, padded_width),
                    grad_db=False,
                )
                interpolated, _ = self._dr.interpolate(camera_z, raster, selected_faces)
                depth = interpolated[0, :, :, 0]
                hit = raster[0, :, :, 3] > 0
                valid = hit & torch.isfinite(depth) & (depth >= camera.near)
                if np.isfinite(camera.far):
                    valid &= depth <= camera.far
            else:
                events["project"].record()
                depth = torch.full(
                    (padded_height, padded_width), float("inf"), dtype=torch.float32, device=self.device
                )
                valid = torch.zeros_like(depth, dtype=torch.bool)
            valid[height:, :] = False
            valid[:, width:] = False
            depth = torch.where(valid, depth, float("inf"))[:height, :width].contiguous()
            coverage = valid[:height, :width].contiguous()
            events["raster"].record()
            sync_start = time.perf_counter()
            events["raster"].synchronize()
            sync_ms = (time.perf_counter() - sync_start) * 1000.0
            cpu_depth = None
            cpu_coverage = None
            d2h_wait_ms = 0.0
            depth_d2h_ms = 0.0
            if copy_to_cpu:
                cpu_depth = torch.empty((height, width), dtype=torch.float32, pin_memory=True)
                cpu_coverage = torch.empty((height, width), dtype=torch.bool, pin_memory=True)
                events["copy_begin"].record()
                cpu_depth.copy_(depth, non_blocking=True)
                cpu_coverage.copy_(coverage, non_blocking=True)
                events["copy_end"].record()
                d2h_wait_start = time.perf_counter()
                events["copy_end"].synchronize()
                d2h_wait_ms = (time.perf_counter() - d2h_wait_start) * 1000.0
                depth_d2h_ms = events["copy_begin"].elapsed_time(events["copy_end"])
        timings = {
            "triangle_ids_h2d_ms": events["begin"].elapsed_time(events["ids"]),
            "depth_projection_gpu_ms": events["ids"].elapsed_time(events["project"]),
            "depth_raster_gpu_ms": events["project"].elapsed_time(events["raster"]),
            "depth_sync_ms": sync_ms,
            "depth_d2h_ms": depth_d2h_ms,
            "depth_d2h_wait_ms": d2h_wait_ms,
            "depth_total_ms": (time.perf_counter() - frame_start) * 1000.0,
        }
        return OnlineDepthResult(
            depth_cpu=cpu_depth.numpy() if cpu_depth is not None else None,
            depth_gpu=depth,
            coverage_cpu=cpu_coverage.numpy() if cpu_coverage is not None else None,
            triangle_ids=triangle_ids,
            timings=timings,
            counters={
                "camera": camera.camera_id,
                "input_triangles": int(len(triangle_ids)),
                "total_triangles": int(len(self.mesh_index.triangles)),
                "finite_pixels": int(cpu_coverage.sum().item()) if cpu_coverage is not None else None,
                "infinite_pixels": (
                    int(width * height - cpu_coverage.sum().item())
                    if cpu_coverage is not None
                    else None
                ),
                "width": width,
                "height": height,
                "near": float(padded_camera.near),
                "far": float(padded_camera.far),
                "pixel_center": 0.5,
                "depth_semantics": "positive OpenCV camera-z; +inf background",
                "resident_mesh_bytes": int(self.resident_bytes),
                "copied_to_cpu": bool(copy_to_cpu),
            },
        )

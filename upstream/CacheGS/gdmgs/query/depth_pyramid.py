"""Discrete native-pixel mesh occlusion with a max-depth tile hierarchy.

This implements pixel-center visibility, not continuous angular coverage.
Unknown pixels are +inf, so one unknown sample keeps its whole tile unknown.
Mesh geometry is resident; each query still uploads its actual triangle IDs,
rasterizes the selected mesh, and transfers the resulting tile depths.
"""

from dataclasses import dataclass
from math import lcm
from time import perf_counter

import numpy as np

from gdmgs.mesh_index import CameraDomain, MeshIndex, MeshQueryResult


def pixel_grid(camera_domain, image_size, tile_size=8):
    """Return padded camera, (width,height), and positive-camera-z clip matrix.

    The original calibrated pixel spacing is retained. Only the right/bottom
    angular endpoints expand. Raster memory row 0 explicitly maps to original
    camera pixel row 0 through increasing NDC y; no later vertical flip occurs.
    """
    camera = CameraDomain.parse(camera_domain)
    if len(image_size) != 2 or any(type(n) is not int or n <= 0 for n in image_size):
        raise ValueError("image_size must be positive integer (width,height)")
    if type(tile_size) is not int or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    width, height = image_size
    multiple = lcm(tile_size, 8)  # CUDA rasterizer dimensions support blocks of 8.
    padded_width = ((width + multiple - 1) // multiple) * multiple
    padded_height = ((height + multiple - 1) // multiple) * multiple
    xmin, xmax, ymin, ymax = camera.angular_domain
    fx, fy = width / (xmax - xmin), height / (ymax - ymin)
    cx, cy = -xmin * fx, -ymin * fy
    padded_domain = (xmin, (padded_width - cx) / fx,
                     ymin, (padded_height - cy) / fy)
    padded_camera = CameraDomain(camera.w2c, padded_domain, camera.near,
                                  camera.far, camera.camera_id)
    projection = np.zeros((4, 4), dtype=np.float64)
    projection[0, 0] = 2 * fx / padded_width
    projection[0, 2] = 2 * cx / padded_width - 1
    projection[1, 1] = 2 * fy / padded_height
    projection[1, 2] = 2 * cy / padded_height - 1
    if np.isfinite(camera.far):
        projection[2, 2] = (camera.far + camera.near) / (camera.far - camera.near)
        projection[2, 3] = -2 * camera.far * camera.near / (camera.far - camera.near)
    else:
        projection[2, 2], projection[2, 3] = 1, -2 * camera.near
    projection[3, 2] = 1
    return padded_camera, (padded_width, padded_height), projection


@dataclass(frozen=True)
class PixelDepthORI:
    depth_bounds: np.ndarray
    angular_domain: tuple
    mesh_token: str
    camera_domain: CameraDomain
    counters: dict
    elapsed_ms: float
    image_size: tuple
    original_image_size: tuple
    tile_size: int
    timings: dict
    pixel_depth: object = None
    pixel_coverage: object = None
    complete: bool = True
    gpu_depth_bounds: object = None

    @property
    def camera_id(self):
        return self.camera_domain.camera_id


@dataclass
class GPUPixelDepthORI:
    """Device-resident tile result; collect logging after the measured frame."""
    gpu_depth_bounds: object
    angular_domain: tuple
    mesh_token: str
    camera_domain: CameraDomain
    image_size: tuple
    original_image_size: tuple
    tile_size: int
    timings: dict
    _counts: dict
    _valid_mask: object
    _invalid_mask: object
    _events: object
    depth_bounds: object = None
    pixel_depth: object = None
    pixel_coverage: object = None
    elapsed_ms: object = None
    complete: bool = True

    @property
    def camera_id(self):
        return self.camera_domain.camera_id

    def collect_counters(self):
        if self._events is not None:
            events = self._events
            events["download"].synchronize()
            for before, after, name in (
                ("begin", "upload", "upload_and_gather_ms"),
                ("upload", "project", "projection_ms"),
                ("project", "raster", "raster_and_interpolate_ms"),
                ("raster", "reduce", "tile_reduction_ms"),
                ("reduce", "download", "download_ms"),
            ):
                self.timings[name] = events[before].elapsed_time(events[after])
            self.timings["gpu_elapsed_ms"] = events["begin"].elapsed_time(events["download"])
            self.elapsed_ms = self.timings["gpu_elapsed_ms"]
            self._events = None
        if self._valid_mask is not None:
            import torch
            counts = torch.stack((self._valid_mask.sum(),
                                  torch.isfinite(self.gpu_depth_bounds).sum(),
                                  self._invalid_mask.sum())).cpu().tolist()
            width, height = self.image_size
            self._counts.update(covered_pixels=int(counts[0]),
                                unknown_pixels=width * height - int(counts[0]),
                                covered_cells=int(counts[1]),
                                unknown_cells=self.gpu_depth_bounds.numel() - int(counts[1]),
                                invalid_raster_hits=int(counts[2]))
            self._valid_mask = self._invalid_mask = None
        return self._counts

    @property
    def counters(self):
        return self.collect_counters()


class MeshDepthRasterizer:
    """CUDA mesh rasterization followed by full-tile maximum camera-z depth."""

    def __init__(self, mesh_index, *, device="cuda:0", tile_size=8):
        import torch
        try:
            import nvdiffrast.torch as dr
        except ImportError as exc:
            raise RuntimeError("The scoped nvdiffrast CUDA dependency is required") from exc
        if not isinstance(mesh_index, MeshIndex):
            raise TypeError("mesh_index must be a validated MeshIndex")
        if type(tile_size) is not int or tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("MeshDepthRasterizer requires an explicit CUDA device")
        if (len(mesh_index.vertices) > np.iinfo(np.int32).max
                or len(mesh_index.triangles) > np.iinfo(np.int32).max):
            raise OverflowError("Mesh exceeds nvdiffrast's int32 index representation")
        start = perf_counter()
        self.mesh_index = mesh_index
        self.mesh_token = mesh_index.mesh_token
        self.device = device
        self.tile_size = tile_size
        self._torch, self._dr = torch, dr
        with torch.cuda.device(device), torch.no_grad():
            vertices = np.column_stack((mesh_index.vertices, np.ones(len(mesh_index.vertices))))
            self._world_vertices = torch.tensor(vertices, dtype=torch.float64, device=device)
            self._triangles = torch.tensor(mesh_index.triangles.astype(np.int32), device=device)
            self._context = dr.RasterizeCudaContext(device=device)
            torch.cuda.synchronize(device)
        self.initialization_ms = (perf_counter() - start) * 1000
        self.resident_bytes = self._world_vertices.numel() * 8 + self._triangles.numel() * 4
        if len(mesh_index.vertices):
            lower, upper = mesh_index.vertices.min(axis=0), mesh_index.vertices.max(axis=0)
            self._world_corners = np.array([
                [upper[0] if corner & 1 else lower[0],
                 upper[1] if corner & 2 else lower[1],
                 upper[2] if corner & 4 else lower[2], 1.0]
                for corner in range(8)], dtype=np.float64)
        else:
            self._world_corners = np.empty((0, 4), dtype=np.float64)

    def build(self, query_result, camera_domain, image_size, *, keep_pixel_depth=False,
              download_tiles=True):
        """Measure all necessary uploads, raster work, reductions and CPU transfer.

        Pixel outputs are optional diagnostics. Their full CPU transfer is
        separately reported and excluded from the production elapsed_ms.
        """
        start = perf_counter()
        torch, dr = self._torch, self._dr
        from gdmgs.mesh_index.gpu_index import GPUMeshQueryResult
        if isinstance(query_result, GPUMeshQueryResult) or not download_tiles:
            return self._build_gpu(query_result, camera_domain, image_size,
                                   keep_pixel_depth=keep_pixel_depth,
                                   download_tiles=download_tiles, started=start)
        camera = CameraDomain.parse(camera_domain)
        if not isinstance(query_result, MeshQueryResult) or not query_result.complete:
            raise ValueError("A complete MeshQueryResult is required")
        if query_result.mesh_token != self.mesh_token:
            raise ValueError("Mesh query and GPU mesh belong to different scenes")
        if not query_result.camera_domain.equivalent(camera):
            raise ValueError("Mesh query belongs to a different camera")
        ids = query_result.triangle_ids
        if ids.dtype != np.int64 or ids.ndim != 1 or (ids.size and (
                ids[0] < 0 or ids[-1] >= len(self.mesh_index.triangles)
                or np.any(ids[1:] <= ids[:-1]))):
            raise ValueError("Triangle IDs must be valid sorted unique int64 rows")
        padded_camera, padded_size, projection = pixel_grid(camera, image_size, self.tile_size)
        padded_width, padded_height = padded_size
        width, height = image_size
        matrix = np.ascontiguousarray(projection @ camera.w2c)
        camera_z_row = np.ascontiguousarray(camera.w2c[2])
        timings = {}
        events = {name: torch.cuda.Event(enable_timing=True) for name in
                  ("begin", "upload", "project", "raster", "reduce", "download")}
        with torch.cuda.device(self.device), torch.no_grad():
            torch.cuda.synchronize(self.device)
            events["begin"].record()
            gpu_ids = torch.tensor(ids.copy(), dtype=torch.int64, device=self.device)
            gpu_matrix = torch.tensor(matrix, dtype=torch.float64, device=self.device)
            gpu_z_row = torch.tensor(camera_z_row.copy(), dtype=torch.float64, device=self.device)
            selected_faces = self._triangles.index_select(0, gpu_ids).contiguous()
            events["upload"].record()
            if len(ids):
                clip = (self._world_vertices @ gpu_matrix.T).float().unsqueeze(0).contiguous()
                camera_z = (self._world_vertices @ gpu_z_row).float()[None, :, None].contiguous()
                if not bool(torch.isfinite(clip).all()):
                    raise ValueError("Mesh clip coordinates exceed finite rasterizer capacity")
                events["project"].record()
                raster, _ = dr.rasterize(self._context, clip, selected_faces,
                                          (padded_height, padded_width), grad_db=False)
                interpolated, _ = dr.interpolate(camera_z, raster, selected_faces)
                pixel_depth = interpolated[0, :, :, 0]
                hit = raster[0, :, :, 3] > 0
                valid = hit & torch.isfinite(pixel_depth) & (pixel_depth >= camera.near)
                if np.isfinite(camera.far):
                    valid &= pixel_depth <= camera.far
                invalid_hit = (hit & ~valid).sum()
                events["raster"].record()
            else:
                events["project"].record()
                pixel_depth = torch.full((padded_height, padded_width), float("inf"), device=self.device)
                valid = torch.zeros_like(pixel_depth, dtype=torch.bool)
                invalid_hit = torch.zeros((), dtype=torch.int64, device=self.device)
                events["raster"].record()
            # Keep every padded sample Unknown even if mesh extends beyond the
            # original image. Interior pixels retain their original calibration.
            valid[height:, :] = False
            valid[:, width:] = False
            pixel_depth = torch.where(valid, pixel_depth, float("inf"))
            # A small explicit upward numerical margin protects the float32
            # interpolated value; the acceptance contract remains image based.
            upward = 32 * torch.finfo(torch.float32).eps * (pixel_depth.abs() + 1)
            upper = torch.nextafter(pixel_depth + upward,
                                     torch.full_like(pixel_depth, float("inf")))
            tiles = upper.reshape(padded_height // self.tile_size, self.tile_size,
                                   padded_width // self.tile_size, self.tile_size).amax(dim=(1, 3))
            counts = torch.stack((valid.sum(), torch.isfinite(tiles).sum(), invalid_hit))
            events["reduce"].record()
            depth_bounds = tiles.to(dtype=torch.float64, device="cpu").numpy()
            count_values = counts.cpu().tolist()
            events["download"].record()
            torch.cuda.synchronize(self.device)
            elapsed_ms = (perf_counter() - start) * 1000
            for before, after, name in (
                ("begin", "upload", "upload_and_gather_ms"),
                ("upload", "project", "projection_ms"),
                ("project", "raster", "raster_and_interpolate_ms"),
                ("raster", "reduce", "tile_reduction_ms"),
                ("reduce", "download", "download_ms"),
            ):
                timings[name] = events[before].elapsed_time(events[after])
            diagnostic_depth = diagnostic_coverage = None
            diagnostic_start = perf_counter()
            if keep_pixel_depth:
                diagnostic_depth = pixel_depth[:height, :width].cpu().numpy().copy()
                diagnostic_coverage = valid[:height, :width].cpu().numpy().copy()
                torch.cuda.synchronize(self.device)
            timings["diagnostic_download_ms"] = (perf_counter() - diagnostic_start) * 1000 if keep_pixel_depth else 0.0
        depth_bounds.flags.writeable = False
        counters = {
            "ori_definition": "native_pixel_depth_tiles_v1",
            "coverage_contract": "native_pixel_centers_max_depth_tiles",
            "continuous_coverage_certificate": False,
            "selected_triangles": len(ids), "original_pixels": width * height,
            "padded_pixels": padded_width * padded_height,
            "original_image_size": [width, height],
            "padded_image_size": [padded_width, padded_height],
            "covered_pixels": int(count_values[0]),
            "unknown_pixels": padded_width * padded_height - int(count_values[0]),
            "covered_cells": int(count_values[1]),
            "unknown_cells": depth_bounds.size - int(count_values[1]),
            "invalid_raster_hits": int(count_values[2]),
            "upload_bytes": ids.nbytes + matrix.nbytes + camera_z_row.nbytes,
            "download_bytes": depth_bounds.nbytes + 3 * 8,
            "resident_mesh_bytes": self.resident_bytes,
            "tile_size": self.tile_size, "pixel_center": 0.5,
            "sidedness": "two_sided", "depth_rounding_relative": float(32 * np.finfo(np.float32).eps),
        }
        return PixelDepthORI(depth_bounds, padded_camera.angular_domain, self.mesh_token,
                              padded_camera, counters, elapsed_ms, padded_size,
                              image_size, self.tile_size, timings,
                              diagnostic_depth, diagnostic_coverage, gpu_depth_bounds=tiles)

    def _build_gpu(self, query_result, camera_domain, image_size, *, keep_pixel_depth,
                   download_tiles, started):
        """Optional same pixel algorithm without full triangle/tile CPU round trips."""
        from gdmgs.mesh_index.gpu_index import GPUMeshQueryResult
        torch, dr = self._torch, self._dr
        camera = CameraDomain.parse(camera_domain)
        if not isinstance(query_result, (MeshQueryResult, GPUMeshQueryResult)) or not query_result.complete:
            raise ValueError("A complete CPU or GPU mesh query result is required")
        if query_result.mesh_token != self.mesh_token or not query_result.camera_domain.equivalent(camera):
            raise ValueError("Mesh query belongs to a different mesh or camera")
        ids = query_result.triangle_ids
        gpu_query = isinstance(query_result, GPUMeshQueryResult)
        if gpu_query:
            query_result.validate()
            if ids.dtype != torch.int64 or ids.ndim != 1 or ids.device != self._world_vertices.device:
                raise ValueError("GPU triangle IDs must be int64 on the rasterizer device")
        elif (ids.dtype != np.int64 or ids.ndim != 1 or (ids.size and (
                ids[0] < 0 or ids[-1] >= len(self.mesh_index.triangles)
                or np.any(ids[1:] <= ids[:-1])))):
            raise ValueError("CPU triangle IDs must be valid sorted unique int64 rows")
        padded_camera, padded_size, projection = pixel_grid(camera, image_size, self.tile_size)
        padded_width, padded_height = padded_size
        width, height = image_size
        matrix = np.ascontiguousarray(projection @ camera.w2c)
        camera_z_row = np.ascontiguousarray(camera.w2c[2])
        # Affine extrema occur at AABB corners. This O(1) CPU check avoids a
        # per-frame GPU->CPU finite test across all resident mesh vertices.
        corner_clip = self._world_corners @ matrix.T
        capacity = float(np.nextafter(np.float32(np.finfo(np.float32).max), np.float32(0)))
        if not np.isfinite(corner_clip).all() or np.any(np.abs(corner_clip) > capacity):
            raise ValueError("Mesh clip coordinates exceed finite float32 raster capacity")
        events = {name: torch.cuda.Event(enable_timing=True) for name in
                  ("begin", "upload", "project", "raster", "reduce", "download")}
        timings = {"diagnostic_download_ms": 0.0, "gpu_elapsed_ms": None}
        with torch.cuda.device(self.device), torch.no_grad():
            events["begin"].record()
            gpu_ids = ids if gpu_query else torch.tensor(ids.copy(), dtype=torch.int64, device=self.device)
            gpu_matrix = torch.tensor(matrix, dtype=torch.float64, device=self.device)
            gpu_z_row = torch.tensor(camera_z_row.copy(), dtype=torch.float64, device=self.device)
            selected_faces = self._triangles.index_select(0, gpu_ids).contiguous()
            events["upload"].record()
            if len(ids):
                clip = (self._world_vertices @ gpu_matrix.T).float().unsqueeze(0).contiguous()
                camera_z = (self._world_vertices @ gpu_z_row).float()[None, :, None].contiguous()
                events["project"].record()
                raster, _ = dr.rasterize(self._context, clip, selected_faces,
                                          (padded_height, padded_width), grad_db=False)
                interpolated, _ = dr.interpolate(camera_z, raster, selected_faces)
                pixel_depth = interpolated[0, :, :, 0]
                hit = raster[0, :, :, 3] > 0
                valid = hit & torch.isfinite(pixel_depth) & (pixel_depth >= camera.near)
                if np.isfinite(camera.far):
                    valid &= pixel_depth <= camera.far
                invalid_hit = hit & ~valid
                events["raster"].record()
            else:
                events["project"].record()
                pixel_depth = torch.full((padded_height, padded_width), float("inf"), device=self.device)
                valid = torch.zeros_like(pixel_depth, dtype=torch.bool)
                invalid_hit = torch.zeros_like(valid)
                events["raster"].record()
            valid[height:, :] = False
            valid[:, width:] = False
            pixel_depth = torch.where(valid, pixel_depth, float("inf"))
            upward = 32 * torch.finfo(torch.float32).eps * (pixel_depth.abs() + 1)
            upper = torch.nextafter(pixel_depth + upward, torch.full_like(pixel_depth, float("inf")))
            tiles = upper.reshape(padded_height // self.tile_size, self.tile_size,
                                   padded_width // self.tile_size, self.tile_size).amax(dim=(1, 3)).contiguous()
            events["reduce"].record()
            depth_bounds = None
            if download_tiles:
                depth_bounds = tiles.to(dtype=torch.float64, device="cpu").numpy()
                depth_bounds.flags.writeable = False
            events["download"].record()
            timings["host_enqueue_ms"] = (perf_counter() - started) * 1000
            diagnostic_depth = diagnostic_coverage = None
            if keep_pixel_depth:
                diagnostic_start = perf_counter()
                diagnostic_depth = pixel_depth[:height, :width].cpu().numpy().copy()
                diagnostic_coverage = valid[:height, :width].cpu().numpy().copy()
                timings["diagnostic_download_ms"] = (perf_counter() - diagnostic_start) * 1000
        id_upload_bytes = 0 if gpu_query else ids.nbytes
        counters = {
            "ori_definition": "native_pixel_depth_tiles_v1",
            "coverage_contract": "native_pixel_centers_max_depth_tiles",
            "continuous_coverage_certificate": False,
            "actual_device": str(self._world_vertices.device),
            "triangle_query_device": str(ids.device) if gpu_query else "cpu",
            "mesh_token": self.mesh_token, "camera_id": camera.camera_id,
            "source_query_index_token": query_result.index_token if gpu_query else None,
            "source_triangle_id_policy": "sorted_unique_global_triangle_rows",
            "selected_triangles": len(ids), "original_pixels": width * height,
            "padded_pixels": padded_width * padded_height,
            "original_image_size": [width, height], "padded_image_size": list(padded_size),
            "covered_pixels": None, "unknown_pixels": None,
            "covered_cells": None, "unknown_cells": None, "invalid_raster_hits": None,
            "upload_bytes": id_upload_bytes + matrix.nbytes + camera_z_row.nbytes,
            "triangle_id_upload_bytes": id_upload_bytes,
            "download_bytes": tiles.numel() * 8 if download_tiles else 0,
            "logging_download_bytes": 3 * 8,
            "resident_mesh_bytes": self.resident_bytes, "tile_size": self.tile_size,
            "pixel_center": 0.5, "sidedness": "two_sided",
            "depth_rounding_relative": float(32 * np.finfo(np.float32).eps),
        }
        return GPUPixelDepthORI(tiles, padded_camera.angular_domain, self.mesh_token,
                               padded_camera, padded_size, image_size, self.tile_size,
                               timings, counters, valid, invalid_hit, events,
                               depth_bounds, diagnostic_depth, diagnostic_coverage)

"""Offline whole-image source-depth checks for mesh triangle interiors.

This deletes geometry contradicted by frozen model ED. It is neither a
ground-truth surface proof nor an image-quality acceptance test.
"""
import numpy as np

from .depth_mesh import valid_depth


class NativePixelCaster:
    """Read-only Open3D CPU raycasting over every native pixel center."""

    def __init__(self, vertices, faces, *, cpu_threads=8):
        import open3d as o3d
        if (vertices.dtype != np.float64 or vertices.ndim != 2 or vertices.shape[1] != 3 or
                faces.dtype != np.int64 or faces.ndim != 2 or faces.shape[1] != 3):
            raise ValueError("Expected float64[V,3] vertices and int64[F,3] faces")
        if (not np.isfinite(vertices).all() or
                (faces.size and (faces.min() < 0 or faces.max() >= len(vertices)))):
            raise ValueError("Invalid triangle table")
        if len(vertices) >= np.iinfo(np.uint32).max or len(faces) >= np.iinfo(np.uint32).max:
            raise OverflowError("Mesh exceeds Open3D raycasting uint32 representation")
        if type(cpu_threads) is not int or cpu_threads <= 0:
            raise ValueError("A positive CPU thread count is required")
        float_vertices = vertices.astype(np.float32)
        if not np.isfinite(float_vertices).all():
            raise OverflowError("Mesh coordinates exceed finite raycasting float32 representation")
        self.o3d, self.face_count = o3d, len(faces)
        self.cpu_threads = cpu_threads
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=cpu_threads)
        if len(faces):
            mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(float_vertices),
                o3d.core.Tensor(faces.astype(np.uint32)))
            self.scene.add_triangles(mesh)

    def conflicts(self, frame, *, relative_tolerance=.02, alpha_threshold=.995,
                  block_rows=128, near=.01, all_intersections=False):
        """Return local face IDs contradicted by any high-confidence pixel.

        The mesh stays unchanged for an entire pass. Callers remove the union of
        conflicting faces only after every source view has been checked.
        """
        if (not np.isfinite(relative_tolerance) or relative_tolerance < 0 or
                not 0 < alpha_threshold <= 1 or type(block_rows) is not int or block_rows <= 0 or
                not np.isfinite(near) or near <= 0):
            raise ValueError("Invalid native-pixel consistency settings")
        height, width = frame.depth.shape
        valid_source = valid_depth(frame, alpha_threshold)
        marked = np.zeros(self.face_count, dtype=bool)
        counters = {"native_pixels": height * width, "cast_pixels": 0,
                    "observed_pixels": int(valid_source.sum()), "mesh_hits": 0,
                    "paired_pixels": 0, "conflicting_pixels": 0, "conflicting_faces": 0}
        if not self.face_count:
            # An empty scene has no geometry for any of these native pixels.
            counters["cast_pixels"] = height * width
            return np.empty(0, dtype=np.int64), counters
        c2w = np.linalg.inv(frame.world_to_camera)
        k = frame.intrinsics
        columns = (np.arange(width, dtype=np.float64) + .5 - k[0, 2]) / k[0, 0]
        center = c2w[:3, 3]
        for first in range(0, height, block_rows):
            last = min(first + block_rows, height)
            rows = (np.arange(first, last, dtype=np.float64) + .5 - k[1, 2]) / k[1, 1]
            camera_rays = np.empty((last - first, width, 3), dtype=np.float64)
            camera_rays[:, :, 0], camera_rays[:, :, 1] = columns[None], rows[:, None]
            camera_rays[:, :, 2] = 1.
            directions = camera_rays @ c2w[:3, :3].T
            # Start on the actual positive-camera-z near plane, so geometry
            # behind that plane cannot hide a subsequent valid surface hit.
            origins = center + near * directions
            rays = np.concatenate([origins, directions], axis=-1).astype(np.float32)
            if not np.isfinite(rays).all():
                raise OverflowError("Camera rays exceed finite raycasting representation")
            result = self.scene.cast_rays(self.o3d.core.Tensor(rays), nthreads=self.cpu_threads)
            distance = result["t_hit"].numpy().astype(np.float64)
            primitive = result["primitive_ids"].numpy()
            finite = np.isfinite(distance)
            if np.any(finite & (primitive >= self.face_count)):
                raise RuntimeError("Finite ray hit has an invalid triangle ID")
            # Recompute camera-z from the actual float32 ray representation;
            # t alone is not exactly camera-z after casting/near-plane shift.
            zrow = frame.world_to_camera[2]
            origin_z = rays[:, :, :3].astype(np.float64) @ zrow[:3] + zrow[3]
            direction_z = rays[:, :, 3:].astype(np.float64) @ zrow[:3]
            mesh_z = origin_z + distance * direction_z
            hit = finite & np.isfinite(mesh_z) & (mesh_z > 0)
            observed = valid_source[first:last]
            paired = hit & observed
            source_depth = frame.depth[first:last].astype(np.float64)
            counters["cast_pixels"] += (last - first) * width
            counters["mesh_hits"] += int(hit.sum())
            counters["paired_pixels"] += int(paired.sum())
            cutoff = source_depth * (1. - relative_tolerance)
            if all_intersections:
                # A later hit cannot violate the depth limit if the nearest
                # hit is already beyond it. Only this necessary query is
                # filtered; every native pixel above was actually cast. The
                # upward screening margin expands work near numerical ties,
                # while the deletion comparison remains exactly mesh_z < cutoff.
                margin = 64 * np.finfo(np.float32).eps * (np.abs(mesh_z) + 1)
                screen = paired & (mesh_z <= cutoff + margin)
                selected_rays = np.flatnonzero(screen)
                counters["all_intersection_screened_pixels"] = counters.get("all_intersection_screened_pixels", 0) + len(selected_rays)
                if not len(selected_rays):
                    continue
                selected = np.ascontiguousarray(rays.reshape(-1, 6)[selected_rays])
                result = self.scene.list_intersections(self.o3d.core.Tensor(selected), nthreads=self.cpu_threads)
                all_primitive = result["primitive_ids"].numpy()
                all_distance = result["t_hit"].numpy().astype(np.float64)
                ray_ids = result["ray_ids"].numpy()
                splits = result["ray_splits"].numpy()
                if (len(splits) != len(selected_rays) + 1 or splits[0] != 0 or
                        splits[-1] != len(all_distance) or len(all_primitive) != len(all_distance) or
                        len(ray_ids) != len(all_distance) or np.any(splits[1:] < splits[:-1]) or
                        np.any(ray_ids >= len(selected_rays)) or np.any(all_primitive >= self.face_count) or
                        not np.isfinite(all_distance).all()):
                    raise RuntimeError("Incomplete or invalid all-intersection buffers")
                full_ray_ids = selected_rays[ray_ids]
                all_z = origin_z.reshape(-1)[full_ray_ids] + all_distance * direction_z.reshape(-1)[full_ray_ids]
                conflict = np.isfinite(all_z) & (all_z > 0) & (all_z < cutoff.reshape(-1)[full_ray_ids])
                marked[all_primitive[conflict]] = True
                counters["conflicting_pixels"] += int(np.unique(full_ray_ids[conflict]).size)
                counters["intersections"] = counters.get("intersections", 0) + len(all_distance)
                counters["conflicting_intersections"] = counters.get("conflicting_intersections", 0) + int(conflict.sum())
            else:
                conflict = paired & (mesh_z < cutoff)
                marked[primitive[conflict]] = True
                counters["conflicting_pixels"] += int(conflict.sum())
        if counters["cast_pixels"] != counters["native_pixels"]:
            raise RuntimeError("Incomplete native-pixel traversal")
        ids = np.flatnonzero(marked).astype(np.int64)
        counters["conflicting_faces"] = len(ids)
        return ids, counters

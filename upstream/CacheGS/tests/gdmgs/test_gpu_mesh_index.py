"""GPU scan/leaf-cluster equivalence and direct pixel-depth handoff."""

import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gdmgs.mesh_index import MeshIndex
from gdmgs.mesh_index.gpu_index import camera_planes, leaf_layout


def camera():
    return {"w2c": np.eye(4, dtype=np.float64),
            "angular_domain": (-1., 1., -1., 1.), "near": .01,
            "far": 100., "camera_id": "gpu-mesh-fixture"}


class GPUMeshPreparationTests(unittest.TestCase):
    def test_planes_use_complete_calibrated_halfspaces(self):
        domain = camera()
        planes = camera_planes(domain)
        points = np.array([[0, 0, 2, 1], [3, 0, 2, 1], [0, 0, -.1, 1],
                           [0, 0, 101, 1]], dtype=np.float64)
        sides = points @ planes[:, :4].T
        np.testing.assert_array_equal(np.all(sides >= 0, axis=1), [True, False, False, False])
        self.assertTrue(np.all(planes[:, 4:] > 0))
        domain["far"] = float("inf")
        self.assertEqual(camera_planes(domain).shape, (5, 8))

    def test_leaf_mapping_matches_every_actual_triangle_once(self):
        rng = np.random.default_rng(987)
        vertices = rng.uniform(-10, 10, (210, 3))
        faces = np.arange(210, dtype=np.int64).reshape(-1, 3)
        for method in ("median", "binned_sah"):
            mesh = MeshIndex(vertices, faces, method=method, leaf_size=5)
            bounds, mapping, counts = leaf_layout(mesh)
            self.assertEqual(int(counts.sum()), len(faces))
            self.assertTrue(np.all(counts <= 5))
            for face_id, leaf in enumerate(mapping):
                triangle = vertices[faces[face_id]]
                self.assertTrue(np.all(triangle >= bounds[leaf, 0]))
                self.assertTrue(np.all(triangle <= bounds[leaf, 1]))


@unittest.skipUnless(os.environ.get("GDMGS_TEST_GPU_INDEX") == "1", "Requires leased CUDA GPU-index run")
class GPUMeshRuntimeTests(unittest.TestCase):
    def test_finite_coordinates_with_overflowing_products_remain_candidates(self):
        from gdmgs.mesh_index.gpu_index import GPUMeshIndex
        s = np.sqrt(.5)
        domain = camera()
        domain["w2c"][:3, :3] = [[s, s, 0], [0, 0, 1], [s, -s, 0]]
        domain["angular_domain"] = (-3., 3., -3., 3.)
        domain["far"] = float("inf")
        vertices = np.array([[1.79e308, 1.5e308, 0], [-1.5e308, -1.79e308, 0],
                             [-1.5e308, -1.79e308, 1e306]], dtype=np.float64)
        mesh = MeshIndex(vertices, np.array([[0, 1, 2]], dtype=np.int64))
        np.testing.assert_array_equal(mesh.query(domain).triangle_ids, [0])
        gpu = GPUMeshIndex(mesh)
        for backend in ("brute_force", "bvh"):
            np.testing.assert_array_equal(gpu.query(domain, backend=backend).triangle_ids.cpu(), [0])

    def test_random_full_and_leaf_queries_are_equal_and_cover_cpu_reference(self):
        from gdmgs.mesh_index.gpu_index import GPUMeshIndex
        rng = np.random.default_rng(147)
        vertices = rng.uniform(-10, 10, (3000, 3))
        faces = np.arange(3000, dtype=np.int64).reshape(-1, 3)
        mesh = MeshIndex(vertices, faces, method="binned_sah", leaf_size=8, mesh_token="random")
        gpu = GPUMeshIndex(mesh)
        for trial in range(8):
            domain = camera()
            domain["w2c"][:3, 3] = rng.uniform(-4, 4, 3)
            full, indexed = gpu.query(domain, backend="brute_force"), gpu.query(domain)
            a, b = full.triangle_ids.cpu().numpy(), indexed.triangle_ids.cpu().numpy()
            np.testing.assert_array_equal(a, b)
            self.assertTrue(set(mesh.query(domain, backend="brute_force").triangle_ids) <= set(a))
            self.assertEqual(full.collect_counters()["tested_triangles"], len(faces))
            self.assertEqual(indexed.collect_counters()["visited_nodes"], 0)
            self.assertEqual(indexed.counters["index_structure"], "gpu_bvh_leaf_cluster_scan")
            self.assertEqual(indexed.counters["triangle_id_download_bytes"], 0)

    def test_cluster_pruning_is_active_without_losing_long_external_triangles(self):
        from gdmgs.mesh_index.gpu_index import GPUMeshIndex
        vertices = [[-1, -1, 2], [100, -1, 2], [-1, 100, 2]]
        for i in range(100):
            vertices.extend([[100 + i, 0, 2], [100 + i, 1, 2], [101 + i, 0, 2]])
        vertices = np.array(vertices, dtype=np.float64)
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
        mesh = MeshIndex(vertices, faces, leaf_size=4)
        gpu = GPUMeshIndex(mesh)
        result = gpu.query(camera())
        self.assertIn(0, result.triangle_ids.cpu().tolist())
        np.testing.assert_array_equal(result.triangle_ids.cpu(), gpu.query(camera(), backend="brute_force").triangle_ids.cpu())
        self.assertLess(result.counters["tested_triangles"], len(faces))

    def test_direct_gpu_pixel_tiles_equal_existing_cpu_query_path(self):
        import torch
        from gdmgs.mesh_index.gpu_index import GPUMeshIndex
        from gdmgs.query.depth_pyramid import MeshDepthRasterizer
        vertices = np.array([[-6, -6, 3], [6, -6, 3], [6, 6, 3], [-6, 6, 3]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mesh = MeshIndex(vertices, faces, mesh_token="gpu-wall")
        gpu = GPUMeshIndex(mesh)
        raster = MeshDepthRasterizer(mesh)
        cpu_ori = raster.build(mesh.query(camera()), camera(), (53, 39))
        gpu_ori = raster.build(gpu.query(camera()), camera(), (53, 39), download_tiles=False)
        self.assertIsNone(gpu_ori.depth_bounds)
        self.assertEqual(gpu_ori.gpu_depth_bounds.dtype, torch.float32)
        self.assertTrue(gpu_ori.gpu_depth_bounds.is_cuda)
        np.testing.assert_array_equal(gpu_ori.gpu_depth_bounds.cpu().double().numpy(), cpu_ori.depth_bounds)
        counters = gpu_ori.collect_counters()
        self.assertEqual(counters["download_bytes"], 0)
        self.assertEqual(counters["triangle_id_upload_bytes"], 0)
        self.assertEqual(counters["covered_cells"], cpu_ori.counters["covered_cells"])

    def test_empty_gpu_query_and_modified_ids(self):
        from gdmgs.mesh_index.gpu_index import GPUMeshIndex
        from gdmgs.query.depth_pyramid import MeshDepthRasterizer
        mesh = MeshIndex(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64))
        gpu = GPUMeshIndex(mesh)
        query = gpu.query(camera())
        self.assertEqual(query.triangle_ids.numel(), 0)
        raster = MeshDepthRasterizer(mesh)
        result = raster.build(query, camera(), (16, 16), download_tiles=False)
        self.assertEqual(result.collect_counters()["covered_pixels"], 0)
        query.triangle_ids = query.triangle_ids.clone()
        with self.assertRaises(ValueError):
            query.validate()


if __name__ == "__main__":
    unittest.main()

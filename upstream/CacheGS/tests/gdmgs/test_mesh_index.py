"""Native mesh-owner regression tests; independent oracles live separately."""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gdmgs.mesh_index import MeshIndex
from gdmgs.query.ori import build_ori


def camera(**kwargs):
    return dict(w2c=np.eye(4, dtype=np.float64),
                angular_domain=(-1.0, 1.0, -1.0, 1.0), near=0.01,
                far=float("inf"), camera_id="fixture", **kwargs)


def quads(rectangles, depth=2.0):
    vertices, faces = [], []
    for xl, xr, yl, yr in rectangles:
        start = len(vertices)
        vertices.extend([[xl*depth, yl*depth, depth], [xr*depth, yl*depth, depth],
                         [xr*depth, yr*depth, depth], [xl*depth, yr*depth, depth]])
        faces.extend([[start, start+1, start+2], [start, start+2, start+3]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


class MeshIndexTests(unittest.TestCase):
    def test_complete_object_split_and_random_camera_agreement(self):
        rng = np.random.default_rng(781)
        vertices = rng.uniform(-10, 10, (900, 3))
        triangles = np.arange(900, dtype=np.int64).reshape(-1, 3)
        for method in ("median", "binned_sah"):
            mesh = MeshIndex(vertices, triangles, method=method, leaf_size=5)
            layout = mesh.inspect_layout()
            np.testing.assert_array_equal(np.sort(layout["triangle_refs"]), np.arange(300))
            for _ in range(8):
                view = camera()
                view["w2c"][:3, 3] = rng.uniform(-4, 4, 3)
                view["far"] = 8.0
                full = mesh.query(view, backend="brute_force")
                indexed = mesh.query(view)
                self.assertEqual(full.counters["tested_triangles"], 300)
                np.testing.assert_array_equal(indexed.triangle_ids, full.triangle_ids)

    def test_long_triangle_survives_centroid_outside_frustum(self):
        vertices = np.array([[-1, -1, 2], [100, -1, 2], [-1, 100, 2]], dtype=np.float64)
        mesh = MeshIndex(vertices, np.array([[0, 1, 2]], dtype=np.int64))
        np.testing.assert_array_equal(mesh.query(camera()).triangle_ids, [0])

    def test_sparse_frustum_actually_prunes_bvh(self):
        vertices, faces = quads([(i*10, i*10+2, -2, 2) for i in range(100)])
        mesh = MeshIndex(vertices, faces, leaf_size=4)
        bvh = mesh.query(camera())
        full = mesh.query(camera(), backend="brute_force")
        np.testing.assert_array_equal(bvh.triangle_ids, full.triangle_ids)
        self.assertLess(bvh.counters["tested_triangles"], full.counters["tested_triangles"])

    def test_wall_union_fills_shared_diagonal_and_matches_scan(self):
        mesh = MeshIndex(*quads([(-2, 2, -2, 2)]), mesh_token="wall")
        a = build_ori(mesh, mesh.query(camera()), camera(), (8, 8))
        b = build_ori(mesh, mesh.query(camera(), backend="brute_force"), camera(), (8, 8))
        np.testing.assert_array_equal(a.depth_bounds, np.full((8, 8), 2.0))
        np.testing.assert_array_equal(a.depth_bounds, b.depth_bounds)
        self.assertFalse(a.depth_bounds.flags.writeable)

    def test_microscopic_gap_stays_unknown(self):
        # Four quads cover the rectangle apart from a hole much smaller than a pixel.
        e = 1e-10
        mesh = MeshIndex(*quads([(-2, -e, -2, 2), (e, 2, -2, 2),
                                (-e, e, -2, -e), (-e, e, e, 2)]))
        ori = build_ori(mesh, mesh.query(camera()), camera(), (1, 1))
        self.assertTrue(np.isinf(ori.depth_bounds[0, 0]))

    def test_partial_thin_triangle_does_not_certify(self):
        mesh = MeshIndex(*quads([(-2, 2, -1e-4, 1e-4)]))
        self.assertTrue(np.isinf(build_ori(mesh, mesh.query(camera()), camera(), (1, 1)).depth_bounds).all())

    def test_sloped_wall_uses_largest_depth_not_center(self):
        # x/z and y/z remain a square, while depth rises from 2 to 4.
        vertices = np.array([[-4, -4, 2], [8, -8, 4], [8, 8, 4], [-4, 4, 2]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mesh = MeshIndex(vertices, faces)
        ori = build_ori(mesh, mesh.query(camera()), camera(), (1, 1))
        self.assertEqual(ori.depth_bounds[0, 0], 4.0)

    def test_near_crossing_triangle_is_clipped_not_divided_behind_camera(self):
        vertices = np.array([[-10, -10, 2], [10, -10, 2], [0, 10, -2]], dtype=np.float64)
        mesh = MeshIndex(vertices, np.array([[0, 1, 2]], dtype=np.int64))
        ori = build_ori(mesh, mesh.query(camera()), camera(), (4, 4))
        self.assertFalse(np.isnan(ori.depth_bounds).any())
        self.assertTrue((ori.depth_bounds[np.isfinite(ori.depth_bounds)] >= 0.01).all())

    def test_two_sided_winding_and_empty_mesh(self):
        vertices, faces = quads([(-2, 2, -2, 2)])
        mesh = MeshIndex(vertices, faces[:, ::-1])
        self.assertTrue(np.isfinite(build_ori(mesh, mesh.query(camera()), camera(), (2, 2)).depth_bounds).all())
        empty = MeshIndex(np.empty((0, 3), np.float64), np.empty((0, 3), np.int64))
        self.assertEqual(empty.query(camera()).triangle_ids.size, 0)
        self.assertTrue(np.isinf(build_ori(empty, empty.query(camera()), camera(), (2, 2)).depth_bounds).all())

    def test_strict_inputs_and_camera_identity(self):
        vertices, faces = quads([(-2, 2, -2, 2)])
        with self.assertRaises(TypeError):
            MeshIndex(vertices.astype(np.float32), faces)
        with self.assertRaises(TypeError):
            MeshIndex(vertices, faces.astype(np.int32))
        with self.assertRaises(ValueError):
            MeshIndex(vertices * np.nan, faces)
        mesh = MeshIndex(vertices, faces)
        result = mesh.query(camera())
        changed = camera()
        changed["w2c"][0, 3] = 0.1
        with self.assertRaises(ValueError):
            build_ori(mesh, result, changed)
        with self.assertRaises(ValueError):
            build_ori(mesh, replace(result, complete=False), camera())
        with self.assertRaises(ValueError):
            build_ori(mesh, np.array([0, 0], dtype=np.int64), camera())

    def test_persistence_keeps_topology_and_rejects_tampering(self):
        mesh = MeshIndex(*quads([(-2, 2, -2, 2), (5, 7, 5, 7)]),
                         leaf_size=1, method="binned_sah", mesh_token="fixture-run")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.npz"
            mesh.save(path)
            loaded = MeshIndex.load(path, mesh_token="fixture-run")
            np.testing.assert_array_equal(mesh.inspect_layout()["nodes"], loaded.inspect_layout()["nodes"])
            np.testing.assert_array_equal(mesh.query(camera()).triangle_ids, loaded.query(camera()).triangle_ids)
            with self.assertRaises(ValueError):
                MeshIndex.load(path, mesh_token="other-scene")
            with np.load(path, allow_pickle=False) as record:
                arrays = {key: record[key] for key in record.files}
            arrays["triangle_refs"][0] = arrays["triangle_refs"][1]
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                MeshIndex.load(path)


if __name__ == "__main__":
    unittest.main()

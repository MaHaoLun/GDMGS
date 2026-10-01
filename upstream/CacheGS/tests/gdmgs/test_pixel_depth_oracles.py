"""Independent pixel-ray oracle for the discrete native-image ORI contract.

CUDA tests require an explicit GDMGS_TEST_PIXEL_GPU=1 lease. The CPU oracle
uses neither production projection nor rasterization/reduction helpers.
"""
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def calibrated_camera(*, near=.1, far=10., transform=None):
    return {"w2c": np.eye(4, dtype=np.float64) if transform is None else transform,
            "angular_domain": (-.8, 1.2, -.6, .9), "near": near, "far": far,
            "camera_id": "independent-discrete-camera"}


def pixel_ray_depth(vertices, faces, camera, image_size):
    """Double-precision Moller-Trumbore intersections on calibrated .5 rays.

    Ray directions are not normalized, so intersection t is camera-z.
    Accepted samples are discrete pixel centers, never continuous coverage.
    """
    width, height = image_size
    xmin, xmax, ymin, ymax = camera["angular_domain"]
    camera_to_world = np.linalg.inv(camera["w2c"])
    origin = camera_to_world[:3, 3]
    output = np.full((height, width), np.inf, dtype=np.float64)
    for row in range(height):
        v = ymin + (row + .5) * (ymax - ymin) / height
        for column in range(width):
            u = xmin + (column + .5) * (xmax - xmin) / width
            direction = camera_to_world[:3, :3] @ np.array([u, v, 1.])
            for ids in faces:
                a, b, c = vertices[ids]
                edge1, edge2 = b - a, c - a
                p = np.cross(direction, edge2)
                determinant = float(edge1 @ p)
                if abs(determinant) < 1e-12:
                    continue
                displacement = origin - a
                first = float(displacement @ p) / determinant
                q = np.cross(displacement, edge1)
                second = float(direction @ q) / determinant
                distance = float(edge2 @ q) / determinant
                if (first >= -1e-10 and second >= -1e-10 and first + second <= 1 + 1e-10
                        and camera["near"] <= distance <= camera["far"]):
                    output[row, column] = min(output[row, column], distance)
    return output


def independent_tiles(depth, tile_size):
    """Literal per-tile maximum after padding every unavailable pixel Unknown."""
    height, width = depth.shape
    multiple = math.lcm(tile_size, 8)
    padded_height = math.ceil(height / multiple) * multiple
    padded_width = math.ceil(width / multiple) * multiple
    result = np.full((padded_height // tile_size, padded_width // tile_size), np.inf)
    for row in range(result.shape[0]):
        for column in range(result.shape[1]):
            top, left = row * tile_size, column * tile_size
            if top + tile_size <= height and left + tile_size <= width:
                result[row, column] = np.max(depth[top:top + tile_size, left:left + tile_size])
    return result


def angular_rectangles(rectangles, *, transform=None):
    """Build independent quads from (umin,umax,vmin,vmax,camera-z)."""
    vertices, faces = [], []
    transform = np.eye(4) if transform is None else np.linalg.inv(transform)
    for left, right, bottom, top, depth in rectangles:
        first = len(vertices)
        for u, v in ((left, bottom), (right, bottom), (right, top), (left, top)):
            point = transform @ np.array([u * depth, v * depth, depth, 1.])
            vertices.append(point[:3])
        faces.extend(((first, first + 1, first + 2), (first, first + 2, first + 3)))
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3), np.asarray(faces, dtype=np.int64).reshape(-1, 3)


class DiscreteOracleReferenceTests(unittest.TestCase):
    def test_asymmetric_camera_rays_hit_known_front_patch(self):
        vertices, faces = angular_rectangles([(-2, 2, -2, 2, 3), (-.8, -.3, -.6, -.2, 1)])
        depth = pixel_ray_depth(vertices, faces, calibrated_camera(), (16, 16))
        self.assertEqual(depth[0, 0], 1.)
        self.assertAlmostEqual(depth[-1, 0], 3.)
        self.assertAlmostEqual(depth[0, -1], 3.)
        self.assertTrue(np.isfinite(depth).all())

    def test_unknown_sample_and_unavailable_padding_propagate(self):
        depth = np.arange(1, 13 * 11 + 1, dtype=np.float64).reshape(11, 13)
        tile = independent_tiles(depth, 8)
        self.assertEqual(tile[0, 0], depth[7, 7])
        self.assertTrue(np.isinf(tile[1]).all() and np.isinf(tile[:, 1]).all())
        depth[3, 4] = np.inf
        self.assertTrue(np.isinf(independent_tiles(depth, 8)).all())


@unittest.skipUnless(os.environ.get("GDMGS_NATIVE_DIR"), "Requires the independent CPU anchor native build")
class DiscreteAnchorCPUOracleTests(unittest.TestCase):
    def test_full_screen_support_reaches_unknown_neighbor_beyond_center_pixel(self):
        from gdmgs.unified_index import AnchorIndex
        pixel_depth = np.full((64, 64), 2., dtype=np.float64)
        # The center pixel is in tile 4; a physically nearby pixel lies in
        # tile 3. This remains inside the v3 alpha-support expansion without
        # assuming the old, unnecessary complete-raster-tile padding.
        pixel_depth[32, 30] = np.inf
        self.assertTrue(np.isfinite(pixel_depth[32, 32]))
        tiles = independent_tiles(pixel_depth, 8)
        positions = np.array([[0., 0., 5.], [-3., 0., 5.], [0., 0., 1.]], dtype=np.float64)
        offsets = np.zeros((3, 1, 3), dtype=np.float64)
        scales = np.full((3, 6), .005, dtype=np.float64)
        scales[:, :3] = 1.
        index = AnchorIndex.from_arrays(positions, offsets, scales, leaf_capacity=1)
        fov = np.array([2, 1, 0], dtype=np.int64)
        for mode in ("linear", "tree"):
            result = index.query(fov, tiles, np.eye(4, dtype=np.float64),
                                 (-1., 1., -1., 1.), (64, 64), mode=mode)
            # Center-covered row 0 must stay because its padded screen support
            # reaches the neighboring hole. Row 1 is wholly behind covered
            # pixels and must actually be removed; row 2 is in front of wall.
            np.testing.assert_array_equal(result.selected_anchor_ids, [2, 0])


@unittest.skipUnless(os.environ.get("GDMGS_TEST_PIXEL_GPU") == "1",
                     "Requires explicit leased CUDA pixel-oracle run")
class DiscretePixelGPUOracleTests(unittest.TestCase):
    def assert_raster_matches_rays(self, vertices, faces, camera, image_size=(16, 16), tile_size=8):
        from gdmgs.mesh_index import MeshIndex
        from gdmgs.query.depth_pyramid import MeshDepthRasterizer
        expected = pixel_ray_depth(vertices, faces, camera, image_size)
        mesh = MeshIndex(vertices, faces, leaf_size=2, mesh_token="independent-pixel-mesh")
        query = mesh.query(camera, backend="bvh")
        rasterizer = MeshDepthRasterizer(mesh, device="cuda:0", tile_size=tile_size)
        ori = rasterizer.build(query, camera, image_size, keep_pixel_depth=True)
        self.assertFalse(ori.counters["continuous_coverage_certificate"])
        self.assertEqual(tuple(ori.original_image_size), image_size)
        np.testing.assert_array_equal(ori.pixel_coverage, np.isfinite(expected))
        np.testing.assert_array_equal(np.isfinite(ori.pixel_depth), np.isfinite(expected))
        valid = np.isfinite(expected)
        np.testing.assert_allclose(ori.pixel_depth[valid], expected[valid], rtol=2e-5, atol=2e-5)
        expected_tiles = independent_tiles(expected, tile_size)
        actual_pixel_tiles = independent_tiles(ori.pixel_depth.astype(np.float64), tile_size)
        np.testing.assert_array_equal(np.isfinite(ori.depth_bounds), np.isfinite(expected_tiles))
        finite = np.isfinite(expected_tiles)
        self.assertTrue(np.all(ori.depth_bounds[finite] >= actual_pixel_tiles[finite]),
                        "A max-depth tile underbounded one of its actual covered pixels")
        np.testing.assert_allclose(ori.depth_bounds[finite], expected_tiles[finite], rtol=4e-5, atol=4e-5)
        self.assertGreaterEqual(ori.elapsed_ms, 0.)
        for field in ("upload_and_gather_ms", "projection_ms", "raster_and_interpolate_ms",
                      "tile_reduction_ms", "download_ms"):
            self.assertIn(field, ori.timings)
            self.assertGreaterEqual(ori.timings[field], 0.)
        return ori

    def test_asymmetric_orientation_camera_translation_and_nearest_layers(self):
        angle = .31
        matrix = np.array([[math.cos(angle), 0, math.sin(angle), .3], [0, 1, 0, -.4],
                           [-math.sin(angle), 0, math.cos(angle), .7], [0, 0, 0, 1]], dtype=np.float64)
        domain = calibrated_camera(transform=matrix)
        vertices, faces = angular_rectangles([(-2, 2, -2, 2, 3), (-.8, -.3, -.6, -.2, 1)], transform=matrix)
        ori = self.assert_raster_matches_rays(vertices, faces, domain)
        self.assertLess(ori.pixel_depth[0, 0], 1.001)
        self.assertGreater(ori.pixel_depth[-1, 0], 2.99)
        self.assertGreater(ori.pixel_depth[0, -1], 2.99)

    def test_slanted_plane_uses_perspective_correct_camera_z(self):
        vertices = []
        for u, v in ((-2., -2.), (2., -2.), (2., 2.), (-2., 2.)):
            depth = 3. / (1 + .2 * u - .1 * v)
            vertices.append((u * depth, v * depth, depth))
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self.assert_raster_matches_rays(np.asarray(vertices, dtype=np.float64), faces, calibrated_camera())

    def test_near_far_clipping_does_not_hide_valid_background(self):
        camera = calibrated_camera(near=.1, far=4.)
        vertices, faces = angular_rectangles([(-2, 2, -2, 2, 3), (-.8, .1, -.6, .2, .05),
                                             (.1, 1.2, .2, .9, 8.)])
        ori = self.assert_raster_matches_rays(vertices, faces, camera)
        np.testing.assert_allclose(ori.pixel_depth, 3., atol=1e-5)

    def test_triangle_crossing_near_plane(self):
        camera = calibrated_camera(near=.3, far=4.)
        vertices, faces = angular_rectangles([(-2, 2, -2, 2, 3)])
        # Slanted foreground spans camera-z .1..1 and is clipped at .3.
        foreground = np.array([[-.1, -.08, .1], [.8, -.4, 1.], [-.7, .9, 1.]], dtype=np.float64)
        faces = np.concatenate((faces, np.array([[4, 5, 6]], dtype=np.int64)))
        self.assert_raster_matches_rays(np.concatenate((vertices, foreground)), faces, camera)

    def test_pixel_center_hole_makes_complete_tile_unknown(self):
        camera = calibrated_camera()
        xmin, xmax, ymin, ymax = camera["angular_domain"]
        u = xmin + 5.5 * (xmax - xmin) / 16
        v = ymin + 2.5 * (ymax - ymin) / 16
        half = .02
        vertices, faces = angular_rectangles([(xmin, u-half, ymin, ymax, 2), (u+half, xmax, ymin, ymax, 2),
                                             (u-half, u+half, ymin, v-half, 2), (u-half, u+half, v+half, ymax, 2)])
        ori = self.assert_raster_matches_rays(vertices, faces, camera)
        self.assertFalse(ori.pixel_coverage[2, 5])
        self.assertTrue(np.isinf(ori.depth_bounds[0, 0]))
        self.assertTrue(np.isfinite(ori.depth_bounds[1, 1]))

    def test_between_pixel_hole_is_not_a_continuous_coverage_claim(self):
        camera = calibrated_camera()
        camera["angular_domain"] = (-1., 1., -1., 1.)
        small = .01
        vertices, faces = angular_rectangles([(-1, -small, -1, 1, 2), (small, 1, -1, 1, 2),
                                             (-small, small, -1, -small, 2), (-small, small, small, 1, 2)])
        ori = self.assert_raster_matches_rays(vertices, faces, camera, image_size=(8, 8))
        self.assertTrue(np.isfinite(ori.depth_bounds).all())
        self.assertFalse(ori.counters["continuous_coverage_certificate"])

    def test_padding_stays_unknown_even_when_mesh_extends_outside_image(self):
        vertices, faces = angular_rectangles([(-4, 4, -4, 4, 2)])
        ori = self.assert_raster_matches_rays(vertices, faces, calibrated_camera(), image_size=(13, 11))
        self.assertEqual(tuple(ori.image_size), (16, 16))
        self.assertTrue(np.isfinite(ori.depth_bounds[0, 0]))
        self.assertTrue(np.isinf(ori.depth_bounds[1]).all() and np.isinf(ori.depth_bounds[:, 1]).all())

    def test_empty_query_is_unknown(self):
        vertices, faces = angular_rectangles([])
        ori = self.assert_raster_matches_rays(vertices, faces, calibrated_camera())
        self.assertTrue(np.isinf(ori.depth_bounds).all())
        self.assertEqual(ori.counters["covered_pixels"], 0)


if __name__ == "__main__":
    unittest.main()

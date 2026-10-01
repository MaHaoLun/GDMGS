"""Calibration/unknown semantics for discrete pixel ORI, separate from rays."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gdmgs.query.depth_pyramid import pixel_grid


class PixelGridTests(unittest.TestCase):
    def test_padding_keeps_all_original_pixel_centers_and_spacing(self):
        camera = {"w2c": np.eye(4), "angular_domain": (-.8, .8, -.4, .4),
                  "near": .01, "far": 100, "camera_id": "fixture"}
        padded, size, projection = pixel_grid(camera, (1600, 891), 8)
        self.assertEqual(size, (1600, 896))
        self.assertEqual(padded.angular_domain[:3], (-.8, .8, -.4))
        fx, fy = 1600 / 1.6, 891 / .8
        for row, column in [(0, 0), (890, 1599), (322, 967)]:
            p = np.array([(column + .5 - 800) / fx * 3,
                          (row + .5 - 891 / 2) / fy * 3, 3, 1])
            clip = projection @ p
            ndc = clip[:2] / clip[3]
            pixels = (ndc + 1) / 2 * size
            np.testing.assert_allclose(pixels, [column + .5, row + .5], atol=1e-12)

    def test_positive_z_near_far_clip_mapping(self):
        camera = {"w2c": np.eye(4), "angular_domain": (-1, 1, -1, 1),
                  "near": .1, "far": 20}
        _, _, projection = pixel_grid(camera, (31, 29), 4)
        for depth, expected in [(.1, -1), (20, 1)]:
            clip = projection @ np.array([0, 0, depth, 1])
            self.assertAlmostEqual(clip[2] / clip[3], expected)

    def test_padded_tile_angles_match_pixel_tile_boundaries(self):
        camera = {"w2c": np.eye(4), "angular_domain": (-.6, .8, -.3, .5)}
        padded, size, _ = pixel_grid(camera, (53, 39), 8)
        x0, x1, y0, y1 = padded.angular_domain
        self.assertAlmostEqual((x1 - x0) / size[0], 1.4 / 53)
        self.assertAlmostEqual((y1 - y0) / size[1], .8 / 39)
        self.assertEqual(size, (56, 40))


if __name__ == "__main__":
    unittest.main()

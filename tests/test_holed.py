import unittest

import numpy as np

from gdmgs.geometry import hole_planes, inside_hole
from gdmgs.occluders import OccluderIndex, anchor_walk


class HoledQueryTests(unittest.TestCase):
    def test_near_plane_and_single_hole_pruning(self):
        index = OccluderIndex([0, 0, 0], [2, 2, 2], 1, [(1, 0, 0, 0)])
        # Camera looks along +x; both anchor bounds meet the broad frustum.
        eye = np.array([-2., .5, .5])
        w2c = np.eye(4)
        w2c[2, :3], w2c[2, 3] = [1, 0, 0], 2
        planes = np.array([[1, 0, 0, 3], [-1, 0, 0, 5],
                           [0, 1, 0, 3], [0, -1, 0, 5],
                           [0, 0, 1, 3], [0, 0, -1, 5]], dtype=float)
        cells = index.query(planes, w2c, .01)
        self.assertEqual(len(cells), 1)
        anchors = np.array([[1.45, .45, .45, 1.55, .55, .55],
                            [1.45, 1.45, .45, 1.55, 1.55, .55]])
        nodes = np.array([[1.45, .45, .45, 1.55, 1.55, .55], *anchors])
        result, diag = anchor_walk(anchors, nodes, [-1, -1, -1],
                                   [-1, -1, -1], [0, 1],
                                   [[0, 2], [0, 1], [1, 2]], planes, cells, eye)
        # A leaf index with two records refines them independently.
        self.assertEqual(result.tolist(), [1])
        self.assertEqual(diag['holes'], 1)

    def test_near_crossing_full_cell_is_not_used(self):
        index = OccluderIndex([0, 0, 0], [2, 2, 2], 1, [(1, 0, 0, 0)])
        w2c = np.eye(4)
        w2c[2, :3], w2c[2, 3] = [1, 0, 0], 0
        planes = np.array([[1, 0, 0, 1], [-1, 0, 0, 5]], dtype=float)
        self.assertEqual(len(index.query(planes, w2c, .5)), 0)

    def test_full_parent_is_split_at_near_plane(self):
        index = OccluderIndex([0, 0, 0], [2, 2, 2], 2, [(1, 0, 0, 0)])
        w2c = np.eye(4)
        w2c[2, :3], w2c[2, 3] = [1, 0, 0], 0
        planes = np.array([[1, 0, 0, 1], [-1, 0, 0, 5]], dtype=float)
        cells = index.query(planes, w2c, .4)
        self.assertEqual(len(cells), 4)
        self.assertTrue(np.all(cells[:, 0] == .5))

    def test_hole_box_complete_and_partial(self):
        hole = hole_planes(np.array([0, 0, 0, 1, 1, 1]), [-2, .5, .5])
        boxes = np.array([[1.45, .45, .45, 1.55, .55, .55],
                          [1.45, 1.45, .45, 1.55, 1.55, .55],
                          [1.45, .45, .45, 1.55, 1.45, .55]])
        self.assertEqual(inside_hole(boxes, hole).tolist(), [True, False, False])


if __name__ == '__main__':
    unittest.main()

"""Contract tests for complete-bundle, bounded sliding cache semantics."""
import unittest

import torch

from gaussian_renderer.raster_batch import BundleMetadata, NeuralGaussianBatch
from gdmgs.cache.full_bundle_cache import CacheIdentity
from gdmgs.cache.sliding_window_cache import SlidingWindowCache


def ids(values):
    return torch.tensor(values, dtype=torch.int64)


def fake_decode(frame):
    def decode(request_ids, levels):
        counts = (request_ids != 0).long()
        owners = request_ids.repeat_interleave(counts)
        row_levels = levels.repeat_interleave(counts)
        rows = owners.numel()
        xyz = torch.stack((owners.float(), torch.full((rows,), float(frame)),
                           torch.zeros(rows)), dim=1)
        return NeuralGaussianBatch(
            anchor_indices=request_ids,
            xyz=xyz,
            color=torch.ones((rows, 3)),
            opacity=torch.ones((rows, 1)),
            scaling=torch.ones((rows, 3)),
            rotation=torch.ones((rows, 4)),
            selection_mask=torch.ones(rows, dtype=torch.bool),
            sh_degree=0,
            bundle_metadata=BundleMetadata(
                request_anchor_ids=request_ids,
                request_level_ids=levels,
                counts=counts,
                offsets=torch.cat((counts.new_zeros(1), counts.cumsum(0))),
                row_owner_ids=owners,
                row_owner_levels=row_levels,
                row_offset_slots=torch.zeros(rows, dtype=torch.int64),
            ),
        )
    return decode


def cache(mode, capacity=16):
    return SlidingWindowCache(
        identity=CacheIdentity("scene", "model", "renderer", "table", "trace"),
        anchor_levels=torch.zeros(16, dtype=torch.int64),
        capacity_rows=capacity, n_offsets=1, mode=mode, audit=True,
    )


class SlidingWindowTests(unittest.TestCase):
    def test_lazy_age_one_expires_old_source(self):
        c = cache("slide_lazy_age1")
        _, s0 = c.resolve(frame_id=0, anchor_ids=ids([1, 2, 3]), decode=fake_decode(0))
        b1, s1 = c.resolve(frame_id=1, anchor_ids=ids([1, 2, 4]), decode=fake_decode(1))
        b2, s2 = c.resolve(frame_id=2, anchor_ids=ids([1, 2, 4]), decode=fake_decode(2))
        self.assertEqual((s0["hit_anchors"], s1["hit_age1_anchors"], s2["hit_age1_anchors"]),
                         (0, 2, 1))
        self.assertEqual((s0["decoded_anchors"], s1["decoded_anchors"], s2["decoded_anchors"]),
                         (3, 1, 2))
        self.assertEqual(b1.xyz[:, 1].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(b2.xyz[:, 1].tolist(), [2.0, 2.0, 1.0])
        self.assertEqual(s2["source_age_max"], 1)
        self.assertLessEqual(s2["resident_generations"], 2)

    def test_age_two_is_separate_and_latest_wins(self):
        c = cache("slide_lazy_age2")
        c.resolve(frame_id=0, anchor_ids=ids([1, 2, 3]), decode=fake_decode(0))
        c.resolve(frame_id=1, anchor_ids=ids([1, 4]), decode=fake_decode(1))
        b2, s2 = c.resolve(frame_id=2, anchor_ids=ids([2, 4]), decode=fake_decode(2))
        self.assertEqual(s2["hit_age1_anchors"], 1)
        self.assertEqual(s2["hit_age2_anchors"], 1)
        self.assertEqual(s2["decoder_calls"], 0)
        self.assertEqual(b2.xyz[:, 1].tolist(), [0.0, 1.0])
        self.assertLessEqual(len(c.generations), 2)

    def test_eager_window_advances_every_frame(self):
        c = cache("slide_eager_age1")
        b0, s0 = c.resolve(frame_id=0, anchor_ids=ids([1, 2]),
                           next_ids=ids([2, 3]), decode=fake_decode(0))
        b1, s1 = c.resolve(frame_id=1, anchor_ids=ids([2, 3]),
                           next_ids=ids([3, 4]), decode=fake_decode(1))
        b2, s2 = c.resolve(frame_id=2, anchor_ids=ids([3, 4]),
                           next_ids=None, decode=fake_decode(2))
        self.assertEqual((s0["decoded_anchors"], s1["decoded_anchors"], s2["decoded_anchors"]),
                         (3, 2, 0))
        self.assertEqual((s1["hit_age1_anchors"], s2["hit_age1_anchors"]), (2, 2))
        self.assertEqual(b0.xyz[:, 1].tolist(), [0.0, 0.0])
        self.assertEqual(b1.xyz[:, 1].tolist(), [0.0, 0.0])
        self.assertEqual(b2.xyz[:, 1].tolist(), [1.0, 1.0])

    def test_zero_row_descriptor_and_whole_bundle_capacity(self):
        c = cache("slide_lazy_age1", capacity=2)
        _, s0 = c.resolve(frame_id=0, anchor_ids=ids([0, 1, 2, 3]), decode=fake_decode(0))
        self.assertEqual(s0["output_rows"], 3)
        self.assertEqual(s0["resident_rows"], 2)
        self.assertTrue(torch.equal(c.generations[-1].batch.anchor_indices, ids([0, 1, 2])))
        _, s1 = c.resolve(frame_id=1, anchor_ids=ids([0, 3]), decode=fake_decode(1))
        self.assertEqual(s1["empty_hits"], 1)
        self.assertEqual(s1["miss_anchors"], 1)
        self.assertLessEqual(s1["resident_rows"], 2)

    def test_bad_requests_and_reset(self):
        c = cache("slide_lazy_age1")
        with self.assertRaises(ValueError):
            c.resolve(frame_id=0, anchor_ids=ids([2, 1]), decode=fake_decode(0))
        with self.assertRaises(ValueError):
            c.resolve(frame_id=0, anchor_ids=ids([1, 1]), decode=fake_decode(0))
        c.resolve(frame_id=0, anchor_ids=ids([1]), decode=fake_decode(0))
        with self.assertRaises(ValueError):
            c.resolve(frame_id=0, anchor_ids=ids([1]), decode=fake_decode(0))
        c.reset()
        _, stats = c.resolve(frame_id=0, anchor_ids=ids([1]), decode=fake_decode(0))
        self.assertEqual(stats["hit_anchors"], 0)


if __name__ == "__main__":
    unittest.main()

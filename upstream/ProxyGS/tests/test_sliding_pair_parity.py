"""Parity and age-two contract checks for the sliding pair directory."""
import unittest

import torch

from gdmgs.cache.full_bundle_cache import CacheIdentity
from gdmgs.cache.sliding_pair_parity_cache import SlidingPairParityCache
from gdmgs.cache.temporal_bundle_cache_v3 import TemporalBundleCache
from test_sliding_window2 import fake_decode, ids


def identity():
    return CacheIdentity("scene", "model", "renderer", "table", "trace")


def parity(max_age, capacity=16):
    return SlidingPairParityCache(
        identity=identity(), anchor_levels=torch.zeros(16, dtype=torch.int64),
        capacity_rows=capacity, n_offsets=1, max_age=max_age, audit=True)


def exact(a, b):
    fields = ("xyz", "color", "opacity", "scaling", "rotation",
              "anchor_indices", "selection_mask")
    metadata = ("request_anchor_ids", "request_level_ids", "counts", "offsets",
                "row_owner_ids", "row_owner_levels", "row_offset_slots")
    return (a.sh_degree == b.sh_degree
            and all(torch.equal(getattr(a, n), getattr(b, n)) for n in fields)
            and all(torch.equal(getattr(a.bundle_metadata, n),
                                getattr(b.bundle_metadata, n)) for n in metadata))


class SlidingPairParityTests(unittest.TestCase):
    def test_age_one_matches_pair2_exactly_with_two_calls(self):
        p = parity(1)
        control = TemporalBundleCache(
            identity=identity(), anchor_levels=torch.zeros(16, dtype=torch.int64),
            capacity_rows=16, n_offsets=1, max_age=1,
            fast_handoff=True, prefix_decode=True, audit=True)
        sets = [ids([1, 2]), ids([2, 3]), ids([3, 4]), ids([4, 5])]
        calls_p = calls_c = 0
        for frame, selected in enumerate(sets):
            if frame % 2 == 0:
                b_p, s_p = p.refresh(frame_id=frame, anchor_ids=selected,
                                     next_ids=sets[frame + 1], decode=fake_decode(frame))
                b_c, s_c = control.resolve(frame_id=frame, anchor_ids=selected,
                                           next_ids=sets[frame + 1], decode=fake_decode(frame))
            else:
                b_p, s_p = p.consume(frame_id=frame, anchor_ids=selected,
                                     decode=fake_decode(frame))
                b_c, s_c = control.resolve(frame_id=frame, anchor_ids=selected,
                                           decode=fake_decode(frame))
            self.assertTrue(exact(b_p, b_c), frame)
            calls_p += s_p["decoder_calls"]
            calls_c += s_c["decoder_calls"]
        self.assertEqual((calls_p, calls_c), (2, 2))

    def test_age_two_crosses_pair_boundary_without_extra_call(self):
        p = parity(2)
        p.refresh(frame_id=0, anchor_ids=ids([1, 2]),
                  next_ids=ids([2, 3]), decode=fake_decode(0))
        p.consume(frame_id=1, anchor_ids=ids([2, 3]), decode=fake_decode(1))
        b2, s2 = p.refresh(frame_id=2, anchor_ids=ids([2, 3]),
                           next_ids=ids([3, 4]), decode=fake_decode(2))
        b3, s3 = p.consume(frame_id=3, anchor_ids=ids([3, 4]), decode=fake_decode(3))
        self.assertEqual(s2["hit_age2_anchors"], 2)
        self.assertEqual((s2["decoder_calls"], s3["decoder_calls"]), (1, 0))
        self.assertEqual(b2.xyz[:, 1].tolist(), [0.0, 0.0])
        self.assertEqual(b3.xyz[:, 1].tolist(), [2.0, 2.0])

    def test_zero_row_descriptor_and_capacity_fallback_counted(self):
        p = parity(1, capacity=1)
        _, s0 = p.refresh(frame_id=0, anchor_ids=ids([0, 1]),
                          next_ids=ids([0, 1, 2]), decode=fake_decode(0))
        _, s1 = p.consume(frame_id=1, anchor_ids=ids([0, 1, 2]),
                          decode=fake_decode(1))
        self.assertEqual(s0["resident_rows"], 1)
        self.assertEqual(s0["evicted_anchors"], 1)
        self.assertEqual(s1["decoder_calls"], 1)
        self.assertEqual(s1["miss_anchors"], 1)

    def test_invalid_requests(self):
        with self.assertRaises(ValueError):
            parity(3)
        p = parity(1)
        with self.assertRaises(ValueError):
            p.refresh(frame_id=1, anchor_ids=ids([1]),
                      next_ids=ids([2]), decode=fake_decode(1))
        with self.assertRaises(ValueError):
            p.refresh(frame_id=0, anchor_ids=ids([2, 1]),
                      next_ids=ids([3]), decode=fake_decode(0))


if __name__ == "__main__":
    unittest.main()

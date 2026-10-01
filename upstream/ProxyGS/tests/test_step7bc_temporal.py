import pytest
import torch

from gdmgs.cache.temporal_bundle_cache import TemporalBundleCache
from test_step7a_full_bundle_cache import ANCHOR_LEVELS, identity, make_batch, assert_payload_equal


def cache(capacity=32):
    return TemporalBundleCache(identity=identity(), anchor_levels=ANCHOR_LEVELS,
                               capacity_rows=capacity, audit=True)


def resolve(c, frame, ids, counts, calls, next_ids=None, **kwargs):
    def decode(requests, levels):
        calls.append(requests.tolist())
        return make_batch(requests.tolist(), counts)
    return c.resolve(frame_id=frame, anchor_ids=torch.tensor(ids, dtype=torch.long),
                     decode=decode,
                     next_ids=None if next_ids is None else torch.tensor(next_ids, dtype=torch.long),
                     **kwargs)


def test_prefetch_complete_next_requests_and_empty_results_skip_decoder():
    c, calls = cache(), []
    counts = {0: 3, 1: 0, 2: 2, 3: 4}
    first, stats = resolve(c, 0, [2, 0], counts, calls, [3, 1, 2])
    assert calls == [[0, 1, 2, 3]]
    assert_payload_equal(first, make_batch([2, 0], counts))
    assert c.generation.rows == 6
    second, stats = resolve(c, 1, [3, 1, 2], counts, calls)
    assert stats['decoder_calls'] == 0 and stats['empty_hits'] == 1
    assert_payload_equal(second, make_batch([3, 1, 2], counts))
    assert c.generation is None
    resolve(c, 2, [1, 2], counts, calls)
    assert calls[-1] == [1, 2]  # Empty hits expire too.


def test_capacity_eviction_preserves_whole_bundles_and_misses_decode_once():
    c, calls = cache(10), []
    counts = {0: 6, 1: 5, 2: 0, 3: 2}
    resolve(c, 0, [0, 1], counts, calls, [0, 1, 2])
    assert c.generation.batch.anchor_indices.tolist() == [0, 2]
    assert c.generation.rows == 6
    out, stats = resolve(c, 1, [3, 0, 2, 1], counts, calls)
    assert calls[-1] == [3, 1]
    assert stats['decoder_calls'] == 1 and stats['hit_anchors'] == 2
    assert_payload_equal(out, make_batch([3, 0, 2, 1], counts))


@pytest.mark.parametrize('frame,allowed', [(2, True), (1, False)])
def test_stale_or_motion_rejected_generation_falls_back(frame, allowed):
    c, calls = cache(), []
    resolve(c, 0, [0], {0: 1}, calls, [0])
    out, stats = resolve(c, frame, [0], {0: 2}, calls, reuse_allowed=allowed)
    assert out.xyz.shape[0] == 2 and stats['hit_anchors'] == 0


def test_failed_decode_does_not_publish_or_advance_frame():
    c, calls = cache(), []
    resolve(c, 0, [0], {0: 1}, calls, [0])
    old = c.generation
    def fail(*_):
        raise RuntimeError('decode failed')
    with pytest.raises(RuntimeError):
        c.resolve(frame_id=1, anchor_ids=torch.tensor([1]), decode=fail)
    assert c.generation is old and c.last_frame == 0


def test_lifecycle_reset_and_invalid_ids():
    c, calls = cache(), []
    resolve(c, 0, [0], {0: 1}, calls, [0])
    with pytest.raises(ValueError):
        resolve(c, 0, [0], {0: 1}, calls)
    c.reset()
    assert c.generation is None
    for ids in ([0, 0], [-1], [5]):
        with pytest.raises(ValueError):
            resolve(c, 0, ids, {0: 1}, calls)
    _, stats = resolve(c, 0, [], {}, calls, [])
    assert stats['decoder_calls'] == 0

import numpy as np
import pytest

from gdmgs.schedule import PairDemandLatch, PairIdentity


def identity(pair=0):
    return PairIdentity("scene", pair, (2 * pair, 2 * pair + 1), (f"c{2*pair}", f"c{2*pair+1}"))


def ids(*values):
    return np.asarray(values, dtype=np.int64)


def test_atomic_pair_is_not_visible_before_publication():
    latch = PairDemandLatch(identity(), submitted_ns=10)
    assert not latch.ready
    with pytest.raises(RuntimeError, match="not ready"):
        latch.demand(identity())
    demand = latch.publish((ids(1, 4), ids(2, 5)), ({"frame": 0}, {"frame": 1}), ready_ns=20)
    assert latch.ready
    assert demand.ready_ns == 20
    assert np.array_equal(latch.demand(identity()).selected_ids[1], ids(2, 5))


def test_partial_or_unsorted_pair_cannot_publish():
    latch = PairDemandLatch(identity(), submitted_ns=10)
    with pytest.raises(ValueError, match="exactly two"):
        latch.publish((ids(1),), ({"frame": 0},), ready_ns=20)
    latch = PairDemandLatch(identity(), submitted_ns=10)
    with pytest.raises(ValueError, match="sorted unique"):
        latch.publish((ids(2, 1), ids(3)), ({}, {}), ready_ns=20)


def test_stale_identity_and_double_publication_fail_closed():
    latch = PairDemandLatch(identity(), submitted_ns=10)
    latch.publish((ids(1), ids(2)), ({}, {}), ready_ns=20)
    with pytest.raises(ValueError, match="mismatched"):
        latch.demand(identity(1))
    with pytest.raises(RuntimeError, match="terminal"):
        latch.publish((ids(1), ids(2)), ({}, {}), ready_ns=30)


def test_failed_pair_never_becomes_ready():
    latch = PairDemandLatch(identity())
    latch.fail(ValueError("depth failed"))
    assert not latch.ready
    with pytest.raises(RuntimeError, match="depth failed"):
        latch.demand(identity())


def test_identity_requires_canonical_pair_frames():
    with pytest.raises(ValueError, match="pair index"):
        PairIdentity("scene", 1, (1, 2), ("a", "b")).validate()

"""CPU metadata failures plus CUDA publication/value-audit coverage."""
import pytest
import torch
from gdmgs.schedule.pair_schedule import PairIdentity
from gdmgs.schedule.gpu_request_contract import GPUPairDemandLatch


def identity(scene="test"):
    return PairIdentity(scene, 0, (0, 1), ("a", "b"))


def test_cpu_tensors_rejected_without_publication():
    latch = GPUPairDemandLatch(identity(), submitted_ns=1)
    with pytest.raises(TypeError, match="CUDA"):
        latch.publish((torch.tensor([1]), torch.tensor([2])), ({}, {}), ready_ns=2)
    assert not latch.ready
    with pytest.raises(RuntimeError, match="not ready"):
        latch.demand(identity())


def test_failure_is_terminal_and_stale_identity_rejected():
    latch = GPUPairDemandLatch(identity())
    latch.fail("query failed")
    assert latch.error == "query failed"
    with pytest.raises(RuntimeError, match="already terminal"):
        latch.publish((), ())
    with pytest.raises(RuntimeError, match="query failed"):
        latch.demand(identity())
    with pytest.raises(ValueError, match="stale"):
        latch.demand(identity("other"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_publication_preserves_storage_and_audits_values():
    ids = (torch.tensor([0, 2], device="cuda"), torch.empty(0, dtype=torch.long, device="cuda"))
    latch = GPUPairDemandLatch(identity(), submitted_ns=1)
    demand = latch.publish(ids, ({}, {}), ready_ns=2)
    assert latch.ready
    assert demand is latch.demand(identity())
    assert demand.selected_ids[0] is ids[0]
    assert demand.selected_ids[1] is ids[1]
    demand.validate_values(anchor_count=3)
    with pytest.raises(ValueError, match="outside"):
        demand.validate_values(anchor_count=2)
    with pytest.raises(RuntimeError, match="terminal"):
        latch.publish(ids, ({}, {}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("values", [[-1, 1], [2, 1], [1, 1]])
def test_value_errors_are_deferred_until_audit(values):
    latch = GPUPairDemandLatch(identity())
    ids = torch.tensor(values, device="cuda")
    demand = latch.publish((ids, ids), ({}, {}))
    assert latch.ready
    with pytest.raises(ValueError, match="sorted unique nonnegative"):
        demand.validate_values()

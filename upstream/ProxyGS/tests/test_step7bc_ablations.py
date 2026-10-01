import pytest
import torch

from gdmgs.cache.temporal_bundle_cache_v2 import TemporalBundleCache
from test_step7a_full_bundle_cache import ANCHOR_LEVELS, identity, make_batch, assert_payload_equal
from test_step7bc_temporal import resolve


def cache(**kwargs):
    return TemporalBundleCache(identity=identity(), anchor_levels=ANCHOR_LEVELS,
                               capacity_rows=kwargs.pop('capacity_rows',32), audit=True, **kwargs)


def test_fast_handoff_transfers_final_generation_without_copy():
    c, calls = cache(fast_handoff=True), []
    resolve(c,0,[1,0],{0:2,1:0,2:3},calls,[0,1,2])
    stored=c.generation.batch
    output, stats=resolve(c,1,[0,1,2],{0:2,1:0,2:3},calls)
    assert output is stored
    assert stats['decoder_calls']==0 and stats['hit_rows']==5
    assert c.generation is None


@pytest.mark.parametrize('age',[2,3])
def test_longer_epoch_preserves_original_source_age_and_expires(age):
    c,calls=cache(max_age=age,fast_handoff=True),[]
    resolve(c,0,[0],{0:1,1:2},calls,[0,1])
    for frame in range(1,age+1):
        output,stats=resolve(c,frame,[0,1],{0:3,1:4},calls)
        assert stats['source_age']==frame and stats['source_frame']==0
        assert stats['decoder_calls']==0
        assert_payload_equal(output,make_batch([0,1],{0:1,1:2}))
    assert c.generation is None
    _,stats=resolve(c,age+1,[0,1],{0:3,1:4},calls)
    assert stats['decoder_calls']==1


@pytest.mark.parametrize('capacity',[10,32])
def test_arena_counts_all_physical_rows_and_falls_back_atomically(capacity):
    c,calls=cache(union_arena=True,capacity_rows=capacity),[]
    resolve(c,0,[0,1],{0:6,1:5,2:2},calls,[1,2])
    assert c.generation.rows<=capacity
    assert c.generation.rows==(7 if capacity==10 else 13)
    output,_=resolve(c,1,[2,1],{0:6,1:5,2:2},calls)
    assert_payload_equal(output,make_batch([2,1],{0:6,1:5,2:2}))


def test_fp16_only_quantizes_future_payload_and_keeps_ownership():
    c,calls=cache(payload_half=True,fast_handoff=True),[]
    fresh,_=resolve(c,0,[0,1],{0:2,1:0},calls,[0,1])
    assert fresh.color.dtype==torch.float32
    assert_payload_equal(fresh,make_batch([0,1],{0:2,1:0}))
    output,stats=resolve(c,1,[0,1],{0:2,1:0},calls)
    assert output.color.dtype==torch.float16
    assert output.bundle_metadata.row_owner_ids.tolist()==[0,0]
    assert stats['empty_hits']==1


def test_long_epoch_partial_miss_does_not_renew_stored_age():
    c,calls=cache(max_age=2),[]
    resolve(c,0,[0],{0:1},calls,[0])
    _,stats=resolve(c,1,[0,1],{0:2,1:3},calls)
    assert stats['decoder_calls']==1 and c.generation.source_frame==0
    _,stats=resolve(c,2,[0,1],{0:2,1:3},calls)
    assert stats['source_age']==2 and calls[-1]==[1]

import pytest
import torch
from gdmgs.cache.temporal_bundle_cache_v3 import TemporalBundleCache
from test_step7a_full_bundle_cache import ANCHOR_LEVELS, identity, make_batch, assert_payload_equal

@pytest.mark.parametrize('current,future', [([3,1],[0,1,2]),([0,1],[0,1]),([],[0,1]),([2,0],[])])
def test_prefix_decode_preserves_unsorted_order_and_payload_alias(current,future):
    c=TemporalBundleCache(identity=identity(),anchor_levels=ANCHOR_LEVELS,
                         capacity_rows=32,audit=True,fast_handoff=True,prefix_decode=True)
    decoded=[]
    counts={0:2,1:0,2:3,3:1}
    def decode(ids,levels):
        b=make_batch(ids.tolist(),counts)
        decoded.append(b)
        return b
    ids=torch.tensor(current,dtype=torch.long)
    out,stats=c.resolve(frame_id=0,anchor_ids=ids,next_ids=torch.tensor(future,dtype=torch.long),decode=decode)
    assert_payload_equal(out,make_batch(current,counts))
    if out.xyz.numel():
        assert out.xyz.data_ptr()==decoded[0].xyz.data_ptr()
    assert decoded[0].anchor_indices[:len(current)].tolist()==current
    out,stats=c.resolve(frame_id=1,anchor_ids=torch.tensor(future,dtype=torch.long),decode=decode)
    assert_payload_equal(out,make_batch(future,counts))
    assert stats['decoder_calls']==0

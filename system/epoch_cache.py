"""Immutable, capacity-bounded epoch arena with truthful fractional source pose."""
import copy,math
import numpy as np
import torch
from ascending_cache import take_prefix,take_requests,FIELDS
from dense_math import interpolate_pose
from batch import NeuralGaussianBatch,BundleMetadata


def epoch_spec(start,k,total,centered):
    if k<1 or start<0 or start>=total:
        raise ValueError('invalid epoch bounds')
    actual=min(k,total-start)
    return dict(start=start,k=actual,source_index=start+(actual-1)/2 if centered else float(start),
        end=start+actual,lookahead_frames=actual-1)


def source_view(e,spec):
    """Decoder-only source pose. Target render cameras are never changed."""
    x=spec['source_index'];lo=int(math.floor(x));hi=int(math.ceil(x))
    a=e.views[lo]
    if lo==hi:return a
    b=e.views[hi];v=copy.copy(a)
    v.R,v.T=interpolate_pose(a.R,a.T,b.R,b.T,x-lo)
    # Model's decode interface uses camera_center and resolution_scale; derive
    # centre with the same float32 world-to-view conversion as dense targets.
    from utils.graphics_utils import getWorld2View2
    v.world_view_transform=torch.tensor(getWorld2View2(v.R,v.T),dtype=a.world_view_transform.dtype,
        device=a.world_view_transform.device).T.contiguous()
    v.camera_center=v.world_view_transform.inverse()[3,:3]
    return v


class EpochArena:
    def __init__(self,spec,requests,levels,decode,capacity):
        self.spec=dict(spec);self.requests=requests;self.levels=levels;self.next_frame=spec['start']
        if len(requests)!=spec['k']:raise ValueError('missing epoch requests')
        union=torch.unique(torch.cat(requests),sorted=True)
        first=requests[0];first_positions=torch.searchsorted(union,first)
        extra=torch.ones(len(union),dtype=torch.bool,device=union.device);extra[first_positions]=False
        ordered=torch.cat((first,union[extra]))
        remap=torch.empty_like(union)
        remap[first_positions]=torch.arange(len(first),device=union.device)
        remap[extra]=torch.arange(len(first),len(union),device=union.device)
        self.maps=[remap[torch.searchsorted(union,ids)] for ids in requests]
        if len(ordered):
            self.batch=decode(ordered,levels[ordered])
        else:
            meta=BundleMetadata(ordered,ordered,ordered,ordered,ordered.new_zeros(1),levels[ordered],levels[ordered])
            z=lambda n:torch.empty((0,n),dtype=torch.float32,device=ordered.device)
            self.batch=NeuralGaussianBatch(ordered,z(3),z(3),z(1),z(3),z(4),torch.empty(0,dtype=torch.bool,device=ordered.device),None,meta)
        assert torch.equal(self.batch.anchor_indices,ordered)
        assert all(getattr(self.batch,k).dtype==torch.float32 for k in FIELDS)
        self.decoded_anchors=len(ordered);self.decoded_rows=len(self.batch.xyz)
        self.decoder_calls=int(len(ordered)>0)
        self.overflow=self.decoded_rows>capacity
        meta=self.batch.bundle_metadata
        self.bytes=sum(getattr(self.batch,n).numel()*getattr(self.batch,n).element_size() for n in FIELDS)
        self.bytes+=sum(v.numel()*v.element_size() for v in vars(meta).values() if isinstance(v,torch.Tensor))
        self.bytes+=self.batch.selection_mask.numel()*self.batch.selection_mask.element_size()
        if self.overflow:
            self.batch=None;self.bytes=0

    def consume(self,frame,ids):
        if self.overflow:raise RuntimeError('over-capacity union is not resident')
        if frame!=self.next_frame or not self.spec['start']<=frame<self.spec['end']:
            raise ValueError('wrong frame/epoch; source identity cannot be renewed')
        j=frame-self.spec['start']
        if not torch.equal(ids,self.requests[j]):raise ValueError('request certificate differs')
        out=take_prefix(self.batch,len(ids)) if j==0 else take_requests(self.batch,self.maps[j],ids,self.levels[ids])
        assert torch.equal(out.anchor_indices,ids)
        self.next_frame+=1
        return out

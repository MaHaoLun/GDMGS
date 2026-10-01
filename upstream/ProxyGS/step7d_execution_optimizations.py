"""Typed epoch set plans and checked certificates; no global cache mutation."""
from dataclasses import dataclass
import torch


@dataclass
class RefreshPlan:
    source: torch.Tensor
    next_ids: torch.Tensor
    union: torch.Tensor
    current_positions: torch.Tensor
    next_positions: torch.Tensor


class GPUSetPlanner:
    def __init__(self, demands):
        self.demands=demands
        self.union,self.inverse=torch.unique(torch.cat(demands),sorted=True,return_inverse=True)
        self.positions=[];offset=0
        for ids in demands:
            self.positions.append(self.inverse[offset:offset+ids.numel()])
            offset+=ids.numel()
        self.source_mask=torch.zeros(self.union.numel(),dtype=torch.bool,device=self.union.device)
        self.source_mask[self.positions[0]]=True
        self.prefix_masks={};self.future_masks={};self.counts={}

    def prepare_prefixes(self):
        cumulative=self.source_mask.clone()
        future=torch.zeros_like(cumulative)
        metrics=[]
        for i in range(1,len(self.demands)):
            p=self.positions[i]
            overlap=self.source_mask[p].sum()
            cumulative[p]=True;future[p]=True
            self.prefix_masks[i+1]=cumulative.clone()
            self.future_masks[i+1]=future.clone()
            metrics.append(torch.stack((overlap,cumulative.sum())))
        if metrics:
            self.counts={k+2:vals for k,vals in enumerate(torch.stack(metrics).cpu().tolist())}

    def feature_vectors(self,pose):
        self.prepare_prefixes()
        minimum_j=minimum_c=1.
        output={}
        ns=self.demands[0].numel()
        for k,(common,union_count) in self.counts.items():
            nt=self.demands[k-1].numel()
            minimum_j=min(minimum_j,common/max(1,ns+nt-common))
            minimum_c=min(minimum_c,common/max(1,nt))
            x=list(pose[k]);x[4:7]=[minimum_j,minimum_c,union_count/max(1,ns)]
            output[k]=x
        return output

    def refresh_plan(self,k):
        if k<2 or k>len(self.demands):raise ValueError('invalid planned epoch')
        if k==2 and len(self.demands)==2:
            # Provider promises sorted, unique IDs; checked once at construction.
            return RefreshPlan(self.demands[0],self.demands[1],self.union,
                               self.positions[0],self.positions[1])
        if k not in self.prefix_masks:self.prepare_prefixes()
        union=self.union[self.prefix_masks[k]]
        nxt=self.union[self.future_masks[k]]
        return RefreshPlan(self.demands[0],nxt,union,torch.searchsorted(union,self.demands[0]),
                           torch.searchsorted(union,nxt))


def planned_refresh(cache,frame,plan,decode):
    """Same prefix decode and packed admission as v3, reusing set maps."""
    from gdmgs.cache.temporal_bundle_cache_v3 import take_prefix,take_requests,TemporalGeneration
    ids,nxt,union=plan.source,plan.next_ids,plan.union
    if frame<=cache.last_frame:raise ValueError('nonmonotonic frame')
    cache._check_ids(ids);cache._check_ids(nxt)
    stats=dict(decoder_calls=0,decoded_anchors=0,selected_anchors=ids.numel(),
        hit_anchors=0,empty_hits=0,prefetch_anchors=0,evicted_anchors=0,
        source_age=0,source_frame=-1,hit_rows=0)
    current=plan.current_positions
    extra=torch.ones_like(union,dtype=torch.bool);extra[current]=False
    ordered=torch.cat((ids,union[extra]))
    mapping=torch.empty_like(union)
    mapping[current]=torch.arange(ids.numel(),device=ids.device)
    mapping[extra]=torch.arange(ids.numel(),union.numel(),device=ids.device)
    batch=cache._decode(ordered,decode,stats)
    result=take_prefix(batch,ids.numel())
    positions=mapping[plan.next_positions]
    counts=batch.bundle_metadata.counts[positions]
    fits=(counts.cumsum(0)<=cache.capacity_rows)|(counts==0)
    kept=nxt[fits]
    retained=take_requests(batch,positions[fits],kept,cache.levels[kept])
    generation=TemporalGeneration(frame,retained)
    cache.generation=generation;cache.last_frame=frame
    stats.update(prefetch_anchors=union.numel()-ids.numel(),evicted_anchors=int((~fits).sum()),
        resident_rows=generation.rows,resident_descriptors=retained.anchor_indices.numel(),
        resident_bytes=generation.memory_bytes(),output_rows=result.xyz.shape[0],
        empty_output_requests=int((result.bundle_metadata.counts==0).sum()),
        resident_empty_descriptors=int((retained.bundle_metadata.counts==0).sum()),
        capacity_rows=cache.capacity_rows,lookahead_requests=nxt.numel(),miss_anchors=ids.numel())
    return result,stats


@dataclass(frozen=True)
class HitCertificate:
    cache_identity: object
    generation: object
    source_frame: int
    target_frame: int
    target_token: object
    request_count: int
    empty_count: int

    def valid(self,cache,frame,token):
        return (cache.identity==self.cache_identity and cache.generation is self.generation
                and cache.last_frame==self.source_frame and frame==self.target_frame
                and self.target_frame==self.source_frame+1 and token is self.target_token)

    def consume(self,cache,frame,token):
        if not self.valid(cache,frame,token):raise ValueError('invalid hit certificate')
        batch=self.generation.batch
        stats=dict(decoder_calls=0,decoded_anchors=0,selected_anchors=self.request_count,
            hit_anchors=self.request_count,empty_hits=self.empty_count,prefetch_anchors=0,
            evicted_anchors=0,resident_rows=0,resident_descriptors=0,resident_bytes=0,
            source_age=1,source_frame=self.source_frame,hit_rows=self.generation.rows,
            output_rows=self.generation.rows,empty_output_requests=self.empty_count,
            resident_empty_descriptors=0,capacity_rows=cache.capacity_rows,
            lookahead_requests=0,miss_anchors=0)
        cache.generation=None;cache.last_frame=frame
        return batch,stats


def certify_pair(cache,source,target_token,target_count,stats):
    g=cache.generation
    if (g is None or g.source_frame!=source or stats['evicted_anchors']
            or g.batch.anchor_indices.numel()!=target_count):return None
    # Called only by the private immutable-demand pair2 provider immediately
    # after refresh; next_ids provenance supplies identity/order, not counts alone.
    return HitCertificate(cache.identity,g,source,source+1,target_token,target_count,
                          stats['resident_empty_descriptors'])

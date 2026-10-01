"""Exact cache construction variants; original model math and checks retained."""
import ast
import inspect
import textwrap
from dataclasses import dataclass,replace
import torch
import triton
import triton.language as tl
from epoch_cache import EpochArena,FIELDS
from epoch_pipeline import pipeline_sequence
from gaussian_renderer import generate_neural_gaussians
from gaussian_renderer.raster_batch import NeuralGaussianBatch,BundleMetadata,batch_from_proxygs_decode


@dataclass(frozen=True)
class UnionPlan:
    ordered: object
    maps: list
    anchor_count: int


def build_union_plan(requests):
    if not requests:raise ValueError('empty epoch')
    union,inverse=torch.unique(torch.cat(requests),sorted=True,return_inverse=True)
    first=requests[0];positions=inverse[:len(first)]
    extra=torch.ones(len(union),dtype=torch.bool,device=union.device);extra[positions]=False
    ordered=torch.cat((first,union[extra]));remap=torch.empty_like(union)
    remap[positions]=torch.arange(len(first),device=union.device)
    remap[extra]=torch.arange(len(first),len(union),device=union.device)
    maps=[];cursor=0
    for ids in requests:
        maps.append(remap[inverse[cursor:cursor+len(ids)]]);cursor+=len(ids)
    return UnionPlan(ordered,maps,len(union))


class PlannedArena(EpochArena):
    def __init__(self,spec,requests,levels,decode,capacity,plan):
        self.spec=dict(spec);self.requests=requests;self.levels=levels;self.next_frame=spec['start']
        if len(requests)!=spec['k']:raise ValueError('missing epoch requests')
        ordered=plan.ordered;self.maps=plan.maps
        if len(ordered):self.batch=decode(ordered,levels[ordered])
        else:
            meta=BundleMetadata(ordered,ordered,ordered,ordered,ordered.new_zeros(1),levels[ordered],levels[ordered])
            z=lambda n:torch.empty((0,n),dtype=torch.float32,device=ordered.device)
            self.batch=NeuralGaussianBatch(ordered,z(3),z(3),z(1),z(3),z(4),torch.empty(0,dtype=torch.bool,device=ordered.device),None,meta)
        assert torch.equal(self.batch.anchor_indices,ordered)
        assert all(getattr(self.batch,k).dtype==torch.float32 for k in FIELDS)
        self.decoded_anchors=len(ordered);self.decoded_rows=len(self.batch.xyz);self.decoder_calls=int(len(ordered)>0)
        self.overflow=self.decoded_rows>capacity
        self.bytes=sum(getattr(self.batch,n).numel()*getattr(self.batch,n).element_size() for n in FIELDS)
        self.bytes+=sum(v.numel()*v.element_size() for v in vars(self.batch.bundle_metadata).values() if isinstance(v,torch.Tensor))
        self.bytes+=self.batch.selection_mask.numel()*self.batch.selection_mask.element_size()
        if self.overflow:self.batch=None;self.bytes=0


@triton.jit
def _identity(MASK,IDS,LEVELS,COUNTS,OFF,OWNERS,SLOTS,ROW_LEVELS,BAD,A,N,O:tl.constexpr,B:tl.constexpr):
    a=tl.program_id(0)*B+tl.arange(0,B);active=a<A
    owner=tl.load(IDS+a,active,0);level=tl.load(LEVELS+a,active,0)
    begin=tl.load(OFF+a,active,0);end=tl.load(OFF+a+1,active,0);count=tl.load(COUNTS+a,active,0)
    bad=active&((owner<0)|(level<0)|(begin<0)|(end>N)|((end-begin)!=count))
    cursor=begin
    for j in tl.static_range(O):
        keep=tl.load(MASK+a*O+j,active,0)!=0
        write=active&keep&(cursor>=0)&(cursor<N)&(cursor<end)
        tl.store(OWNERS+cursor,owner,write);tl.store(SLOTS+cursor,j,write);tl.store(ROW_LEVELS+cursor,level,write)
        cursor+=keep.to(tl.int64)
    bad=bad|(active&(cursor!=end))
    total=tl.load(OFF+A)
    if (tl.sum(bad.to(tl.int32),0)>0)|(total!=N):tl.atomic_or(BAD,1)


@triton.jit
def _finite(X,C,O,S,R,BAD,N,
            X0:tl.constexpr,X1:tl.constexpr,C0:tl.constexpr,C1:tl.constexpr,O0:tl.constexpr,O1:tl.constexpr,
            S0:tl.constexpr,S1:tl.constexpr,R0:tl.constexpr,R1:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);active=i<N;bad=tl.full((B,),False,tl.int1)
    for j in tl.static_range(3):
        x=tl.load(X+i*X0+j*X1,active,0.);c=tl.load(C+i*C0+j*C1,active,0.);s=tl.load(S+i*S0+j*S1,active,0.)
        bad=bad|(~(tl.abs(x)<float('inf')))|(~(tl.abs(c)<float('inf')))|(~(tl.abs(s)<float('inf')))
    op=tl.load(O+i*O0,active,0.);bad=bad|(~(tl.abs(op)<float('inf')))
    for j in tl.static_range(4):
        r=tl.load(R+i*R0+j*R1,active,0.);bad=bad|(~(tl.abs(r)<float('inf')))
    if tl.sum((bad&active).to(tl.int32),0)>0:tl.atomic_or(BAD,2)


def fused_batch(anchor_ids,decoded,n_offsets,request_level_ids):
    if len(decoded)!=6:raise ValueError('six decoder tensors required')
    xyz,color,opacity,scaling,rotation,mask=decoded
    if anchor_ids.dtype!=torch.long or anchor_ids.ndim!=1 or not anchor_ids.is_contiguous():raise ValueError('invalid anchor IDs')
    device=anchor_ids.device;a=len(anchor_ids);n=len(xyz)
    if device.type!='cuda':raise ValueError('CUDA required')
    if request_level_ids is None or request_level_ids.dtype!=torch.long or request_level_ids.shape!=anchor_ids.shape or request_level_ids.device!=device or not request_level_ids.is_contiguous():raise ValueError('invalid request levels')
    if mask.dtype!=torch.bool or mask.ndim!=1 or mask.numel()!=a*n_offsets or mask.device!=device or not mask.is_contiguous():raise ValueError('invalid selection mask')
    for t,width in zip(decoded[:5],(3,3,1,3,4)):
        if t.dtype!=torch.float32 or t.device!=device or t.shape!=(n,width):raise ValueError('invalid FP32 payload contract')
    counts=mask.reshape(a,n_offsets).sum(dim=1,dtype=torch.long)
    offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
    owner=torch.empty(n,dtype=torch.long,device=device);slots=torch.empty_like(owner);row_levels=torch.empty_like(owner)
    bad=torch.zeros((),dtype=torch.int32,device=device)
    if a:_identity[(triton.cdiv(a,128),)](mask,anchor_ids,request_level_ids,counts,offsets,owner,slots,row_levels,bad,a,n,n_offsets,128)
    elif n:raise ValueError('row count without anchors')
    if n:_finite[(triton.cdiv(n,256),)](xyz,color,opacity,scaling,rotation,bad,n,*xyz.stride(),*color.stride(),*opacity.stride(),*scaling.stride(),*rotation.stride(),256)
    # All numeric validations remain inside the real decode/construction timer.
    error=int(bad.item())
    if error:raise ValueError(f'invalid bundle identity/count/nonnegative/finite contract: {error}')
    meta=BundleMetadata(anchor_ids,owner,slots,counts,offsets,request_level_ids,row_levels)
    return NeuralGaussianBatch(anchor_ids,xyz,color,opacity,scaling,rotation,mask,None,meta)


def optimized_decode(view,model,ids,levels,identity=False,layout=False):
    decoded=generate_neural_gaussians(view,model,is_training=False,anchor_indices=ids)
    batch=fused_batch(ids,decoded,model.n_offsets,levels) if identity else batch_from_proxygs_decode(anchor_ids=ids,decoded=decoded,n_offsets=model.n_offsets,request_level_ids=levels)
    if layout:batch=replace(batch,color=batch.color.contiguous())
    return batch


def make_sequence(options):
    """Clone only this function; preserve original files, module globals and API."""
    options=frozenset(options);source=textwrap.dedent(inspect.getsource(pipeline_sequence));tree=ast.parse(source)
    class Rewrite(ast.NodeTransformer):
        def __init__(self):self.changed=dict(imports=0,reservation=0,arena=0,decode=0)
        def visit_ImportFrom(self,node):
            if node.module=='epoch_cache':
                node.names=[n for n in node.names if n.name!='EpochArena'];self.changed['imports']+=1
            return node
        def visit_Assign(self,node):
            node=self.generic_visit(node)
            if 'U' in options and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                if node.targets[0].id=='reservation':
                    self.changed['reservation']+=1
                    return ast.parse('plan = build_union_plan(requests)\nreservation = plan.anchor_count * int(e.rt.model.n_offsets)').body
                if node.targets[0].id=='arena':
                    self.changed['arena']+=1
                    node.value.keywords.append(ast.keyword(arg='plan',value=ast.Name(id='plan',ctx=ast.Load())))
                    return [node,ast.Delete(targets=[ast.Name(id='plan',ctx=ast.Del())])]
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if ast.unparse(node.func)=='e.decode_batch':node.func=ast.Name(id='cache_decode',ctx=ast.Load());self.changed['decode']+=1
            return node
    rewrite=Rewrite();tree=rewrite.visit(tree);ast.fix_missing_locations(tree)
    assert rewrite.changed==dict(imports=1,reservation=int('U' in options),arena=int('U' in options),decode=1),rewrite.changed
    namespace=dict(pipeline_sequence.__globals__)
    namespace.update(EpochArena=PlannedArena if 'U' in options else EpochArena,build_union_plan=build_union_plan,
                     cache_decode=lambda view,model,ids,levels:optimized_decode(view,model,ids,levels,'I' in options,'L' in options))
    transformed=ast.unparse(tree);exec(compile(tree,'<cache-build:'+''.join(sorted(options))+'>','exec'),namespace)
    return namespace['pipeline_sequence'],dict(options=sorted(options),changes=rewrite.changed,transformed_source=transformed)

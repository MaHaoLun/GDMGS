"""Shared Morton topology with typed bounds and a single CUDA frontier.

The topology is data dependent. No center-only spatial cell is used to reject
geometry. Original support and complete-triangle leaf predicates are retained.
"""
from pathlib import Path
import hashlib
import time
import numpy as np
import torch


def native():
    from torch.utils.cpp_extension import load
    p=Path(__file__).parent/'native'
    return load(name='joint_frustum_20260924',sources=[str(p/f) for f in
        ('bindings.cpp','radix_build.cu','query.cu')],extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)


class Radix:
    def __init__(self,bounds,domain=None):
        self.ext=native();self.bounds=bounds;self.N=len(bounds);self.L=(self.N+31)//32
        if not self.N:raise ValueError('nonempty combined geometry required')
        torch.cuda.synchronize();start=time.perf_counter()
        centers=(bounds[:,:3]+bounds[:,3:6])*0.5
        self.lo,self.hi=(centers.amin(0).contiguous(),centers.amax(0).contiguous()) if domain is None else domain
        self.keys=torch.sort(self.ext.morton_codes(bounds,self.lo,self.hi)).values.contiguous()
        self.order=(self.keys&((1<<32)-1)).contiguous()
        slots=torch.arange(self.L,device=bounds.device,dtype=torch.int64)
        self.leaf_keys=(((self.keys[slots*32]>>32)<<32)|slots).contiguous()
        self.nodes,self.left,self.right,self.parent,self.axis,self.split=self.ext.build_radix_bvh(
            bounds,self.order,self.leaf_keys,self.lo,self.hi,32)
        torch.cuda.synchronize();self.build_ms=(time.perf_counter()-start)*1000

    def typed_nodes(self,first,last):
        b=torch.empty_like(self.bounds);b[:,:3]=float('inf');b[:,3:6]=-float('inf');b[:,6]=0
        b[first:last]=self.bounds[first:last]
        nodes,left,right,parent,axis,split=self.ext.build_radix_bvh(b,self.order,self.leaf_keys,self.lo,self.hi,32)
        assert torch.equal(left,self.left) and torch.equal(right,self.right) and torch.equal(parent,self.parent)
        return nodes


class Joint:
    def __init__(self,anchor_bounds,vertices,faces,mesh_bounds):
        self.ab=anchor_bounds;self.vertices=vertices;self.faces=faces;self.na=len(anchor_bounds)
        torch.cuda.synchronize();start=time.perf_counter()
        self.tree=Radix(torch.cat((anchor_bounds,mesh_bounds),0).contiguous())
        self.an=self.tree.typed_nodes(0,self.na)
        self.mn=self.tree.typed_nodes(self.na,self.tree.N)
        torch.cuda.synchronize();self.build_ms=(time.perf_counter()-start)*1000

    def query(self,cp,planes,eye,ids,mode=3):
        t=self.tree
        am,mm,states,active=t.ext.query(self.ab,self.vertices,self.faces,self.an,self.mn,
            t.left,t.right,t.parent,t.order,cp,planes,t.axis,t.split,eye,mode,64)
        a=ids[am[ids]] if mode&1 else None
        m=torch.nonzero(mm,as_tuple=False).flatten() if mode&2 else None
        return a,m,states,active


def fingerprint(t):
    def hashed(x):return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
    return dict(objects=t.N,leaves=t.L,nodes=2*t.L-1,scene_lo=t.lo.cpu().tolist(),scene_hi=t.hi.cpu().tolist(),
        hashes={k:hashed(getattr(t,k)) for k in ('left','right','parent','axis','split','leaf_keys')})


def audit(t,typed=None):
    order=t.order.cpu().numpy();parent=t.parent.cpu().numpy();left=t.left.cpu().numpy();right=t.right.cpu().numpy()
    assert np.array_equal(np.sort(order),np.arange(t.N))
    assert parent[0]==-1 and np.count_nonzero(parent==-1)==1
    seen=np.zeros(len(parent),bool);stack=[(0,0)];depth=0
    while stack:
        node,d=stack.pop();assert not seen[node];seen[node]=True;depth=max(depth,d)
        if node<t.L-1:
            for c in (left[node],right[node]):assert parent[c]==node;stack.append((int(c),d+1))
    assert seen.all() and depth<64
    groups=[(t.bounds,t.nodes)]
    if typed is not None:
        na,an,mn=typed
        for first,last,nodes in ((0,na,an),(na,t.N,mn)):
            b=torch.empty_like(t.bounds);b[:,:3]=float('inf');b[:,3:6]=-float('inf');b[:,6]=0
            b[first:last]=t.bounds[first:last];groups.append((b,nodes))
    for bounds,nodes in groups:
        b=bounds.cpu().numpy()[order];n=nodes.cpu().numpy();starts=np.arange(0,t.N,32)
        expected=np.concatenate((np.minimum.reduceat(b[:,:3],starts),np.maximum.reduceat(b[:,3:6],starts),np.maximum.reduceat(b[:,6:],starts)),1)
        assert np.array_equal(expected,n[t.L-1:])
        assert np.array_equal(n[:t.L-1,:3],np.minimum(n[left,:3],n[right,:3]))
        assert np.array_equal(n[:t.L-1,3:],np.maximum(n[left,3:],n[right,3:]))
    return dict(valid=True,max_depth=depth,every_object_once=True,connected=True,all_typed_bounds_exact=True)


def counters(states,active,tree,na):
    s=states.cpu().numpy();order=tree.order.cpu().numpy();leaf=s[tree.L-1:];parent=tree.parent.cpu().numpy()
    cnt={}
    for ty,name in ((0,'anchor'),(1,'mesh')):
        labels=leaf[np.arange(tree.N)//32,ty];belongs=order<na if ty==0 else order>=na
        eligible=(s[:,ty]!=255)&((parent<0)|(s[np.maximum(parent,0),ty]==2))
        cnt[name]=dict(visited=int((s[:,ty]!=255).sum()),classifier_invocations=int(eligible.sum()),cull=int((s[:,ty]==0).sum()),keep=int((s[:,ty]==1).sum()),descend=int((s[:,ty]==2).sum()),leaf_fallback=int(((labels==2)&belongs).sum()))
    cnt['active_nodes']=int(active.sum());cnt['active_levels']=int((active>0).sum())
    return cnt

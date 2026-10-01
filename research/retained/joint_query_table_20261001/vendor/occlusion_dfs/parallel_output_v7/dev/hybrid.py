from pathlib import Path
import torch
import torch.nn.functional as F


def native():
    from torch.utils.cpp_extension import load
    p=Path(__file__).parent
    return load(name='parallel_subtree_output_hybrid_size_20260924',sources=[str(p/'bindings.cpp'),str(p/'query.cu')],
                extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)


def pyramid(depth):
    # Max reduction with unknown padded samples, including at every upper level.
    x=torch.where(torch.isfinite(depth)&(depth>0),depth,float('inf'))
    x=torch.nextafter(x+32*torch.finfo(x.dtype).eps*(x.abs()+1),torch.full_like(x,float('inf')))
    h,w=x.shape;x=F.pad(x,(0,(-w)%8,0,(-h)%8),value=float('inf'))
    x=F.max_pool2d(x[None,None],8,8);levels=[];meta=[];offset=0
    while True:
        h,w=x.shape[-2:];levels.append(x.flatten());meta.append((offset,w,h));offset+=w*h
        if h==1 and w==1:break
        x=F.max_pool2d(F.pad(x,(0,w%2,0,h%2),value=float('inf')),2,2)
    return torch.cat(levels),torch.tensor(meta,device=depth.device,dtype=torch.int64)


def partial_upper(depth,near,far):
    # Pre-registered budget: 1e-3 absolute, 1e-4 relative, or the inverse
    # projection sensitivity of 64 float32 eps in normalized depth, whichever
    # is largest. This is an experimentally checked tolerance, not a universal
    # rasterizer error theorem. Unbounded inversion is unknown (+inf).
    eps=64*torch.finfo(torch.float32).eps
    b=2*near*far/(far-near) if far!=float('inf') else 2*near
    z=depth.double();den=b/z-eps
    inv=torch.where((den>0)&torch.isfinite(z),b/den,torch.full_like(z,float('inf')))
    upper=torch.maximum(inv,torch.maximum(z+1e-3,z+1e-4*z.abs()))
    upper=torch.where(torch.isfinite(z)&(z>0),upper,torch.full_like(z,float('inf')))
    return torch.nextafter(upper.float(),torch.full_like(depth,float('inf')))


class Hybrid:
    def __init__(self,joint,cut=20):
        self.j=joint;self.t=joint.tree;self.ext=native();self.cut=cut
        self.lo,self.hi=self.ext.spans(self.t.parent)

    def filter(self,ids,cp,depth):
        hz,meta=pyramid(depth)
        return ids[self.ext.filter(self.j.ab,cp,hz,meta,ids)]

    def query(self,cp,planes,eye,ids,*,raster=None,domain=None,epochs=1,occlusion=False,incremental=None):
        j,t=self.j,self.t
        flags,states,active=self.ext.prefix(j.an,j.mn,t.left,t.right,t.parent,cp,planes,t.axis,t.split,eye,self.cut)
        # Source-order bitmap compaction creates deterministic roots without an
        # atomic queue plus an extra root-ID sort.
        roots=torch.nonzero(flags).flatten().to(torch.int32)
        if epochs>1:
            b=j.mn[roots.long()];r=cp[8:11]
            z=torch.minimum(b[:,:3]*r,b[:,3:6]*r).sum(1)+cp[11]
            z=torch.where(b[:,0]<=b[:,3],z,float('inf'))
            roots=roots[torch.argsort(z,stable=True)].contiguous()
        am=torch.zeros(j.na,device=cp.device,dtype=torch.bool)
        mm=torch.zeros(len(j.faces),device=cp.device,dtype=torch.bool)
        hz=torch.empty(0,device=cp.device,dtype=torch.float32)
        meta=torch.empty((0,3),device=cp.device,dtype=torch.int64)
        counters=[];partial=None;depth=None
        if incremental is not None:
            if not occlusion:raise ValueError("incremental depth requires occlusion")
            incremental.begin(domain,(1600,900))
        for k in range(epochs):
            rr=roots[len(roots)*k//epochs:len(roots)*(k+1)//epochs]
            batch,counts=self.ext.dfs(j.ab,j.vertices,j.faces,j.an,j.mn,t.left,t.right,t.parent,t.order,
                cp,planes,self.lo,self.hi,rr,states,hz,meta,am)
            mm|=batch;counters.append(counts)
            if incremental is not None:
                depth=incremental.append(torch.nonzero(batch).flatten())
                if k<epochs-1:hz,meta=pyramid(partial_upper(depth,domain.near,domain.far))
            elif occlusion and k<epochs-1:
                d=raster.render(torch.nonzero(batch).flatten(),domain,(1600,900),copy_to_cpu=False).depth_gpu
                partial=d if partial is None else torch.minimum(partial,d)
                hz,meta=pyramid(partial_upper(partial,domain.near if domain is not None else .01,domain.far if domain is not None else 100.))
        a=ids[am[ids]];m=torch.nonzero(mm).flatten()
        if occlusion:
            # Final exact complete-mesh raster remains part of the measured cost.
            if incremental is None:depth=raster.render(m,domain,(1600,900),copy_to_cpu=False).depth_gpu
            a=self.filter(a,cp,depth)
        return a,m,depth,dict(counts=torch.stack(counters),frontier=len(roots),active=active,partial=partial,incremental=incremental.diagnostics() if incremental is not None else None)

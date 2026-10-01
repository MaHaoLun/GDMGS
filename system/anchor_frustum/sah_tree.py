"""Balanced, 8-split-candidate SAH BVH over the frozen support-center bounds.

The fixed complete-tree layout keeps the GPU traversal kernel identical to the
Morton experiment. SAH considers all three axes and up to eight admissible
leaf-aligned split ranks at every internal node. It is SAH-guided, not a claim
of global optimality or identity with the historical CPU point-BVH builder.
"""
import numpy as np


def _area(lo,hi):
    e=np.maximum(hi-lo,0)
    return float(2*(e[0]*e[1]+e[0]*e[2]+e[1]*e[2]))


def make_sah_tree(bounds,leaf_size=32):
    if leaf_size<1 or bounds.ndim!=2 or bounds.shape[1]!=7 or not np.isfinite(bounds).all():
        raise ValueError('finite [N,7] support bounds required')
    if (bounds[:,3:6]<bounds[:,:3]).any() or (bounds[:,6]<0).any():
        raise ValueError('invalid bound')
    count=len(bounds);occupied=max(1,(count+leaf_size-1)//leaf_size)
    base=1<<(occupied-1).bit_length()
    nodes=np.zeros((2*base,7),dtype=np.float64)
    order=np.full(base*leaf_size,-1,dtype=np.int64)
    centers=(bounds[:,:3]+bounds[:,3:6])/2

    def build(node,ids,leaf_begin,slots):
        if len(ids):
            rows=bounds[ids];nodes[node,:3]=rows[:,:3].min(0)
            nodes[node,3:6]=rows[:,3:6].max(0)
            nodes[node,6]=rows[:,6].max()
        if slots==1:
            if len(ids)>leaf_size:raise RuntimeError('SAH leaf capacity exceeded')
            order[leaf_begin*leaf_size:leaf_begin*leaf_size+len(ids)]=ids
            return
        half=slots//2;capacity=half*leaf_size;n=len(ids)
        if n<=leaf_size:
            build(node*2,ids,leaf_begin,half)
            build(node*2+1,ids[:0],leaf_begin+half,half)
            return
        low=max(1,(n-capacity+leaf_size-1)//leaf_size)
        high=min(half,(n-1)//leaf_size)
        if low>high:raise RuntimeError('SAH split capacity impossible')
        choices=np.unique(np.linspace(low,high,num=min(8,high-low+1),dtype=np.int64))*leaf_size
        best=None
        for axis in range(3):
            permutation=ids[np.argsort(centers[ids,axis],kind='stable')]
            ordered=bounds[permutation]
            for cut in choices:
                left,right=ordered[:cut],ordered[cut:]
                cost=(len(left)*_area(left[:,:3].min(0),left[:,3:6].max(0))
                      +len(right)*_area(right[:,:3].min(0),right[:,3:6].max(0)))
                if best is None or cost<best[0]:best=(cost,permutation,int(cut))
        _,permutation,cut=best
        build(node*2,permutation[:cut],leaf_begin,half)
        build(node*2+1,permutation[cut:],leaf_begin+half,half)

    build(1,np.arange(count,dtype=np.int64),0,base)
    valid=order[order>=0]
    if len(valid)!=count or (count and not np.array_equal(np.sort(valid),np.arange(count))):
        raise RuntimeError('SAH order lost original row IDs')
    return nodes,order,base

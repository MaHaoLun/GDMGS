"""GPU-resident Morton radix BVH, prefix octree, and Morton-bit k-d metadata.

All anchor-derived codes, sorts, topology and support reductions are CUDA
operations. Host code only dispatches kernels and holds shape metadata.
"""
from pathlib import Path
import math
import os
import time
import torch


def native():
    from torch.utils.cpp_extension import load
    path=Path(__file__).parent/'gpu_construct_native'
    build=Path(os.environ['ANCHOR_GPU_BUILD']);build.mkdir(parents=True,exist_ok=True)
    return load(name='anchor_gpu_karras_three_v1',sources=[str(path/x) for x in
        ('bindings.cpp','radix_build.cu','radix_query.cu','octree.cu')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false'],
        build_directory=str(build),verbose=False)


def camera_tensor(view):
    v=view.world_view_transform.T.double().contiguous()
    w,h=int(view.image_width),int(view.image_height)
    fx=float(torch.tensor(w/(2*math.tan(view.FoVx/2)),dtype=torch.float32))
    fy=float(torch.tensor(h/(2*math.tan(view.FoVy/2)),dtype=torch.float32))
    r=v[:3,:3].abs()
    norm=(r.sum(0).max()*r.sum(1).max()*1.0001).reshape(1)
    settings=torch.tensor([fx,fy,w/2,h/2,w,h],dtype=torch.float64,device=v.device)
    return torch.cat((v.reshape(-1),settings,norm)).contiguous()


class GPUThreeTrees:
    def __init__(self,model,leaf_size=32):
        if model.get_anchor.device.type!='cuda' or model.get_color_mlp.training:
            raise ValueError('frozen CUDA inference model required')
        if leaf_size!=32:raise ValueError('v1 qualification requires leaf size 32')
        self.model=model;self.device=model.get_anchor.device;self.leaf_size=leaf_size
        self.ext=native();torch.cuda.synchronize();all_start=time.perf_counter()
        with torch.no_grad():
            scale=model.get_scaling.detach()
            centers=model.get_anchor.detach()[:,None,:]+model._offset.detach()*scale[:,None,:3]
            self.bounds=torch.cat((centers.amin(1),centers.amax(1),scale[:,3:].amax(1,keepdim=True)),1).double().contiguous()
        self.N=int(self.bounds.shape[0]);self.L=(self.N+leaf_size-1)//leaf_size
        if not self.N:raise ValueError('nonempty scene required; empty candidates are supported by query')
        midpoint=(self.bounds[:,:3]+self.bounds[:,3:6])*0.5
        self.scene_lo=midpoint.amin(0).contiguous();self.scene_hi=midpoint.amax(0).contiguous()
        torch.cuda.synchronize();self.preparation_ms=(time.perf_counter()-all_start)*1000
        stage=time.perf_counter()
        keys=self.ext.morton_codes(self.bounds,self.scene_lo,self.scene_hi)
        torch.cuda.synchronize();self.code_ms=(time.perf_counter()-stage)*1000
        stage=time.perf_counter()
        self.sorted_keys=torch.sort(keys).values.contiguous()
        self.sorted_ids=(self.sorted_keys & ((1<<32)-1)).contiguous()
        self.sorted_codes=(self.sorted_keys >> 32).contiguous()
        torch.cuda.synchronize();self.sort_ms=(time.perf_counter()-stage)*1000
        stage=time.perf_counter()
        leaf_slots=torch.arange(self.L,device=self.device,dtype=torch.int64)
        first=self.sorted_codes[leaf_slots*leaf_size]
        self.leaf_keys=((first<<32)|leaf_slots).contiguous()
        self.nodes,self.left,self.right,self.parent,self.axis,self.plane=self.ext.build_radix_bvh(
            self.bounds,self.sorted_ids,self.leaf_keys,self.scene_lo,self.scene_hi,leaf_size)
        torch.cuda.synchronize();self.bvh_build_ms=(time.perf_counter()-stage)*1000
        stage=time.perf_counter()
        prefix_keys=[]
        for depth in range(11):
            prefix=self.sorted_codes>>(30-3*depth)
            prefix_keys.append(((torch.full_like(prefix,depth)<<32)|prefix).contiguous())
        self.oct_keys=torch.unique_consecutive(torch.cat(prefix_keys)).contiguous()
        leaf_cell_key=(torch.full_like(self.sorted_codes,10)<<32)|self.sorted_codes
        leaf_sorted=torch.searchsorted(self.oct_keys,leaf_cell_key).contiguous()
        self.oct_leaf_for_original=torch.empty((self.N,),device=self.device,dtype=torch.int64)
        self.oct_leaf_for_original.scatter_(0,self.sorted_ids,leaf_sorted)
        self.oct_bounds,self.oct_parent,self.oct_children,self.oct_child_count=self.ext.build_octree(
            self.bounds,self.sorted_ids,self.oct_keys,leaf_sorted)
        torch.cuda.synchronize();self.oct_build_ms=(time.perf_counter()-stage)*1000
        self.total_build_ms=(time.perf_counter()-all_start)*1000
        self.bvh_bytes=sum(x.numel()*x.element_size() for x in
            (self.bounds,self.sorted_ids,self.leaf_keys,self.nodes,self.left,self.right,self.parent,self.axis,self.plane))
        self.kd_metadata_bytes=self.axis.numel()*self.axis.element_size()+self.plane.numel()*self.plane.element_size()
        self.oct_bytes=sum(x.numel()*x.element_size() for x in
            (self.bounds,self.oct_keys,self.oct_leaf_for_original,self.oct_bounds,self.oct_parent,self.oct_children,self.oct_child_count))

    def query(self,view,candidate_ids,mode='gpu_bvh',camera=None):
        if candidate_ids.device!=self.device or candidate_ids.dtype!=torch.int64 or candidate_ids.ndim!=1:
            raise ValueError('CUDA int64 source-order candidate IDs required')
        cp=camera_tensor(view) if camera is None else camera
        if mode=='gpu_dense':
            mask=self.ext.gpu_dense_mask(self.bounds,cp);states=None;active=None
        elif mode in ('gpu_bvh','gpu_kd'):
            mask,states,active=self.ext.query_radix(self.bounds,self.nodes,self.left,self.right,
                self.parent,self.sorted_ids,cp,self.axis,self.plane,self.leaf_size,
                mode=='gpu_kd',view.camera_center.double().contiguous())
        elif mode=='gpu_octree':
            mask,states,active=self.ext.query_octree(self.bounds,self.oct_bounds,self.oct_parent,
                self.oct_children,self.oct_child_count,self.oct_leaf_for_original,cp)
        else:raise ValueError(mode)
        return candidate_ids[mask[candidate_ids]],mask,states,active

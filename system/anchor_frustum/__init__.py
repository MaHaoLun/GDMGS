"""Pre-decode spatial selection for frozen ProxyGS + gsplat pinhole inference.

No MLP is evaluated to construct bounds. This is additive to the old depth
predicate, not a replacement for it. It is NOT a general Gaussian renderer API.
"""
from pathlib import Path
import math
import os
import time
import numpy as np
import torch


def native():
    from torch.utils.cpp_extension import load
    root = Path(__file__).parent / "native"
    directory = Path(os.environ["ANCHOR_FRUSTUM_BUILD"])
    directory.mkdir(parents=True, exist_ok=True)
    return load(name="anchor_frustum_v3", sources=[str(root/"bindings.cpp"), str(root/"query.cu")],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--fmad=false"],
                build_directory=str(directory), verbose=False)


def camera_parameters(view):
    v = view.world_view_transform.T.detach().cpu().double().numpy()
    w, h = int(view.image_width), int(view.image_height)
    fx = float(np.float32(w / (2 * math.tan(view.FoVx / 2))))
    fy = float(np.float32(h / (2 * math.tan(view.FoVy / 2))))
    # ||R||_2^2 <= ||R||_1 ||R||_inf. Valid even for nonideal camera matrices.
    r = np.abs(v[:3,:3])
    norm_bound = float(r.sum(0).max() * r.sum(1).max()) * 1.0001
    p = np.r_[v.ravel(), fx, fy, w/2, h/2, w, h, norm_bound]
    if not np.isfinite(p).all() or min(w,h,fx,fy,norm_bound)<=0:
        raise ValueError("invalid pinhole camera")
    return torch.from_numpy(p)


def make_tree(bounds, leaf_size=32):
    if leaf_size < 1 or bounds.ndim != 2 or bounds.shape[1] != 7:
        raise ValueError("invalid bounds or leaf size")
    if not np.isfinite(bounds).all() or (bounds[:,3:6]<bounds[:,:3]).any() or (bounds[:,6]<0).any():
        raise ValueError("nonfinite/inverted support bounds")
    count = len(bounds)
    centers = (bounds[:,:3]+bounds[:,3:6])/2
    if count:
        extent=np.maximum(np.ptp(centers,axis=0),1e-20)
        q=np.clip((centers-centers.min(0))/extent*1023,0,1023).astype(np.uint64)
        code=np.zeros(count,dtype=np.uint64)
        for bit in range(10):
            for axis in range(3): code |= ((q[:,axis]>>bit)&1) << (3*bit+axis)
        order=np.argsort(code,kind="stable").astype(np.int64)
    else: order=np.empty(0,dtype=np.int64)
    base=1 << max(0, ((max(1,count)+leaf_size-1)//leaf_size-1).bit_length())
    # Padding leaves are zero-size distant boxes; no IDs belong to them.
    nodes=np.zeros((2*base,7),dtype=np.float64)
    for i in range(base):
        rows=bounds[order[i*leaf_size:(i+1)*leaf_size]]
        if len(rows):
            nodes[base+i,:3]=rows[:,:3].min(0)
            nodes[base+i,3:6]=rows[:,3:6].max(0)
            nodes[base+i,6]=rows[:,6].max()
    for i in range(base-1,0,-1):
        nodes[i,:3]=np.minimum(nodes[2*i,:3],nodes[2*i+1,:3])
        nodes[i,3:6]=np.maximum(nodes[2*i,3:6],nodes[2*i+1,3:6])
        nodes[i,6]=max(nodes[2*i,6],nodes[2*i+1,6])
    return nodes,order,base


class AnchorFrustumIndex:
    def __init__(self, model, leaf_size=32):
        if model.get_anchor.device.type!="cuda" or model.get_color_mlp.training:
            raise ValueError("frozen CUDA inference model required")
        self.model=model
        self.binding=self._binding()
        self.extension=native()
        torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.no_grad():
            s=model.get_scaling.detach()
            # Same separate float32 multiply/add as the actual decoder.
            centers=model.get_anchor.detach()[:,None,:]+model._offset.detach()*s[:,None,:3]
            b=torch.cat([centers.amin(1),centers.amax(1),s[:,3:].amax(1,keepdim=True)],1)
            values=b.cpu().double().numpy()
        self.nodes,self.order,self.base=make_tree(values,leaf_size)
        self.leaf_size=leaf_size
        self.cpu=(torch.from_numpy(values),torch.from_numpy(self.nodes),torch.from_numpy(self.order))
        self.gpu=tuple(x.to(model.get_anchor.device) for x in self.cpu)
        self.gpu_sorted=self.gpu[0][self.gpu[2]].contiguous()
        inverse=np.empty(len(self.order),dtype=np.int64)
        inverse[self.order]=np.arange(len(self.order),dtype=np.int64)//leaf_size
        self.leaf_for_anchor=torch.from_numpy(inverse).to(model.get_anchor.device)
        torch.cuda.synchronize()
        self.build_upload_ms=(time.perf_counter()-start)*1000
        self.resident_bytes=sum(x.numel()*x.element_size() for x in (*self.gpu,self.gpu_sorted,self.leaf_for_anchor))

    def _binding(self):
        return tuple((x.data_ptr(),x._version,tuple(x.shape)) for x in
                     (self.model._anchor,self.model._offset,self.model._scaling))

    def query(self, view, candidate_ids, mode="gpu_tree", camera=None):
        if self._binding()!=self.binding: raise RuntimeError("model geometry changed; rebuild index")
        if mode not in ("cpu_tree","cpu_dense","gpu_tree","gpu_dense","gpu_sorted","gpu_blocks","cpu_blocks1","cpu_blocks8","cpu_blocks16","cpu_refine8"):
            raise ValueError(mode)
        if candidate_ids.dtype!=torch.int64 or candidate_ids.ndim!=1 or candidate_ids.device!=self.gpu[0].device:
            raise ValueError("candidate IDs must be CUDA int64 source-order IDs")
        cp=camera_parameters(view) if camera is None else camera
        if "blocks" in mode or "refine" in mode:
            if mode.startswith("cpu"):
                flags=self.extension.cpu_blocks(self.cpu[1],cp,self.base,int(mode.replace("cpu_blocks","").replace("cpu_refine","")))
                # Pinned compact flags; transfer and dependent gather remain on
                # the same stream, so no host callback/lifetime race is hidden.
                flags=flags.to(candidate_ids.device,non_blocking=True)
            else:
                flags=self.extension.gpu_blocks(self.gpu[1],cp.to(candidate_ids.device),self.base)
            mask=(self.extension.gpu_refine(self.gpu_sorted,self.gpu[2],cp.to(candidate_ids.device),flags,self.leaf_size)
                  if "refine" in mode else flags[self.leaf_for_anchor])
            counts=torch.tensor([self.base,0,0],dtype=torch.int64)
        elif mode.startswith("cpu"):
            mask,counts=self.extension.cpu_query(*self.cpu,cp,self.base,self.leaf_size,mode.endswith("dense"))
            # Include this mask transfer in CPU query cost; all output IDs stay CUDA.
            mask=mask.to(candidate_ids.device)
        else:
            gpu=(self.gpu_sorted,self.gpu[1],self.gpu[2]) if mode=="gpu_sorted" else self.gpu
            mask,counts=self.extension.gpu_query(*gpu,cp.to(candidate_ids.device),self.base,self.leaf_size,mode.endswith("dense"),mode=="gpu_sorted")
        return candidate_ids[mask[candidate_ids]],mask,counts

    def query_then_depth(self,view,candidate_ids,depth_index,depth,world_view,full_projection,**kwargs):
        """Production handoff: only frustum survivors enter pointwise projection.

        Returns the depth result and the reduced candidate domain. Its canonical
        ranges refer to that domain, never to the original unreduced candidates.
        """
        candidates,_,_=self.query(view,candidate_ids)
        result=depth_index.query(candidates,depth,world_view,full_projection,**kwargs)
        return result,candidates

    def profile_cpu_blocks(self, view, candidate_ids, threads=8):
        """Diagnostic only. Main benchmark separately times full query wall time."""
        torch.cuda.synchronize();wall=time.perf_counter()
        cp=camera_parameters(view);camera_ms=(time.perf_counter()-wall)*1000
        start=time.perf_counter()
        flags=self.extension.cpu_blocks(self.cpu[1],cp,self.base,threads)
        cpu_ms=(time.perf_counter()-start)*1000
        a,b,c=(torch.cuda.Event(enable_timing=True) for _ in range(3))
        transfer_start=time.perf_counter();a.record()
        gpu_flags=flags.to(candidate_ids.device,non_blocking=True);b.record()
        mask=gpu_flags[self.leaf_for_anchor]
        ids=candidate_ids[mask[candidate_ids]];c.record();c.synchronize()
        return dict(cpu_camera_ms=camera_ms,cpu_query_ms=cpu_ms,
                    flags_h2d_bytes=flags.numel()*flags.element_size(),
                    camera_d2h_bytes=16*4,selected_ids_h2d_bytes=0,
                    h2d_event_ms=a.elapsed_time(b),gpu_materialize_event_ms=b.elapsed_time(c),
                    transfer_and_materialize_wall_ms=(time.perf_counter()-transfer_start)*1000,
                    total_wall_ms=(time.perf_counter()-wall)*1000,selected_count=ids.numel())

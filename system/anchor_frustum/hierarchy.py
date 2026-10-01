"""Full GPU active-frontier traversal with six-plane support classification."""
from pathlib import Path
import os
import time
import torch
from . import AnchorFrustumIndex,camera_parameters


def _native(path_name,build_var,module_name):
    from torch.utils.cpp_extension import load
    root=Path(__file__).parent/path_name
    build=Path(os.environ[build_var]);build.mkdir(parents=True,exist_ok=True)
    return load(name=module_name,
        sources=[str(root/'bindings.cpp'),str(root/'frontier.cu')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false'],
        build_directory=str(build),verbose=False)


def native_serial():
    return _native('hierarchy_native_serial','ANCHOR_FRUSTUM_HIER_SERIAL_BUILD','anchor_six_plane_frontier_serial_v1')


def native():
    return _native('hierarchy_native_warp','ANCHOR_FRUSTUM_HIER_WARP_BUILD','anchor_six_plane_frontier_warp_v1')


class HierarchicalAnchorIndex(AnchorFrustumIndex):
    def __init__(self,model,leaf_size=32):
        super().__init__(model,leaf_size)
        self.hierarchy_native=native()
        self.hierarchy_native_serial=native_serial()
        self.sah_gpu=None
        self.sah_build_upload_ms=None
        self.sah_resident_extra_bytes=None

    def build_sah(self):
        from .sah_tree import make_sah_tree
        torch.cuda.synchronize();start=time.perf_counter()
        nodes,order,base=make_sah_tree(self.cpu[0].numpy(),self.leaf_size)
        if base!=self.base:raise RuntimeError('SAH base layout differs from Morton')
        self.sah_order_cpu=order
        self.sah_gpu=(self.gpu[0],torch.from_numpy(nodes).to(self.gpu[0].device),torch.from_numpy(order).to(self.gpu[0].device))
        torch.cuda.synchronize()
        self.sah_build_upload_ms=(time.perf_counter()-start)*1000
        self.sah_resident_extra_bytes=sum(x.numel()*x.element_size() for x in self.sah_gpu[1:])
        return self.sah_build_upload_ms

    def query_frontier(self,view,candidate_ids,camera=None,tree="morton",implementation="warp"):
        if self._binding()!=self.binding:raise RuntimeError('model geometry changed')
        if candidate_ids.dtype!=torch.int64 or candidate_ids.ndim!=1 or candidate_ids.device!=self.gpu[0].device:
            raise ValueError('candidate IDs must be ordered CUDA int64 IDs')
        if tree not in ('morton','sah'):raise ValueError(tree)
        if implementation not in ('warp','serial'):raise ValueError(implementation)
        if tree=='sah' and self.sah_gpu is None:raise RuntimeError('build_sah must run first')
        cp=camera_parameters(view) if camera is None else camera
        gpu=self.gpu if tree=='morton' else self.sah_gpu
        extension=self.hierarchy_native if implementation=="warp" else self.hierarchy_native_serial
        mask,states,active=extension.frontier_query(
            *gpu,cp.to(candidate_ids.device),self.base,self.leaf_size)
        return candidate_ids[mask[candidate_ids]],mask,states,active

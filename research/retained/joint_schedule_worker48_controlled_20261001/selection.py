"""Independent CPU/GPU workers over the frozen combined joint index."""
import copy, ctypes, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from joint import Joint
from hybrid import Hybrid
from gdmgs.anchor_frustum.gpu_construction import camera_tensor
from gdmgs.mesh_index.gpu_index import camera_planes
from gdmgs.mesh_index import MeshIndex

class CView(ctypes.Structure):
    _fields_=[(n,ctypes.c_void_p) for n in ('ab','vertices','an','mn','faces','order','left','right','lo','hi','lod_position','extra','levels')]+[(n,ctypes.c_int64) for n in ('na','nm','nodes')]

class Assets:
    def __init__(self,e,root,scene):
        self.e=e;self.local=threading.local();m=e.rt.model;scale=m.get_scaling.detach()
        centers=m.get_anchor.detach()[:,None,:]+m._offset.detach()*scale[:,None,:3]
        ab=torch.cat((centers.amin(1),centers.amax(1),scale[:,3:].amax(1,keepdim=True)),1).double().contiguous()
        mesh=MeshIndex.load(Path('/ssddata/lun/gdmgs_artifacts')/f'proxygs_step4_cpu_mesh_index_g1_v2_20260914/indices/{scene}/mesh_bvh.npz')
        vs=torch.tensor(mesh.vertices,device='cuda',dtype=torch.float64);fs=torch.tensor(mesh.triangles,device='cuda',dtype=torch.int64)
        tri=vs[fs];mb=torch.cat((tri.amin(1),tri.amax(1),torch.zeros((len(fs),1),device='cuda',dtype=torch.float64)),1).contiguous();del tri
        self.joint=Joint(ab,vs,fs,mb);h=Hybrid(self.joint,20);t=self.joint.tree
        tensors=dict(ab=ab,vertices=vs,an=self.joint.an,mn=self.joint.mn,faces=fs,order=t.order,left=t.left,right=t.right,lo=h.lo,hi=h.hi)
        self.host={name:np.ascontiguousarray(x.detach().cpu().numpy()) for name,x in tensors.items()}
        anchors=m.get_anchor.detach().cpu().numpy().astype(np.float32)
        levels=m._level.detach().cpu().numpy().reshape(-1).astype(np.int32)
        shift=np.float32(float(m.voxel_size)/2)/np.power(np.float32(m.fork),levels,dtype=np.float32)
        self.host.update(lod_position=np.ascontiguousarray(anchors+shift[:,None]),extra=np.ascontiguousarray(m._extra_level.detach().cpu().numpy().reshape(-1),dtype=np.float32),levels=np.ascontiguousarray(levels))
        self.view=CView(*[self.host[name].ctypes.data for name,_ in CView._fields_[:13]],len(ab),len(fs),len(t.parent))
        self.native=ctypes.CDLL(str(root/'cpu/joint_cpu_v2.so'))
        self.native.joint_query.argtypes=[ctypes.POINTER(CView),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_float,ctypes.c_float,ctypes.c_float,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
        self.native.joint_query.restype=ctypes.c_int
        self.native.cpu_lod_keep.argtypes=[ctypes.POINTER(CView),ctypes.c_int64,ctypes.c_void_p,ctypes.c_float,ctypes.c_float,ctypes.c_float,ctypes.c_int,ctypes.c_void_p]
        self.native.cpu_lod_keep.restype=ctypes.c_int
        self.standard=float(m.standard_dist);self.fork=float(m.fork);self.maxlevel=int(m.levels)-1
        self.cameras=[]
        for v,d in zip(e.views,e.rt.domains):
            cp=camera_tensor(v);planes=np.ascontiguousarray(camera_planes(d),dtype=np.float64)
            self.cameras.append(dict(cp=cp,planes_gpu=torch.tensor(planes,device='cuda'),eye_gpu=v.camera_center.double().contiguous(),camera=np.ascontiguousarray(cp.cpu().numpy()),planes=planes,eye=np.ascontiguousarray(v.camera_center.cpu().numpy(),dtype=np.float32),resolution=float(v.resolution_scale)))
        self.cpu_index_bytes=sum(x.nbytes for x in self.host.values());self.gpu_workers=[];self.cpu_pool=None
        self.gpu_pool=None;self.cpu_workers=0;self.gpu_count=0
        torch.cuda.synchronize()
    def configure(self,cpu_workers,gpu_workers):
        if self.cpu_pool:self.cpu_pool.shutdown(wait=True)
        if self.gpu_pool:self.gpu_pool.shutdown(wait=True)
        self.cpu_workers=cpu_workers;self.gpu_count=gpu_workers
        self.cpu_pool=ThreadPoolExecutor(max_workers=cpu_workers)
        self.gpu_pool=ThreadPoolExecutor(max_workers=gpu_workers)
        self.gpu_workers=[GPUWorker(self,i) for i in range(gpu_workers)]
    def cpu(self,f,copy_gpu=True,force_all_lod=False):
        start=time.perf_counter();c=self.cameras[f]
        a=np.empty(self.view.na,np.uint8);m=np.empty(self.view.nm,np.uint8);counts=np.zeros(4,np.int64)
        rc=self.native.joint_query(ctypes.byref(self.view),c['camera'].ctypes.data,c['planes'].ctypes.data,len(c['planes']),c['eye'].ctypes.data,self.standard,self.fork,c['resolution'],self.maxlevel,a.ctypes.data,m.ctypes.data,counts.ctypes.data,0 if force_all_lod else 1)
        if rc:raise RuntimeError(f'joint CPU query {rc}')
        ids=np.flatnonzero(a).astype(np.int64);mesh=np.flatnonzero(m).astype(np.int64)
        query_end=time.perf_counter();host_bytes=a.nbytes+m.nbytes+ids.nbytes+mesh.nbytes
        if copy_gpu:
            with torch.cuda.device(0):
                if not hasattr(self.local,"stream"):self.local.stream=torch.cuda.Stream()
                stream=self.local.stream
                with torch.cuda.stream(stream):
                    pinned=torch.from_numpy(ids).pin_memory();gpu=pinned.to('cuda',non_blocking=True);done=torch.cuda.Event();done.record()
                done.synchronize()
            result=gpu
        else:result=ids
        end=time.perf_counter()
        return result,dict(frame=f,side='cpu',begin=start,end=end,query_ms=(query_end-start)*1000,h2d_ms=(end-query_end)*1000,anchor_count=len(ids),mesh_count=len(mesh),id_bytes=ids.nbytes,host_scratch_bytes=host_bytes+(ids.nbytes if copy_gpu else 0),counters=counts.tolist()),mesh if not copy_gpu else None
    def batch(self,frames,assignment,parallel=True):
        selected={};records=[];begin=time.perf_counter()
        cf=[f for f in frames if assignment[f]=='cpu'];gf=[f for f in frames if assignment[f]=='gpu']
        def cpu_submit():return [self.cpu_pool.submit(self.cpu,f) for f in cf]
        def gpu_submit():
            chunks=[gf[i::self.gpu_count] for i in range(self.gpu_count)]
            return [self.gpu_pool.submit(worker.many,chunk) for worker,chunk in zip(self.gpu_workers,chunks)]
        def collect_cpu(fs):
            for future in fs:
                ids,row,_=future.result();selected[row['frame']]=ids;records.append(row)
        def collect_gpu(fs):
            for future in fs:
                for ids,row in future.result():selected[row['frame']]=ids;records.append(row)
        if parallel:
            cs=cpu_submit();gs=gpu_submit();collect_cpu(cs);collect_gpu(gs)
        else:
            collect_cpu(cpu_submit());collect_gpu(gpu_submit())
        end=time.perf_counter();assert set(selected)==set(frames)
        ends={side:max((r['end'] for r in records if r['side']==side),default=begin) for side in ('cpu','gpu')}
        # CPU scratch includes masks, both typed outputs and pinned staging.
        events=[]
        for r in records:
            if r['side']=='cpu':events.extend([(r['begin'],r['host_scratch_bytes']),(r['end'],-r['host_scratch_bytes'])])
        current=peak=0
        for _,delta in sorted(events):current+=delta;peak=max(peak,current)
        return selected,dict(frames=frames,x_C=len(cf),x_G=len(gf),wall_ms=(end-begin)*1000,begin=begin,end=end,cpu_finish_ms=(ends['cpu']-begin)*1000,gpu_finish_ms=(ends['gpu']-begin)*1000,selected_id_bytes=sum(t.numel()*t.element_size() for t in selected.values()),cpu_scratch_peak_estimated_bytes=peak,cpu_id_h2d_bytes=sum(r['id_bytes'] for r in records if r['side']=='cpu'),records=sorted(records,key=lambda r:r['frame']))
    def close(self):
        if self.cpu_pool:self.cpu_pool.shutdown(wait=True)
        if self.gpu_pool:self.gpu_pool.shutdown(wait=True)

class GPUWorker:
    def __init__(self,assets,index):
        self.a=assets;self.index=index;self.stream=torch.cuda.Stream();self.hybrid=Hybrid(assets.joint,20)
        self.model=copy.copy(assets.e.rt.model)
        for name in ('_anchor_mask','_prog_ratio','transition_mask'):
            x=getattr(self.model,name,None)
            if isinstance(x,torch.Tensor):setattr(self.model,name,x.clone())
        self.initialized=torch.cuda.Event()
        self.initialized.record(torch.cuda.current_stream())
        self.stream.wait_event(self.initialized)
    def one(self,f,return_mesh=False):
        start=time.perf_counter();e=self.a.e;c=self.a.cameras[f];v=e.views[f]
        with torch.no_grad(),torch.cuda.device(0),torch.cuda.stream(self.stream):
            self.model.set_anchor_mask(v.camera_center,40000,v.resolution_scale)
            candidate=torch.nonzero(self.model._anchor_mask).flatten().contiguous()
            ids,mesh,depth,diag=self.hybrid.query(c['cp'],c['planes_gpu'],c['eye_gpu'],candidate,epochs=1,occlusion=False)
            done=torch.cuda.Event();done.record()
        done.synchronize();end=time.perf_counter();assert depth is None
        row=dict(frame=f,side='gpu',worker=self.index,begin=start,end=end,anchor_count=len(ids),mesh_count=len(mesh),id_bytes=ids.numel()*ids.element_size(),stream=int(self.stream.cuda_stream))
        return ids,row,mesh if return_mesh else None
    def many(self,frames):
        rows=[]
        for f in frames:
            ids,row,_=self.one(f);rows.append((ids,row))
        return rows

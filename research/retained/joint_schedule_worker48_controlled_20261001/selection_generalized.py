"""Same query math with isolated submit cores and private reusable host buffers."""
import ctypes,os,threading,time,json,resource
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from selection import Assets,GPUWorker,CView
RESOURCES=json.loads((Path(__file__).resolve().parent/'resources.json').read_text())
assert RESOURCES['state']=='resolved'
PROFILE=os.environ['SCHEDULE_PROFILE']
assert PROFILE=='48w'
GPU_CORES=tuple(RESOURCES['gpu_submit_cores'])
CPU_CORES=tuple(RESOURCES['original_cpu_cores'] if PROFILE=='16w' else RESOURCES['cpu48_options'][RESOURCES['cpu48_choice']])
CPU_PHYSICAL={c:(int(Path('/sys/devices/system/cpu/cpu%d/topology/physical_package_id'%c).read_text()),int(Path('/sys/devices/system/cpu/cpu%d/topology/core_id'%c).read_text())) for c in CPU_CORES}

HELPER_PHYSICAL={(int(Path('/sys/devices/system/cpu/cpu%d/topology/physical_package_id'%c).read_text()),int(Path('/sys/devices/system/cpu/cpu%d/topology/core_id'%c).read_text())) for c in (*GPU_CORES,*RESOURCES['main_cores'])}
assert len(CPU_CORES)==48 and len(set(CPU_CORES))==48
assert len(set(CPU_PHYSICAL.values()))==40 and len(HELPER_PHYSICAL)==8
assert set(CPU_PHYSICAL.values()).isdisjoint(HELPER_PHYSICAL)
assert list(CPU_CORES[:16])==RESOURCES['original_cpu_cores']

class GPUWorkerOpt(GPUWorker):
    def many(self,frames):
        os.sched_setaffinity(0,{GPU_CORES[self.index]})
        result=super().many(frames)
        for _,row in result:row['submit_cpu_affinity']=sorted(os.sched_getaffinity(0))
        return result

class AssetsOpt(Assets):
    def configure(self,cpu_workers=len(CPU_CORES),gpu_workers=4):
        assert gpu_workers in (1,2,4) and cpu_workers<=len(CPU_CORES)
        if self.cpu_pool:self.cpu_pool.shutdown(wait=True)
        if self.gpu_pool:self.gpu_pool.shutdown(wait=True)
        self.local=threading.local();self.pin_lock=threading.Lock();self.pin_next=0;self.buffer_registry={}
        self.cpu_workers=cpu_workers;self.gpu_count=gpu_workers
        def init_cpu():
            with self.pin_lock:index=self.pin_next;self.pin_next+=1
            os.sched_setaffinity(0,{CPU_CORES[index]});self.local.core=CPU_CORES[index]
        self.cpu_pool=ThreadPoolExecutor(max_workers=cpu_workers,initializer=init_cpu)
        self.gpu_pool=ThreadPoolExecutor(max_workers=gpu_workers)
        self.gpu_workers=[GPUWorkerOpt(self,i) for i in range(gpu_workers)]
    def cpu(self,f,copy_gpu=True,force_all_lod=False):
        start=time.perf_counter();c=self.cameras[f]
        if not hasattr(self.local,'am'):
            self.local.am=np.empty(self.view.na,np.uint8);self.local.mm=np.empty(self.view.nm,np.uint8)
            self.local.counts=np.zeros(4,np.int64)
            with self.pin_lock:self.buffer_registry[threading.get_ident()]=self.local.am.nbytes+self.local.mm.nbytes+self.local.counts.nbytes
        a=self.local.am;m=self.local.mm;counts=self.local.counts
        rc=self.native.joint_query(ctypes.byref(self.view),c['camera'].ctypes.data,c['planes'].ctypes.data,len(c['planes']),c['eye'].ctypes.data,self.standard,self.fork,c['resolution'],self.maxlevel,a.ctypes.data,m.ctypes.data,counts.ctypes.data,0 if force_all_lod else 1)
        if rc:raise RuntimeError(rc)
        ids=np.flatnonzero(a).astype(np.int64);mesh=np.flatnonzero(m).astype(np.int64)
        query_end=time.perf_counter();host=a.nbytes+m.nbytes+ids.nbytes+mesh.nbytes
        if copy_gpu:
            with torch.cuda.device(0):
                if not hasattr(self.local,'stream'):
                    self.local.stream=torch.cuda.Stream();self.local.pinned=torch.empty(self.view.na,dtype=torch.int64,pin_memory=True)
                    with self.pin_lock:self.buffer_registry[threading.get_ident()]+=self.local.pinned.numel()*self.local.pinned.element_size()
                pinned=self.local.pinned[:len(ids)];pinned.copy_(torch.from_numpy(ids))
                with torch.cuda.stream(self.local.stream):
                    gpu=pinned.to('cuda',non_blocking=True);done=torch.cuda.Event();done.record()
                done.synchronize()
                host+=self.local.pinned.numel()*self.local.pinned.element_size()
            result=gpu
        else:result=ids
        end=time.perf_counter()
        return result,dict(frame=f,side='cpu',begin=start,end=end,query_ms=(query_end-start)*1000,h2d_ms=(end-query_end)*1000,anchor_count=len(ids),mesh_count=len(mesh),id_bytes=ids.nbytes,host_scratch_bytes=host,counters=counts.tolist(),cpu_core=getattr(self.local,'core',None),cpu_affinity=sorted(os.sched_getaffinity(0))),mesh if not copy_gpu else None
    def batch(self,frames,assignment,parallel=True):
        selected={};records=[];begin=time.perf_counter()
        cf=[f for f in frames if assignment[f]=='cpu'];gf=[f for f in frames if assignment[f]=='gpu']
        def gs():return [self.gpu_pool.submit(w.many,gf[i::self.gpu_count]) for i,w in enumerate(self.gpu_workers)]
        def cs():return [self.cpu_pool.submit(self.cpu,f) for f in cf]
        def gc(fs):
            for future in fs:
                for ids,row in future.result():selected[row['frame']]=ids;records.append(row)
        def cc(fs):
            for future in fs:
                ids,row,_=future.result();selected[row['frame']]=ids;records.append(row)
        if parallel:
            gfuts=gs();cfuts=cs();gc(gfuts);cc(cfuts)
        else:cc(cs());gc(gs())
        end=time.perf_counter();assert set(selected)==set(frames)
        ends={side:max((r['end'] for r in records if r['side']==side),default=begin) for side in ('cpu','gpu')}
        events=[]
        for row in records:
            if row['side']=='cpu':events.extend([(row['begin'],row['host_scratch_bytes']),(row['end'],-row['host_scratch_bytes'])])
        active=peak=0
        for _,delta in sorted(events):active+=delta;peak=max(peak,active)
        with self.pin_lock:owned=sum(self.buffer_registry.values())
        used=sorted({r['cpu_core'] for r in records if r['side']=='cpu'})
        qevents=[]
        for r in records:
            if r['side']=='cpu':qevents.extend([(r['begin'],1),(r['end'],-1)])
        active_q=peak_q=0
        for _,d in sorted(qevents):active_q+=d;peak_q=max(peak_q,active_q)
        return selected,dict(frames=frames,x_C=len(cf),x_G=len(gf),begin=begin,end=end,wall_ms=(end-begin)*1000,cpu_finish_ms=(ends['cpu']-begin)*1000,gpu_finish_ms=(ends['gpu']-begin)*1000,selected_id_bytes=sum(t.numel()*t.element_size() for t in selected.values()),cpu_scratch_peak_estimated_bytes=peak,cpu_owned_persistent_bytes=owned,cpu_cores_used=used,cpu_physical_cores_used=sorted({CPU_PHYSICAL[c] for c in used}),cpu_peak_active_queries=peak_q,process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,host_loadavg=os.getloadavg(),cpu_id_h2d_bytes=sum(row['id_bytes'] for row in records if row['side']=='cpu'),records=sorted(records,key=lambda r:r['frame']))
